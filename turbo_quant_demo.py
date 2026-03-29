"""
TurboQuant KV Cache Quantization Demo — GPT-2 Medium
=====================================================
Faithful implementation of TurboQuant (Zandieh et al., Google Research, 2026)
"Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874)

Algorithm overview (from the paper):
  1. Randomly rotate input vectors via orthogonal matrix Pi (QR of Gaussian)
     -> induces a concentrated Beta distribution on each coordinate
  2. Apply precomputed Lloyd-Max optimal scalar quantizer per coordinate
     -> minimizes MSE (Algorithm 1: TurboQuant_mse)
  3. Compute residual, apply 1-bit QJL (sign(S*r)) to debias inner products
     -> unbiased inner-product estimator (Algorithm 2: TurboQuant_prod)

Companion papers also in this directory:
  - QJL (arXiv:2406.03482): 1-bit Quantized JL Transform
  - PolarQuant (arXiv:2502.02617): KV cache quantization via polar coordinates
"""

import gc
import json
import math
import os
import time
from math import lgamma
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
SEED = 42
MODEL_NAME = "gpt2-medium"
MAX_NEW_TOKENS = 50
QUANT_BITS = 4  # Total: (QUANT_BITS-1) for MSE + 1 for QJL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

TEST_PROMPTS = [
    "The future of artificial intelligence in healthcare will",
    "Quantum computing represents a fundamental shift in",
    "The most significant challenge facing climate science today is",
    "In the field of natural language processing, transformer models have",
]

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------
# Lloyd-Max Codebook Computation (Section 3.1 of the paper)
# ---------------------------------------------------------------
def compute_lloyd_max_codebook(dim: int, n_levels: int,
                                n_iter: int = 300,
                                n_grid: int = 80000) -> np.ndarray:
    """
    Compute Lloyd-Max optimal scalar quantizer centroids for the Beta
    distribution that arises from coordinates of a randomly rotated
    unit vector in R^d (Lemma 1 of the paper):

        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
        for x in [-1, 1]

    Uses the iterative Lloyd-Max algorithm (continuous 1-D k-means)
    on a fine numerical grid.

    Returns:
        centroids: sorted array of n_levels centroid values
    """
    x = np.linspace(-1 + 1e-10, 1 - 1e-10, n_grid)
    dx = x[1] - x[0]

    # Compute Beta PDF (Lemma 1)
    log_const = lgamma(dim / 2) - 0.5 * np.log(np.pi) - lgamma((dim - 1) / 2)
    log_pdf = log_const + ((dim - 3) / 2) * np.log(np.maximum(1 - x ** 2, 1e-30))
    pdf = np.exp(log_pdf)
    pdf = pdf / (np.sum(pdf) * dx)  # normalize to integrate to 1

    # CDF for quantile-based initialization
    cdf = np.cumsum(pdf) * dx
    cdf = cdf / cdf[-1]

    # Initialize centroids at quantile midpoints
    centroids = np.zeros(n_levels)
    for i in range(n_levels):
        target = (i + 0.5) / n_levels
        idx = np.searchsorted(cdf, target)
        centroids[i] = x[min(idx, n_grid - 1)]

    # Lloyd-Max iterations
    for _ in range(n_iter):
        # Decision boundaries = midpoints between consecutive centroids
        bounds = np.concatenate([[x[0]], (centroids[:-1] + centroids[1:]) / 2, [x[-1]]])

        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = bounds[i], bounds[i + 1]
            mask = (x >= lo) & (x <= hi)
            w = pdf[mask]
            if w.sum() > 1e-30:
                new_centroids[i] = np.average(x[mask], weights=w)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    return np.sort(centroids)


# ---------------------------------------------------------------
# TurboQuant_mse -- MSE-optimal vector quantizer (Algorithm 1)
# ---------------------------------------------------------------
class TurboQuantMSE:
    """
    Algorithm 1 from the paper: MSE-optimal vector quantizer.

    Steps:
      1. Store the L2 norm of the input vector
      2. Normalize to unit sphere
      3. Multiply by random rotation matrix Pi (QR decomposition of Gaussian)
      4. Quantize each rotated coordinate using precomputed Lloyd-Max codebook
      5. Dequantize by looking up centroids, rotating back, rescaling by norm
    """

    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits
        self.device = device

        # Generate random rotation matrix Pi via QR decomposition of Gaussian
        # (Section 3.1: "We start by randomizing this vector by multiplying
        #  it with a random rotation matrix Pi in R^{d x d}")
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        G = torch.randn(dim, dim, generator=rng, dtype=torch.float32)
        Q, _ = torch.linalg.qr(G)
        self.rotation = Q.to(device)  # orthogonal matrix, shape (d, d)

        # Compute Lloyd-Max codebook for the exact Beta distribution
        # (Section 3.1, Eq. 4: "we aim to partition [-1,1] into 2^b clusters")
        codebook_np = compute_lloyd_max_codebook(dim, self.n_levels)
        self.codebook = torch.tensor(codebook_np, dtype=torch.float32, device=device)

        # Decision boundaries = midpoints between consecutive centroids
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input vectors.

        Args:
            x: (..., dim) float16 tensor

        Returns:
            indices: (..., dim) int8 -- codebook index per coordinate
            norms:   (..., 1) float32 -- original L2 norms
        """
        x_f32 = x.float()

        # Step 1: Store norms
        norms = torch.norm(x_f32, dim=-1, keepdim=True)

        # Step 2: Normalize to unit sphere (paper assumes ||x||=1)
        x_hat = x_f32 / (norms + 1e-8)

        # Step 3: Random rotation  y = Pi * x_hat
        # (paper line 5: "y <- Pi * x")
        y = x_hat @ self.rotation.T  # (..., dim)

        # Step 4: Nearest centroid per coordinate
        # (paper line 6: "idx_j <- argmin_k |y_j - c_k|")
        indices = torch.bucketize(y, self.boundaries).to(torch.int8)

        return indices, norms

    def dequantize(self, indices: torch.Tensor,
                   norms: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vectors from quantized indices.

        Args:
            indices: (..., dim) int8
            norms:   (..., 1) float32

        Returns:
            x_hat: (..., dim) float16
        """
        # Look up centroids (paper line 9: "y_tilde_j <- c_{idx_j}")
        y_hat = self.codebook[indices.long()]  # (..., dim)

        # Rotate back (paper line 10: "x_tilde <- Pi^T * y_tilde")
        x_hat = y_hat @ self.rotation  # Pi^T = Pi.T for orthogonal, y @ Pi = Pi^T * y

        # Rescale by original norm
        x_hat = x_hat * norms

        return x_hat.half()


# ---------------------------------------------------------------
# TurboQuant_prod -- Inner-product optimal quantizer (Algorithm 2)
# ---------------------------------------------------------------
class TurboQuantProd:
    """
    Algorithm 2 from the paper: inner-product optimal vector quantizer.

    Two-stage approach:
      Stage 1: Apply TurboQuant_mse with (b-1) bits -> minimizes residual L2 norm
      Stage 2: Apply 1-bit QJL (sign(S * r_hat)) on the normalized residual
               -> makes inner-product estimation unbiased

    Dequantization:
      x_tilde = DeQuant_mse(idx) + gamma * sqrt(pi/2)/d * S^T * qjl_signs

    (Definition 1 and Theorem 2 in the paper)
    """

    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device

        # Stage 1: MSE quantizer with (b-1) bits
        # (paper line 2: "Instantiate a TurboQuant_mse with bit-width b-1")
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, device, seed=seed)

        # Stage 2: QJL random projection matrix S in R^{d x d}, S_{ij} ~ N(0,1)
        # (paper line 3: "Generate random projection matrix S")
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed + 10000)  # different seed from rotation
        self.S = torch.randn(dim, dim, generator=rng,
                             dtype=torch.float32).to(device)

        # QJL dequantization scale factor: sqrt(pi/2) / d
        # (Definition 1: Q_qjl^{-1}(z) = sqrt(pi/2)/d * S^T * z)
        self.qjl_scale = math.sqrt(math.pi / 2) / dim

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
        """
        Quantize with (b-1)-bit MSE + 1-bit QJL on residual.

        Args:
            x: (..., dim) float16

        Returns:
            indices:    (..., dim) int8  -- MSE codebook indices [(b-1) bits each]
            norms:      (..., 1) float32 -- original vector norms
            qjl_signs:  (..., dim) int8  -- QJL sign bits {-1, +1}
            gamma:      (..., 1) float32 -- residual norm ||r||
        """
        # Stage 1: MSE quantize with (b-1) bits
        # (paper line 5: "idx <- Quant_mse(x)")
        indices, norms = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(indices, norms).float()

        # Residual (paper line 6: "r <- x - DeQuant_mse(idx)")
        r = x.float() - x_mse

        # Residual norm (paper line 7: "gamma <- ||r||")
        gamma = torch.norm(r, dim=-1, keepdim=True)

        # Normalize residual to unit sphere
        # (paper line 8: "r_hat <- r / gamma")
        r_hat = r / (gamma + 1e-8)

        # Stage 2: QJL -- sign(S * r_hat)
        # (paper line 9: "qjl <- sign(S * r_hat)")
        # In batched form: r_hat @ S.T computes S * r_hat for each vector
        projection = r_hat @ self.S.T  # (..., dim)
        qjl_signs = torch.sign(projection)
        qjl_signs[qjl_signs == 0] = 1  # break ties to {-1, +1}
        qjl_signs = qjl_signs.to(torch.int8)

        return indices, norms, qjl_signs, gamma

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor,
                   qjl_signs: torch.Tensor,
                   gamma: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct with unbiased inner-product estimation.

        x_tilde = DeQuant_mse(idx, norms) + gamma * sqrt(pi/2)/d * S^T * qjl
        (paper line 12 of Algorithm 2)
        """
        # MSE reconstruction
        x_mse = self.mse_quantizer.dequantize(indices, norms).float()

        # QJL reconstruction: gamma * sqrt(pi/2)/d * S^T * z
        # (Definition 1: Q_qjl^{-1}(z) = sqrt(pi/2)/d * S^T * z)
        # In batched form: z @ S computes S^T * z for each vector
        x_qjl = gamma * self.qjl_scale * (qjl_signs.float() @ self.S)

        return (x_mse + x_qjl).half()


# ---------------------------------------------------------------
# TurboQuantKVCache -- per-layer compressed KV storage
# ---------------------------------------------------------------
class TurboQuantKVCache:
    """
    Per-layer compressed KV cache using TurboQuant_prod.

    Each key/value vector (d_head-dimensional) is independently quantized:
      - (b-1) bits for MSE codebook indices per coordinate
      - 1 bit for QJL sign per coordinate
      - 1 float32 for original norm
      - 1 float32 for residual norm (gamma)

    New tokens are quantized individually and appended, avoiding
    compound requantization error on already-compressed entries.
    """

    def __init__(self, num_layers: int, d_head: int, bits: int, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.bits = bits
        self.device = device

        # Separate TurboQuant_prod instances for keys and values per layer
        # (different random matrices for decorrelation)
        self.key_quantizers: List[TurboQuantProd] = []
        self.val_quantizers: List[TurboQuantProd] = []
        for li in range(num_layers):
            self.key_quantizers.append(
                TurboQuantProd(d_head, bits, device, seed=SEED + li * 100))
            self.val_quantizers.append(
                TurboQuantProd(d_head, bits, device, seed=SEED + li * 100 + 50))

        # Cache: list of ((k_indices, k_norms, k_qjl, k_gamma),
        #                  (v_indices, v_norms, v_qjl, v_gamma))
        self.cache: List[Optional[Tuple]] = [None] * num_layers

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store full KV for a layer."""
        k_quant = self.key_quantizers[layer_idx].quantize(key)
        v_quant = self.val_quantizers[layer_idx].quantize(value)
        self.cache[layer_idx] = (k_quant, v_quant)

    def append(self, layer_idx: int, new_key: torch.Tensor,
               new_value: torch.Tensor):
        """Quantize new token's KV and append to existing cache."""
        new_k_q = self.key_quantizers[layer_idx].quantize(new_key)
        new_v_q = self.val_quantizers[layer_idx].quantize(new_value)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k_q, new_v_q)
            return

        old_k_q, old_v_q = self.cache[layer_idx]

        # Concatenate each component along seq dimension (dim=2)
        merged_k = tuple(torch.cat([o, n], dim=2) for o, n in
                         zip(old_k_q, new_k_q))
        merged_v = tuple(torch.cat([o, n], dim=2) for o, n in
                         zip(old_v_q, new_v_q))

        self.cache[layer_idx] = (merged_k, merged_v)

    def retrieve(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return KV."""
        entry = self.cache[layer_idx]
        if entry is None:
            raise ValueError(f"No cache for layer {layer_idx}")

        k_quant, v_quant = entry
        key = self.key_quantizers[layer_idx].dequantize(*k_quant)
        val = self.val_quantizers[layer_idx].dequantize(*v_quant)
        return key, val

    def compressed_size_kb(self) -> float:
        """
        Effective compressed size in KB based on actual bit packing.

        Per vector of dimension d:
          - MSE indices: d * (bits-1) bits
          - QJL signs:   d * 1 bit
          - norm:        32 bits (float32)
          - gamma:       32 bits (float32)
        Total per vector: d * bits + 64 bits
        """
        total_bits = 0
        for entry in self.cache:
            if entry is None:
                continue
            for quant_data in entry:  # k_quant, v_quant
                indices, norms, qjl_signs, gamma = quant_data
                n_elements = indices.numel()     # total coordinates
                n_vectors = norms.numel()        # total vectors
                # (b-1) bits per MSE index + 1 bit per QJL sign = b bits per coord
                total_bits += n_elements * self.bits
                # float32 norms + float32 gamma per vector
                total_bits += n_vectors * 32 * 2
        return total_bits / (8 * 1024)

    def clear(self):
        self.cache = [None] * self.num_layers


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_model_and_tokenizer():
    """Load GPT-2 Medium in float16 on GPU."""
    print(f"Loading {MODEL_NAME} on {DEVICE} ({DTYPE}) ...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {n_params:.1f}M params, "
          f"d_head={model.config.n_embd // model.config.n_head}, "
          f"n_heads={model.config.n_head}, n_layers={model.config.n_layer}")
    return model, tokenizer


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def kv_cache_size_kb(past_key_values) -> float:
    """Size of HuggingFace past_key_values tuple in KB."""
    total_bytes = 0
    if past_key_values is None:
        return 0.0
    for layer_kv in past_key_values:
        for t in layer_kv:
            total_bytes += t.nelement() * t.element_size()
    return total_bytes / 1024.0


# ---------------------------------------------------------------
# Baseline Runner -- standard float16 inference with native KV cache
# ---------------------------------------------------------------
def run_baseline(model, tokenizer, prompt: str) -> Dict:
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    mem_before = get_gpu_memory_mb()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    mem_after = get_gpu_memory_mb()
    gen_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Measure full KV cache size
    with torch.no_grad():
        fwd = model(gen_ids.unsqueeze(0), use_cache=True)
        kv_size = kv_cache_size_kb(fwd.past_key_values)

    elapsed = t1 - t0
    n_tokens = gen_ids.shape[0] - input_ids.shape[1]
    ppl = compute_perplexity(model, tokenizer, generated_text)

    return {
        "tokens_per_sec": n_tokens / elapsed,
        "kv_cache_kb": kv_size,
        "memory_delta_mb": mem_after - mem_before,
        "perplexity": ppl,
        "generated_text": generated_text,
        "elapsed_sec": elapsed,
        "n_tokens": n_tokens,
    }


# ---------------------------------------------------------------
# TurboQuant Runner -- quantized KV cache inference
# ---------------------------------------------------------------
def run_turboquant(model, tokenizer, prompt: str) -> Dict:
    """
    Inference with TurboQuant_prod KV cache quantization.

    Strategy (faithful to paper Section 4.3):
      1. Forward pass on prompt -> get initial KV cache
      2. Quantize initial KV using TurboQuant_prod (Algorithm 2)
      3. For each new token:
         a. Dequantize cached KV
         b. Forward pass with dequantized past + new token
         c. Extract ONLY the new token's KV (avoids compound requantization)
         d. Quantize and append to compressed cache

    APPROXIMATION NOTE: The paper implements custom CUDA kernels and integrates
    quantization inside the attention mechanism. Here we use PyTorch ops and
    HuggingFace's generate API, injecting quantization between forward passes.
    This adds overhead from dequantize/requantize round-trips but correctly
    demonstrates the algorithm's quality characteristics.
    """
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    turbo_cache = TurboQuantKVCache(
        num_layers=num_layers, d_head=d_head,
        bits=QUANT_BITS, device=DEVICE
    )

    mem_before = get_gpu_memory_mb()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generated_ids = input_ids.clone()

    with torch.no_grad():
        # Initial forward pass -- full KV cache for the prompt
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values

        # Quantize and store initial KV cache
        for li in range(num_layers):
            k, v = past_kv[li]  # (batch, heads, prompt_len, d_head)
            turbo_cache.store(li, k, v)

        # First generated token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Generate remaining tokens one by one
        for step in range(MAX_NEW_TOKENS - 1):
            # Dequantize cached KV
            dequant_past = []
            for li in range(num_layers):
                dk, dv = turbo_cache.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)

            # Forward with single new token + dequantized past
            outputs = model(
                next_token,
                past_key_values=dequant_past,
                use_cache=True,
            )
            logits = outputs.logits
            new_past = outputs.past_key_values

            # Extract ONLY the new token's KV and append to compressed cache
            # (avoids compound requantization error on old entries)
            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]  # last position only
                new_v = new_past[li][1][:, :, -1:, :]
                turbo_cache.append(li, new_k, new_v)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    mem_after = get_gpu_memory_mb()
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    elapsed = t1 - t0
    n_tokens = generated_ids.shape[1] - input_ids.shape[1]
    compressed_kb = turbo_cache.compressed_size_kb()
    ppl = compute_perplexity(model, tokenizer, generated_text)

    # Compute what the baseline KV size would be for the same sequence
    with torch.no_grad():
        fwd = model(generated_ids, use_cache=True)
        baseline_kv_kb = kv_cache_size_kb(fwd.past_key_values)

    turbo_cache.clear()

    return {
        "tokens_per_sec": n_tokens / elapsed,
        "kv_cache_kb": compressed_kb,
        "kv_cache_baseline_kb": baseline_kv_kb,
        "memory_delta_mb": mem_after - mem_before,
        "perplexity": ppl,
        "generated_text": generated_text,
        "elapsed_sec": elapsed,
        "n_tokens": n_tokens,
    }


# ---------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------
def generate_html_report(results: Dict, output_path: str):
    baseline = results["baseline_avg"]
    turbo = results["turboquant_avg"]

    compression_ratio = baseline["kv_cache_kb"] / max(turbo["kv_cache_kb"], 0.01)
    ppl_change = ((turbo["perplexity"] - baseline["perplexity"])
                  / baseline["perplexity"]) * 100
    speed_change = ((turbo["tokens_per_sec"] - baseline["tokens_per_sec"])
                    / baseline["tokens_per_sec"]) * 100

    metrics = [
        {
            "title": "Tokens / sec",
            "baseline": baseline["tokens_per_sec"],
            "turbo": turbo["tokens_per_sec"],
            "unit": "tok/s",
            "color_b": "#4a9eff",
            "color_t": "#ff6b6b",
        },
        {
            "title": "KV Cache Size",
            "baseline": baseline["kv_cache_kb"],
            "turbo": turbo["kv_cache_kb"],
            "unit": "KB",
            "color_b": "#4a9eff",
            "color_t": "#50fa7b",
        },
        {
            "title": "GPU Mem Delta",
            "baseline": abs(baseline["memory_delta_mb"]),
            "turbo": abs(turbo["memory_delta_mb"]),
            "unit": "MB",
            "color_b": "#4a9eff",
            "color_t": "#ff79c6",
        },
        {
            "title": "Perplexity",
            "baseline": baseline["perplexity"],
            "turbo": turbo["perplexity"],
            "unit": "",
            "color_b": "#4a9eff",
            "color_t": "#ffb86c",
        },
    ]

    cards_html = ""
    for m in metrics:
        max_val = max(m["baseline"], m["turbo"], 0.01)
        b_pct = (m["baseline"] / max_val) * 100
        t_pct = (m["turbo"] / max_val) * 100

        cards_html += f"""
        <div class="card">
            <h3>{m["title"]}</h3>
            <div class="bar-group">
                <div class="bar-label">Baseline</div>
                <div class="bar-container">
                    <div class="bar" style="width:{b_pct:.1f}%;background:{m["color_b"]};"></div>
                </div>
                <div class="bar-value">{m["baseline"]:.2f} {m["unit"]}</div>
            </div>
            <div class="bar-group">
                <div class="bar-label">TurboQuant</div>
                <div class="bar-container">
                    <div class="bar" style="width:{t_pct:.1f}%;background:{m["color_t"]};"></div>
                </div>
                <div class="bar-value">{m["turbo"]:.2f} {m["unit"]}</div>
            </div>
        </div>"""

    ppl_class = "warn" if ppl_change > 5 else ""
    speed_class = "warn" if speed_change < -20 else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TurboQuant KV Cache Benchmark</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f1117;
    color: #e6e6e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 40px 20px;
    min-height: 100vh;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
  }}
  header {{
    text-align: center;
    margin-bottom: 48px;
  }}
  header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }}
  header .subtitle {{
    font-size: 14px;
    color: #8b8fa3;
    font-weight: 400;
  }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}
  .card {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    transition: border-color 0.2s;
  }}
  .card:hover {{
    border-color: #4a9eff44;
  }}
  .card h3 {{
    font-size: 13px;
    font-weight: 600;
    color: #8b8fa3;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 20px;
  }}
  .bar-group {{
    display: grid;
    grid-template-columns: 85px 1fr 90px;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .bar-label {{
    font-size: 12px;
    color: #a0a4b8;
    font-weight: 500;
  }}
  .bar-container {{
    height: 22px;
    background: #12141d;
    border-radius: 6px;
    overflow: hidden;
  }}
  .bar {{
    height: 100%;
    border-radius: 6px;
    min-width: 4px;
    transition: width 0.6s ease;
  }}
  .bar-value {{
    font-size: 13px;
    font-weight: 600;
    color: #e6e6e6;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .summary {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
  }}
  .summary h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 24px;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 20px;
  }}
  .stat {{
    text-align: center;
  }}
  .stat .value {{
    font-size: 28px;
    font-weight: 700;
    color: #50fa7b;
    display: block;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
  }}
  .stat .value.warn {{
    color: #ffb86c;
  }}
  .stat .value.neutral {{
    color: #4a9eff;
  }}
  .stat .label {{
    font-size: 12px;
    color: #8b8fa3;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .about {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
  }}
  .about h2 {{
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
  }}
  .about p {{
    font-size: 13px;
    color: #a0a4b8;
    line-height: 1.7;
  }}
  .about a {{
    color: #4a9eff;
    text-decoration: none;
  }}
  .approx {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
  }}
  .approx h2 {{
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
  }}
  .approx ul {{
    font-size: 12px;
    color: #8b8fa3;
    line-height: 1.8;
    padding-left: 20px;
  }}
  .footer {{
    text-align: center;
    margin-top: 24px;
    font-size: 12px;
    color: #555;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>TurboQuant KV Cache Benchmark &mdash; GPT-2 Medium</h1>
    <div class="subtitle">{QUANT_BITS}-bit TurboQuant&bull;prod &middot; ({QUANT_BITS - 1})-bit Lloyd-Max MSE + 1-bit QJL &middot; {MAX_NEW_TOKENS} tokens &middot; {DEVICE.upper()}</div>
  </header>

  <div class="cards">
    {cards_html}
  </div>

  <div class="summary">
    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="stat">
        <span class="value">{compression_ratio:.1f}x</span>
        <span class="label">Compression Ratio</span>
      </div>
      <div class="stat">
        <span class="value {ppl_class}">{ppl_change:+.1f}%</span>
        <span class="label">Perplexity Change</span>
      </div>
      <div class="stat">
        <span class="value {speed_class}">{speed_change:+.1f}%</span>
        <span class="label">Speed Change</span>
      </div>
      <div class="stat">
        <span class="value neutral">{MODEL_NAME}</span>
        <span class="label">Model</span>
      </div>
      <div class="stat">
        <span class="value neutral">{DEVICE.upper()}</span>
        <span class="label">Device</span>
      </div>
      <div class="stat">
        <span class="value neutral">{QUANT_BITS}-bit</span>
        <span class="label">Quantization</span>
      </div>
    </div>
  </div>

  <div class="about">
    <h2>About This Implementation</h2>
    <p>
      <strong>TurboQuant</strong> (Zandieh, Daliri, Hadian &amp; Mirrokni, Google Research, 2026)
      is a data-oblivious vector quantization algorithm that achieves near-optimal
      distortion rates within a &asymp;2.7&times; factor of the information-theoretic lower bound.
      It works by randomly rotating input vectors via an orthogonal matrix &Pi;
      (QR decomposition of a Gaussian matrix)
      to induce a concentrated Beta distribution on each coordinate, then applying
      precomputed Lloyd-Max optimal scalar quantizers per coordinate.
      For unbiased inner-product estimation &mdash; critical for attention score accuracy &mdash;
      TurboQuant combines the MSE-optimal quantizer with a 1-bit Quantized
      Johnson&ndash;Lindenstrauss (QJL) transform on the residual.
      The paper demonstrates quality neutrality at 3.5 bits per channel and
      marginal degradation at 2.5 bits, with over 4&times; KV cache compression
      on Llama-3.1-8B and Ministral-7B.
    </p>
    <p style="margin-top:8px;font-size:12px;color:#666;">
      Paper: <a href="https://arxiv.org/abs/2504.19874">arXiv:2504.19874</a> &middot;
      QJL: <a href="https://arxiv.org/abs/2406.03482">arXiv:2406.03482</a> &middot;
      PolarQuant: <a href="https://arxiv.org/abs/2502.02617">arXiv:2502.02617</a>
    </p>
  </div>

  <div class="approx">
    <h2>Approximations vs. the Paper</h2>
    <ul>
      <li>GPT-2 Medium has d_head=64; the paper targets d_head=128 (Llama/Ministral). At d=64 the Beta distribution is still well-concentrated.</li>
      <li>No outlier channel separation (paper assigns different bit-widths to outlier vs. non-outlier channels based on calibration).</li>
      <li>No custom CUDA kernels &mdash; all quantization uses PyTorch ops applied between forward passes, not fused inside attention.</li>
      <li>Codebook is computed via numerical Lloyd-Max on the exact Beta PDF with d=64 (not the Gaussian approximation).</li>
    </ul>
  </div>

  <div class="footer">
    Generated by TurboQuant Demo &middot; Seed {SEED} &middot; torch {torch.__version__}
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved to {output_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print("=" * 70)
    print("  TurboQuant KV Cache Quantization Demo -- GPT-2 Medium")
    print("  Paper: arXiv:2504.19874 (Zandieh et al., Google Research, 2026)")
    print("=" * 70)
    print(f"  Device       : {DEVICE}"
          f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Dtype        : {DTYPE}")
    print(f"  Quant bits   : {QUANT_BITS} "
          f"({QUANT_BITS - 1}-bit Lloyd-Max MSE + 1-bit QJL)")
    print(f"  Max tokens   : {MAX_NEW_TOKENS}")
    print(f"  Seed         : {SEED}")
    print(f"  Prompts      : {len(TEST_PROMPTS)}")
    print()

    # APPROXIMATION: GPT-2 Medium has d_head=64. The paper targets d_head=128
    # (Llama/Ministral). At d=64 the Beta distribution is still well-concentrated
    # and close to Gaussian, so the algorithm works correctly. The codebook is
    # computed for the exact Beta distribution with d=64, not the Gaussian approx.
    d_head = 64  # GPT-2 Medium: n_embd=1024, n_head=16 -> d_head=64
    n_mse_bits = QUANT_BITS - 1
    n_levels = 2 ** n_mse_bits
    print(f"Computing Lloyd-Max codebook for d={d_head}, "
          f"{n_mse_bits}-bit ({n_levels} levels) ...")
    codebook = compute_lloyd_max_codebook(d_head, n_levels)
    print(f"  Centroids: {np.array2string(codebook, precision=6)}")
    print()

    model, tokenizer = load_model_and_tokenizer()

    # Warmup
    print("Warmup run ...")
    warmup_ids = tokenizer.encode("Hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model.generate(warmup_ids, max_new_tokens=5, use_cache=True)
    torch.cuda.synchronize()
    print()

    baseline_results = []
    turbo_results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[Prompt {i + 1}/{len(TEST_PROMPTS)}] {prompt[:60]}...")

        print("  Running baseline ...", end=" ", flush=True)
        b = run_baseline(model, tokenizer, prompt)
        print(f"done ({b['tokens_per_sec']:.1f} tok/s)")
        baseline_results.append(b)

        torch.cuda.empty_cache()
        gc.collect()

        print("  Running TurboQuant ...", end=" ", flush=True)
        t = run_turboquant(model, tokenizer, prompt)
        print(f"done ({t['tokens_per_sec']:.1f} tok/s)")
        turbo_results.append(t)

        torch.cuda.empty_cache()
        gc.collect()
        print()

    # -- Aggregate --
    def avg_metrics(results_list):
        keys = ["tokens_per_sec", "kv_cache_kb", "memory_delta_mb", "perplexity"]
        return {k: float(np.mean([r[k] for r in results_list])) for k in keys}

    baseline_avg = avg_metrics(baseline_results)
    turbo_avg = avg_metrics(turbo_results)

    # -- Print Table --
    print("=" * 70)
    print("  RESULTS (averaged over {} prompts)".format(len(TEST_PROMPTS)))
    print("=" * 70)

    table_data = {
        "Metric": ["Tokens/sec", "KV Cache (KB)", "GPU Mem Delta (MB)",
                    "Perplexity"],
        "Baseline": [
            f"{baseline_avg['tokens_per_sec']:.2f}",
            f"{baseline_avg['kv_cache_kb']:.2f}",
            f"{abs(baseline_avg['memory_delta_mb']):.2f}",
            f"{baseline_avg['perplexity']:.2f}",
        ],
        "TurboQuant": [
            f"{turbo_avg['tokens_per_sec']:.2f}",
            f"{turbo_avg['kv_cache_kb']:.2f}",
            f"{abs(turbo_avg['memory_delta_mb']):.2f}",
            f"{turbo_avg['perplexity']:.2f}",
        ],
    }
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print()

    compression = baseline_avg["kv_cache_kb"] / max(turbo_avg["kv_cache_kb"], 0.01)
    ppl_delta = ((turbo_avg["perplexity"] - baseline_avg["perplexity"])
                 / baseline_avg["perplexity"]) * 100
    speed_delta = ((turbo_avg["tokens_per_sec"] - baseline_avg["tokens_per_sec"])
                   / baseline_avg["tokens_per_sec"]) * 100

    print(f"  Compression ratio : {compression:.1f}x")
    print(f"  Perplexity change : {ppl_delta:+.1f}%")
    print(f"  Speed change      : {speed_delta:+.1f}%")
    print()

    # -- Save results.json --
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results.json")

    all_results = {
        "config": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "CPU"),
            "dtype": str(DTYPE),
            "quant_bits": QUANT_BITS,
            "quant_scheme": (f"{QUANT_BITS - 1}-bit Lloyd-Max MSE "
                             f"+ 1-bit QJL residual"),
            "d_head": d_head,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "torch_version": torch.__version__,
            "num_prompts": len(TEST_PROMPTS),
            "paper": "arXiv:2504.19874",
        },
        "baseline_avg": {k: round(v, 4) for k, v in baseline_avg.items()},
        "turboquant_avg": {k: round(v, 4) for k, v in turbo_avg.items()},
        "summary": {
            "compression_ratio": round(compression, 2),
            "perplexity_change_pct": round(ppl_delta, 2),
            "speed_change_pct": round(speed_delta, 2),
        },
        "codebook": {
            "dim": d_head,
            "mse_bits": n_mse_bits,
            "n_levels": n_levels,
            "centroids": codebook.tolist(),
        },
        "per_prompt": {
            "baseline": [
                {
                    "prompt": TEST_PROMPTS[i],
                    "tokens_per_sec": round(r["tokens_per_sec"], 4),
                    "kv_cache_kb": round(r["kv_cache_kb"], 4),
                    "memory_delta_mb": round(r["memory_delta_mb"], 4),
                    "perplexity": round(r["perplexity"], 4),
                    "generated_text": r["generated_text"],
                }
                for i, r in enumerate(baseline_results)
            ],
            "turboquant": [
                {
                    "prompt": TEST_PROMPTS[i],
                    "tokens_per_sec": round(r["tokens_per_sec"], 4),
                    "kv_cache_kb": round(r["kv_cache_kb"], 4),
                    "memory_delta_mb": round(r["memory_delta_mb"], 4),
                    "perplexity": round(r["perplexity"], 4),
                    "generated_text": r["generated_text"],
                }
                for i, r in enumerate(turbo_results)
            ],
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # -- Generate HTML report --
    html_path = os.path.join(script_dir, "results.html")
    generate_html_report(all_results, html_path)

    print()
    print("Done!")


if __name__ == "__main__":
    main()

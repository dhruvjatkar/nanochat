# Mano: Restriking Manifold Optimization for LLM Training

**Paper:** [arXiv:2601.23000](https://arxiv.org/abs/2601.23000)
**Authors:** Yufei Gu, Zeke Xie (xLeaF Lab, HKUST Guangzhou)
**Venue:** ICML 2026 submission

---

## Core Idea

Mano (**MA**nifold-**N**ormalized **O**ptimizer) reformulates manifold optimization for LLM training. Instead of the expensive Newton-Schulz / Polar Express matrix orthogonalization that Muon uses, Mano projects the momentum onto the tangent space of the Oblique manifold and normalizes it -- using only cheap element-wise and column/row-wise operations, no MatMul. A "rotational manifold" scheme alternates between column-wise and row-wise normalization at each step to avoid privileging one dimension.

The key departure from classical manifold optimization: **parameters stay in Euclidean space** (no retraction onto the manifold). Only the momentum/update direction is projected and normalized onto the manifold surface. This "soft manifold constraint" provides the geometric benefits (escape from local minima, spectral regularization) without restricting parameter expressivity.

## Algorithm (Algorithm 1)

Given weight `θ_t ∈ R^{m×n}`, momentum `M_t`, learning rate `η_t`, momentum coefficient `μ`, weight decay `λ`:

```
Initialize M_0 = 0
for each step:
    g_t = ∇f(θ_t)                           # gradient
    M_t = μ * M_{t-1} + g_t                  # momentum accumulation
    k = t mod 2                               # rotating manifold dimension
    θ̂_t = θ_t / ‖θ_t‖_{2,k}                  # normalize params along dim k
    v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k          # tangent space projection
    v̂_t = v_t / ‖v_t‖_{2,k}                   # normalize update along dim k
    θ_{t+1} = θ_t - η_t * (0.2√n_k * v̂_t + λ * θ_t)  # update with rescaling
```

Where:
- `‖·‖_{2,k}` is the norm along dimension k (column-wise when k=0, row-wise when k=1)
- `⟨·,·⟩_k` is the inner product along dimension k
- `⊙` is element-wise multiplication
- `n_k ∈ {m, n}` with n_0=m, n_1=n (for rescaling to match AdamW update RMS range)

**Oblique manifold**: The set of matrices with unit-norm columns. Projecting onto its tangent space removes the component parallel to the normalized parameters, keeping only the perpendicular component.

## Key Design Choices

1. **Rotational manifold scheme**: Alternates column-wise (k=0) and row-wise (k=1) normalization each step. This avoids the assumption that one dimension dominates. The paper notes this is related to Sinkhorn-Knopp iteration, which converges toward doubly stochastic matrices. Ablation shows static manifold (k=0 always) scales poorly to larger models.

2. **No retraction on parameters**: Unlike classical manifold optimization, parameters are NOT constrained to lie on the manifold. Only the update step is projected/normalized. This is critical -- standard Riemannian SGD-M on the Oblique manifold completely fails to train LLMs (loss > 6.0 vs Mano's ~2.5).

3. **Momentum not retracted**: The momentum buffer stores standard SGD-M momentum, not tangent momentum. Ablation shows `M_t = v_t` (retracting momentum onto tangent space) gives essentially identical results.

4. **Update RMS calibration**: The factor `0.2√n_k` ensures the update RMS is in the range [0.2, 0.4], matching AdamW's scale. This follows the same convention as Muon (Liu et al., 2025).

5. **Embedding and output layers use AdamW**: Same as Muon -- the structural properties of input/output layers (sparse vocabulary activations) make per-parameter adaptive rates more appropriate.

## Experimental Results

### Sample Efficiency (10K steps)

| Dataset | Model | AdamW | Muon | Mano |
|---------|-------|-------|------|------|
| C4      | LLaMA-350M | 23.85 | 22.49 | **21.18** |
| C4      | LLaMA-1.3B | 19.69 | 18.37 | **17.80** |
| Pile    | LLaMA-350M | 11.80 | 11.02 | **10.55** |
| Pile    | LLaMA-1.3B | 9.95 | 9.23 | **8.99** |
| Pile    | Qwen3-0.6B | 15.68 | 14.02 | **13.69** |
| Pile    | Qwen3-1.7B | 13.62 | 12.28 | **12.03** |

Mano consistently outperforms both Muon and AdamW across all tested model/dataset combinations.

### Convergence Pattern

Mano exhibits a distinctive convergence pattern:
- **Early training**: Mano converges slightly slower than Muon
- **Later training**: Mano's loss descent rate overtakes both AdamW and Muon, which begin to plateau
- This "late-stage acceleration" effect is hypothesized to come from better escape from local minima via the manifold-projected updates

### Wall-clock Time

This is where Mano really shines over Muon:

| Module | Metric | Newton-Schulz (Muon) | Mano |
|--------|--------|---------------------|------|
| LLaMA-1B Attn | Time | 2.01 ms | **0.14 ms** (14x faster) |
| LLaMA-1B MLP | Time | 4.68 ms | **0.17 ms** (28x faster) |
| LLaMA-7B Attn | Time | 14.83 ms | **0.34 ms** (44x faster) |
| LLaMA-7B MLP | Time | 30.22 ms | **1.45 ms** (21x faster) |
| LLaMA-70B Attn | Time | 110.79 ms | **2.19 ms** (51x faster) |
| LLaMA-70B MLP | Time | 184.33 ms | **4.35 ms** (42x faster) |

Mano's overhead is **linear** in model dimension (only element-wise ops), while Muon/Newton-Schulz grows **super-linearly** (MatMul). In a one-day pretraining experiment:
- LLaMA-350M: Mano achieves **1.75x** Muon's convergence speed in wall-clock time
- LLaMA-1.3B: Mano achieves **1.38x** Muon's convergence speed in wall-clock time

### Memory

Mano uses the same memory as SGD-M (1 momentum buffer per parameter), which is:
- **Same** as Muon
- **Half** the memory of AdamW (which needs 2 buffers: exp_avg + exp_avg_sq)

### Learning Dynamics

- **Lower gradient variance** than Muon (same momentum coefficient μ=0.95)
- **Higher SNR** (signal-to-noise ratio) than Muon
- **Spectral regularization**: Mano lifts the update spectrum (increasing relative magnitude of rare/small directions) while **preserving singular value ordering** -- unlike Muon which flattens the spectrum entirely. This connects to Su (2025)'s theoretical argument that discarding singular order information can be suboptimal.

### Convergence Theory

For a simplified version (no momentum, static Oblique manifold), Mano achieves:

```
min_{t∈[0,T]} E[‖∇f(θ_t)‖] ≤ O(Lm^{3/2} / (γ√T))
```

under standard L-smoothness assumptions and the condition that gradient is never perfectly aligned with parameters (sin(φ) ≥ γ > 0).

## Hyperparameters

The paper uses the same hyperparameters as AdamW/Muon (no additional tuning):
- Momentum coefficient: μ = 0.95 (same as Muon)
- Weight decay: λ = 0.1
- Learning rate: same schedule as AdamW/Muon (cosine decay, min ratio 0.1)
- Gradient clipping: 1.0
- Warmup: 1000 steps
- No additional hyperparameters beyond standard SGD-M + weight decay

---

## Relevance to nanochat

### Direct Connection: nanochat's Optimizer Architecture

nanochat's `MuonAdamW` optimizer in `nanochat/optim.py` uses:
1. **AdamW** for embeddings, lm_head, and scalar/1D parameters
2. **Muon** (via Polar Express) for all 2D matrix parameters (attention Q/K/V/O, MLP)

The Muon step in `muon_step_fused` currently does:
1. Nesterov momentum (lerp-based)
2. **Polar Express orthogonalization** (5 iterations of `X.mT @ X`, `X @ B` MatMul chains)
3. NorMuon variance reduction (per-neuron adaptive scaling)
4. Cautious weight decay + update

Mano would replace steps 2-3 with:
1. **Column/row-wise normalization of parameters** (just `θ / ‖θ‖_{2,k}`)
2. **Tangent space projection** (just `M - θ̂ ⊙ ⟨M, θ̂⟩_k`, all element-wise)
3. **Column/row-wise normalization of update** (just `v / ‖v‖_{2,k}`)

This eliminates ALL MatMul operations from the optimizer step.

### What a nanochat Implementation Would Look Like

The Mano update is strikingly simple. A fused kernel would be something like:

```python
@torch.compile(dynamic=False, fullgraph=True)
def mano_step_fused(
    stacked_grads,      # (K, m, n)
    stacked_params,     # (K, m, n)
    momentum_buffer,    # (K, m, n)
    momentum_t,         # () scalar
    lr_t,               # () scalar
    wd_t,               # () scalar
    step_t,             # () int - for rotational manifold
) -> None:
    # Standard momentum
    mu = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mu)
    M = stacked_grads.lerp_(momentum_buffer, mu)  # Nesterov

    # Rotational manifold: alternate column/row normalization
    k = int(step_t.item()) % 2
    dim = -2 if k == 0 else -1  # column-wise or row-wise
    n_k = stacked_params.size(dim)

    # Normalize parameters along dim k
    theta_hat = stacked_params / stacked_params.norm(dim=dim, keepdim=True).clamp(min=1e-8)

    # Tangent space projection: remove component parallel to θ̂
    dot = (M * theta_hat).sum(dim=dim, keepdim=True)
    v = M - theta_hat * dot

    # Normalize update along dim k
    v_hat = v / v.norm(dim=dim, keepdim=True).clamp(min=1e-8)

    # Update with rescaling
    lr = lr_t.to(v_hat.dtype)
    wd = wd_t.to(v_hat.dtype)
    scale = 0.2 * (n_k ** 0.5)
    stacked_params.sub_(lr * (scale * v_hat + wd * stacked_params))
```

### Key Advantages for nanochat

1. **Massive compute savings**: The Polar Express step dominates Muon's per-step cost. Mano replaces it with normalizations (reductions) and element-wise ops. For nanochat's typical matrix sizes (768x3072, 768x768), this could save significant wall-clock time per step.

2. **No MatMul in optimizer**: The Polar Express currently involves repeated `X.mT @ X` and `X @ B` operations in bfloat16. These are the same expensive operations that dominate the forward/backward pass. Eliminating them from the optimizer is a pure win for throughput.

3. **Simpler code**: The Mano kernel is much shorter and simpler than the Polar Express + variance reduction pipeline. Easier to maintain, debug, and fuse.

4. **Same memory footprint**: Mano uses a single momentum buffer, same as the current Muon setup. The NorMuon `second_momentum_buffer` could potentially be dropped.

5. **Better spectral properties**: Mano preserves singular value ordering while still regularizing the spectrum. This addresses a theoretical concern about Muon (flat spectrum discards curvature info).

### Considerations and Risks

1. **Early convergence may be slower**: The paper consistently shows Mano starts slower than Muon but catches up and overtakes. For nanochat's speed-benchmark setting (often short training runs), this initial lag could matter. However, the wall-clock advantage from faster steps may compensate.

2. **Interaction with NorMuon**: nanochat uses NorMuon (variance reduction) on top of Muon. Mano doesn't use any variance reduction. It's unclear whether adding NorMuon-style variance reduction on top of Mano would help or hurt.

3. **Interaction with Cautious updates**: nanochat uses cautious masking (`mask = (g * params) >= 0`). The paper doesn't mention cautious updates. This could be an orthogonal improvement to try on top of Mano.

4. **No Nesterov in the paper**: Mano uses standard momentum (not Nesterov), though the authors note NAG is available as an option. nanochat's current Muon uses Nesterov. Worth testing both.

5. **Scale validation**: The paper's largest model is 1.7B params. nanochat targets much wider scale ranges. The rotational manifold scheme was shown to be important for scaling -- static manifold degraded at 1.3B. Need to verify this holds at nanochat's target scales.

### Relationship to Other Optimizer Papers in nanochat Knowledge

**vs FISMO** (summary_fismo_fisher_structured_muon.md):
- FISMO adds Fisher-structured preconditioners P, Q before orthogonalization (more expensive)
- Mano removes orthogonalization entirely and replaces it with normalization (cheaper)
- Both argue Muon's isotropic spectrum is suboptimal, but from different angles: FISMO adds curvature info, Mano preserves singular value ordering via monotone transformation
- Mano is much cheaper and simpler than FISMO

**vs TEON** (summary_teon_tensor_cross_layer_muon.md):
- TEON generalizes Muon to cross-layer (tensor-wise) orthogonalization
- Mano replaces orthogonalization with manifold normalization
- These could potentially be combined: stack layers (TEON-style) then apply Mano normalization instead of Polar Express

### Ideas for Experimentation

1. **Direct Mano implementation**: Add a `mano_step_fused` kernel to `nanochat/optim.py` and benchmark against the current `muon_step_fused`. This is straightforward -- the algorithm is simple and the paper provides complete pseudocode.

2. **Mano + NorMuon hybrid**: Apply NorMuon variance reduction after the manifold normalization step. The per-neuron scaling might complement the manifold projection.

3. **Mano + Cautious updates**: Keep nanochat's cautious masking on top of Mano updates.

4. **Mano with Nesterov momentum**: The paper uses standard momentum but notes Nesterov as an option. Test both variants.

5. **Benchmark the compute savings**: The biggest potential win is wall-clock time. Profile `mano_step_fused` vs `muon_step_fused` on nanochat's actual parameter shapes to quantify the speedup.

6. **Cross-layer Mano (Mano + TEON)**: Stack same-shape parameters (like TEON does) and apply Mano normalization along the third dimension as well. This could capture cross-layer structure without the MatMul cost.

7. **Spectral monitoring**: Log the singular value distribution of Muon vs Mano updates during nanochat training to verify the paper's claims about spectrum preservation.

### Key Takeaway

Mano's central proposition for nanochat is radical simplification: **replace the entire Polar Express orthogonalization pipeline with cheap column/row normalizations and tangent projections**, while achieving equal or better convergence. The theoretical FLOP overhead is `O(mn)` vs Muon's `O(m²n)` or `O(mn²)`, and empirically the normalization ops are 14-50x faster than Newton-Schulz iterations. Combined with the paper's consistent sample-efficiency gains across models and datasets, Mano is a compelling candidate for nanochat's optimizer -- especially given nanochat's focus on training speed.

The main risk is early-training convergence speed and whether the results generalize to nanochat's specific architecture (which has many optimizations like GQA, U-net skip connections, paired heads, etc. that might interact with the optimizer differently).

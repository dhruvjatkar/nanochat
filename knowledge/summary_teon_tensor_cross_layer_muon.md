# TEON: Tensorized Orthonormalization Beyond Layer-Wise MUON

**Paper:** [arXiv:2601.23261v2](https://arxiv.org/abs/2601.23261v2)
**Authors:** Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Dongyang Li, Yupeng Su, Sijia Liu, Zheng Zhang (UC Santa Barbara, Michigan State University)
**Venue:** ICML 2026 submission (preprint Feb 3, 2026)

---

## Core Idea

TEON generalizes Muon from **layer-wise** orthogonalization to **tensor-wise** (cross-layer) orthogonalization. Instead of independently orthogonalizing each layer's gradient, TEON stacks gradients from multiple layers of the same type into a 3D tensor and orthogonalizes the tensor as a whole. This captures **cross-layer correlations** in the gradient structure that layer-wise Muon ignores.

The key analogy: just as Adam uses a diagonal approximation of the Fisher matrix (ignoring inter-parameter correlations), Muon uses a block-diagonal approximation across layers (ignoring inter-layer correlations). TEON fills in some of those off-diagonal blocks by jointly orthogonalizing grouped layers.

## Algorithm (TEON mode-1, the recommended variant)

Given K layers with gradient matrices G^(1), ..., G^(K), each in R^{m x n}:

1. **Stack** gradients into an order-3 tensor: G_tensor in R^{m x n x K}, where G_tensor[:,:,k] = G^(k)
2. **Momentum** accumulation on the stacked tensor (same as Muon)
3. **Mode-1 matricization**: unfold tensor along mode 1 to get Z in R^{m x (nK)} -- this horizontally concatenates the K gradient matrices
4. **Orthogonalize** Z via Polar Express / Newton-Schulz / exact SVD to get Q = Ortho(Z)
5. **Fold back** Q into the original tensor shape to get per-layer updates
6. **Update** each layer's parameters with its slice of the orthogonalized tensor

The critical difference from Muon: step 3-5 perform a **single joint** orthogonalization across K layers rather than K independent ones.

## Practical Configuration (from theory + ablation)

- **Which layers to stack**: Only Q, K, V attention projections (same type across consecutive transformer layers). NOT MLP layers.
- **How many layers to stack (K)**: K=2 works best. Larger K degrades because singular vector alignment weakens.
- **Matricization mode**: Mode-1 (unfold along rows). In transformers, the top right singular vectors (v_1) of QKV momentum are strongly aligned across layers, while top left singular vectors (u_1) are nearly orthogonal. Mode-1 exploits the right-vector alignment.

## Theoretical Results

### Convergence Bound (Theorem 4.5)

Under standard smoothness assumptions with the Non-Euclidean Trust Region (NTR) framework:

- TEON convergence: ||nabla f||_{TEON,*} <= sqrt(2 * L_TEON * Delta_0 / T)
- Muon convergence: ||nabla f||_{TEON,*} <= sqrt(2 * L_MUON * Delta_0 / T)
- Key relationship: L_TEON <= L_MUON <= K * L_TEON

In the best case (when gradient singular vectors are well-aligned across stacked layers), TEON achieves up to **sqrt(K)x better convergence** than Muon. With K=2, that's ~1.41x improvement.

### Maximal Gain Condition (Proposition 4.6)

The sqrt(K) gain is approached when:
- Mode-1: top **right** singular vectors v_1^(k) are aligned across layers
- Mode-2: top **left** singular vectors u_1^(k) are aligned across layers

In practice, QKV attention matrices exhibit strong right-singular-vector alignment, which is why mode-1 + QKV stacking is the recommended configuration.

### Norm Definitions

- **Muon norm**: ||X||_MUON = max_k ||X^(k)||_op (layer-wise max operator norm)
- **TEON-i norm**: ||X||_TEON-i = ||M_i(X)||_op (operator norm of mode-i matricization)
- These satisfy: ||X||_MUON <= ||X||_TEON-1 <= sqrt(K) * ||X||_MUON

The TEON norm is strictly tighter than the Muon norm, leading to potentially smaller smoothness constants and better convergence.

## Experimental Results

### GPT-2 Pre-training (FineWeb, 10B tokens)

| Model     | AdamW | Muon (PE) | TEON (PE) |
|-----------|-------|-----------|-----------|
| GPT-Small | 32.84 | 28.53     | **27.12** |
| GPT-Base  | 29.33 | 21.64     | **20.92** |
| GPT-Large | 27.31 | 19.26     | **18.73** |

(PE = PolarExpress orthogonalization, 5 iterations)

TEON improves over Muon by 1.4-1.5 PPL on GPT-Small and ~0.5-0.7 PPL on larger models. Consistent across all three SVD approximation methods tested (You, Jordan, PolarExpress).

### LLaMA Pre-training (FineWeb, compute-optimal tokens)

| Model | AdamW | Muon (PE) | TEON (PE) |
|-------|-------|-----------|-----------|
| 60M   | 33.10 | 26.13     | **25.62** |
| 130M  | 23.64 | 19.45     | **18.92** |
| 350M  | 16.18 | 14.11     | **13.80** |
| 1B    | 14.38 | 11.19     | **10.84** |

Gains are consistent across all scales, with TEON always beating Muon.

### Variance (5 runs, GPT-Small)

| Method     | Mean PPL | Variance |
|------------|----------|----------|
| TEON (PE)  | 27.15    | 0.0020   |
| Muon (PE)  | 28.57    | 0.0009   |

The gain is real and well outside the noise margin (1.4 PPL gap vs ~0.05 std dev).

## Key Ablation Findings

1. **Mode-1 > Mode-2** (PPL 34.29 vs 34.54 with exact SVD) -- consistent with right-singular-vector alignment
2. **QKV-only stacking is best** -- adding O, MLP1, or MLP2 to the stack hurts. QKV gradients share a retrieval-oriented functional role; MLP gradients store heterogeneous memorization patterns.
3. **K=2 is optimal** -- K=4 slightly worse, K=12 noticeably worse. Alignment degrades with more layers stacked.
4. **Robust across SVD methods** -- PolarExpress > exact SVD > Jordan/You. TEON improves over Muon regardless of the orthogonalization method.

---

## Relevance to nanochat

### Direct Connection: nanochat Already Uses Muon with Polar Express

nanochat's optimizer (`nanochat/optim.py`) already implements exactly the Muon + Polar Express setup that TEON builds on. The `muon_step_fused` function performs:
1. Nesterov momentum
2. Polar Express orthogonalization (5 iterations)
3. NorMuon variance reduction
4. Cautious weight decay + update

Parameters are already **stacked by shape** for efficient batch orthogonalization (line 257-258 in `optim.py`):
```python
stacked_grads = torch.stack([p.grad for p in params])
stacked_params = torch.stack(params)
```

This stacking is currently done for params with identical shapes for computational efficiency, but **not** to exploit cross-layer correlations a la TEON.

### What TEON Would Change in nanochat

#### Current Architecture (from `gpt.py`)

nanochat's attention uses separate Q, K, V projections:
```python
self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)      # (768, 768) 
self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)    # (768, 768) with GQA
self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)    # (768, 768) with GQA
self.c_proj = nn.Linear(n_embd, n_embd, bias=False)               # (768, 768)
```

And MLP:
```python
self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)             # (768, 3072)
self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)           # (3072, 768)
```

#### Current Parameter Grouping

In `setup_optimizer` (line 468-475), all matrix params from `transformer.h` are grouped **by shape**:
```python
for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(kind='muon', params=group_params, ...))
```

This means all (768, 768) params (c_q, c_k, c_v, c_proj across all layers) end up in one group, and all (768, 3072) and (3072, 768) params (MLP layers) in their respective groups.

The stacking in `_step_muon` then stacks all same-shape params together and runs a single Polar Express pass on each layer independently (the stacking dimension is treated as a batch dimension, not a tensor mode).

#### What TEON Would Require

To implement TEON, we would need to change the **semantics** of the stacking:

1. **Identify QKV params by type and layer**: Instead of just grouping by shape, group c_q from layer i with c_q from layer i+1 (and similarly for c_k, c_v) into pairs of K=2.

2. **Mode-1 matricization before orthogonalization**: For a pair of (768, 768) Q matrices from consecutive layers, form a (768, 768*2) = (768, 1536) matrix by horizontal concatenation, then apply Polar Express to this wider matrix.

3. **Fold back and apply updates**: After orthogonalization, split the (768, 1536) result back into two (768, 768) updates for each layer.

### Implementation Strategy

#### Option A: Minimal Change -- Reshape Within `muon_step_fused`

The simplest approach would be to:
1. In `setup_optimizer`, create TEON groups that pair consecutive-layer QKV params
2. In the fused kernel, reshape stacked_grads from (2, 768, 768) to (768, 1536) before Polar Express, then reshape back

The Polar Express code already handles arbitrary rectangular matrices, so no changes needed there. The key change is just the reshape before/after.

#### Option B: New TEON Optimizer Step

Create a `teon_step_fused` that:
1. Takes pairs of gradient tensors
2. Concatenates along mode-1 (columns)
3. Runs Polar Express on the concatenated matrix
4. Splits the result back
5. Applies NorMuon variance reduction per-layer (not across the concatenated matrix)
6. Applies updates

#### Practical Considerations

**GQA complication**: With GQA (`n_kv_head != n_head`), c_q has shape (768, 768) but c_k and c_v might have shape (768, 384) if n_kv_head = 3. TEON requires stacking same-shape matrices, so we'd need to group Q-from-layer-i with Q-from-layer-i+1, K-from-layer-i with K-from-layer-i+1, etc. -- not mix Q with K.

**Distributed training**: The `DistMuonAdamW` optimizer shards across ranks by dividing the stacked params. TEON grouping would need to ensure that paired layers land on the same rank for the joint orthogonalization, or the communication pattern would need to change.

**PolarExpress on wider matrices**: Concatenating two (768, 768) matrices gives a (768, 1536) matrix with aspect ratio 1:2. The paper shows PolarExpress works well at this ratio (their ablations use it). However, nanochat's existing Polar Express code already handles wide matrices differently from tall ones (lines 117-126 in optim.py), so this should work out of the box.

**Interaction with NorMuon variance reduction**: After the joint orthogonalization, the variance reduction step should probably be applied per-layer (not on the concatenated matrix), since the per-neuron statistics are layer-specific.

### Comparison with FISMO

Both TEON and FISMO (the other paper in nanochat's knowledge base) aim to improve upon Muon, but from different angles:

- **FISMO**: Adds anisotropic curvature via Fisher preconditioners P, Q (within each layer). Changes the metric of the trust region.
- **TEON**: Captures cross-layer correlations via tensor stacking. Keeps the isotropic metric but enlarges the optimization scope.

These are **orthogonal improvements** and could potentially be combined: apply FISMO's preconditioners to each layer's gradient, then stack and jointly orthogonalize via TEON.

### Estimated Implementation Effort

TEON is significantly easier to implement than FISMO:
- No new optimizer state (no preconditioner matrices)
- No matrix inverse square roots
- Just reshaping before/after the existing Polar Express step
- Memory cost: negligible (temporary concatenation buffer)
- Compute cost: nearly identical per-step (slightly wider matrix to orthogonalize, but fewer total orthogonalizations)

### Recommended Experiment Plan

1. **Quick prototype**: Modify `setup_optimizer` to create TEON-style groups that pair consecutive-layer Q params (K=2). Reshape in `muon_step_fused` before Polar Express. Measure PPL impact.

2. **Ablate layer types**: Test Q-only vs K-only vs V-only vs QKV stacking. The paper finds QKV-all best, but nanochat's architecture (with GQA, value embeddings, paired heads, etc.) may differ.

3. **Ablate K**: Try K=2 (recommended) and K=3 (possible since nanochat has 12 layers = 6 pairs or 4 triples).

4. **Monitor singular vector alignment**: Log the inner product of top singular vectors across consecutive layers during training to verify the theoretical conditions hold for nanochat's specific architecture.

5. **Combine with NorMuon**: Test whether variance reduction should be applied before or after the joint orthogonalization, and whether it should be per-layer or per-tensor.

### Key Takeaway

TEON is a **low-cost, architecture-compatible** improvement over Muon that nanochat can adopt with minimal code changes. The core insight -- that QKV attention gradients across consecutive transformer layers share aligned singular vector structure -- is well-motivated theoretically and empirically. Given that nanochat already stacks same-shape params for batched orthogonalization, the change to TEON-style cross-layer stacking is essentially a reshape operation. The expected gain is ~1-2 PPL points for a GPT-Small-scale model, with negligible computational overhead.

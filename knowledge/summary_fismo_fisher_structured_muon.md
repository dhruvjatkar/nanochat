# FISMO: Fisher-Structured Momentum-Orthogonalized Optimizer

**Paper:** [arXiv:2601.21750](https://arxiv.org/abs/2601.21750)
**Authors:** Chenrui Xu, Wenjing Yan, Ying-Jun Angela Zhang (CUHK)
**Venue:** ICML 2026 submission

---

## Core Idea

FISMO generalizes Muon's isotropic orthogonalized updates by incorporating **anisotropic curvature information** derived from the Fisher Information Matrix (FIM). Where Muon enforces uniform singular values (all = 1) via polar decomposition, FISMO maintains a degree of curvature-aware anisotropy that preserves informative spectral variation in the gradient.

The key insight: Muon's strict isotropy (condition number kappa = 1) discards potentially valuable curvature information encoded in the gradient's singular value spectrum. FISMO operates in an "optimal conditioning trade-off" regime where kappa ~ 10^2-10^3, which is:
- Orders of magnitude better conditioned than Adam (kappa > 10^8)
- But not fully isotropic like Muon (kappa = 1)

## Algorithm (Algorithm 1)

Given matrix parameter W in R^{m x n}, learning rate eta, momentum beta, EMA decay gamma, damping mu:

1. **Compute gradient** G_t (mini-batch average)
2. **Update left preconditioner P_t:**
   - L_t = (1/n) G_t Q_{t-1}^{-1} G_t^T + mu * (tr(P_{t-1})/m) * I_m
   - P_tilde_t = gamma * P_{t-1} + (1-gamma) * L_t  (EMA smoothing)
   - P_t = sym(m / tr(P_tilde_t) * P_tilde_t)  (trace normalization)
3. **Update right preconditioner Q_t** (same structure, using updated P_t -- Gauss-Seidel ordering):
   - R_t = (1/m) G_t^T P_t^{-1} G_t + mu * (tr(Q_{t-1})/n) * I_n
   - Q_tilde_t = gamma * Q_{t-1} + (1-gamma) * R_t
   - Q_t = sym(n / tr(Q_tilde_t) * Q_tilde_t)
4. **Whiten gradient:** G_tilde_t = P_t^{-1/2} G_t Q_t^{-1/2}
5. **Momentum in whitened space:** M_t = beta * M_{t-1} + (1-beta) * G_tilde_t
6. **Polar decomposition in whitened space:** Delta_W_t = P_t^{-1/2} Polar(M_t) Q_t^{-1/2}
7. **Update:** W_t = W_{t-1} - eta * Delta_W_t

## Theoretical Framework

### From Muon to FISMO

- **Muon** solves: min <G, Delta_W> s.t. ||Delta_W||_2 <= eta
  - Solution: Delta_W* = -eta * Polar(G) = -eta * U V^T
  - This is isotropic: all singular values of the update are equal

- **FISMO** solves: min <G, Delta_W> s.t. ||P^{1/2} Delta_W Q^{1/2}||_2 <= eta
  - Where P, Q are Kronecker factors approximating the Fisher Information Matrix: F_W ~ Q otimes P
  - Solution: Delta_W* = -eta * P^{-1/2} Polar(P^{-1/2} G Q^{-1/2}) Q^{-1/2}
  - The preconditioners P and Q inject curvature-aware anisotropy

### Optimal Kronecker Approximation (Theorem 1)

The best P and Q (in log-det divergence sense) satisfy coupled fixed-point equations:
- P*(Q) = (1/n) E[G Q^{-1} G^T] + (mu * tr(Q^{-1})/n) I_m
- Q*(P) = (1/m) E[G^T P^{-1} G] + (mu * tr(P^{-1})/m) I_n

These are computed via Gauss-Seidel iteration (update P first using old Q, then Q using new P).

### Convergence (Theorem 3)

FISMO achieves O(1/sqrt(T)) convergence rate for the expected squared gradient norm in stochastic nonconvex settings:

(1/T) sum E[||nabla L(W_{t-1})||_*] = O((R + rL + G*) / sqrt(T) + G*/T + sigma*sqrt(r)/sqrt(B))

This matches Muon's rate (up to constants). The sigma*sqrt(r)/sqrt(B) term captures irreducible variance from mini-batch sampling.

## Key Design Choices

1. **Gauss-Seidel updates** for P and Q (sequential, not simultaneous) -- better conditioning, less memory
2. **EMA smoothing** of preconditioners with decay gamma for stability
3. **Identity regularization** (damping mu) to keep preconditioners positive definite
4. **Trace normalization** to eliminate scale ambiguity between P and Q
5. **Newton-Schulz iterations** for approximate polar decomposition (same as Muon)
6. **Momentum in whitened space** (not original space)

## Experimental Results

### Language Modeling (GPT-2 124M on OpenWebText via NanoGPT)
- FISMO achieves lowest training and validation loss, beating Muon, Shampoo, AdamW, and SGD
- Consistent ordering between train/val suggests the gains are from better optimization, not overfitting

### Image Classification (SimpleDLA on CIFAR-10)
- FISMO achieves highest train/test accuracy with smoothest validation trajectory
- Notably smoother convergence than Muon, suggesting better stability

### Condition Number Analysis
- Adam: kappa > 10^8 (pathological)
- FISMO: kappa ~ 10^2-10^3 (structured anisotropy)
- Muon (NS=5): kappa ~ 1-10 (approximate isotropy)
- Muon (ideal): kappa = 1 (perfect isotropy)

This aligns with the "Isotropic Curvature Model" theory that optimal updates require partial (not complete) spectrum homogenization.

---

## Relevance to nanochat

### Direct Connection: Muon is nanochat's Core Optimizer

nanochat already uses Muon (via Polar Express variant) for all 2D matrix parameters in the transformer (attention Q/K/V/proj, MLP layers), combined with AdamW for embeddings and scalars. FISMO is a direct generalization of exactly this setup.

### What FISMO Would Change in nanochat

The current `muon_step_fused` in `nanochat/optim.py` performs:
1. Nesterov momentum
2. Polar Express orthogonalization (5 iterations)
3. NorMuon variance reduction
4. Cautious weight decay + update

FISMO would replace this with:
1. Maintain and update left/right preconditioners P, Q per layer
2. Whiten gradient: G_tilde = P^{-1/2} G Q^{-1/2}
3. Momentum accumulation in whitened space
4. Polar Express orthogonalization of whitened momentum
5. Transform back: Delta_W = P^{-1/2} Polar(M) Q^{-1/2}
6. Parameter update

### Practical Considerations for Implementation

**Computational cost of preconditioners:**
- P is m x m, Q is n x n (for a weight matrix W in R^{m x n})
- For nanochat's typical layers (e.g., 768 x 3072), P is 768x768 and Q is 3072x3072
- Requires computing P^{-1/2} and Q^{-1/2} at each step -- this involves matrix inverse square root
- The paper uses full eigendecomposition/Cholesky for these, which is expensive
- Could potentially approximate P^{-1/2} and Q^{-1/2} with Newton-Schulz iterations too

**Memory overhead:**
- Need to store P_t, Q_t per Muon group (or per unique shape)
- For nanochat's stacked parameter approach, these could be shared across layers with the same shape
- P: m x m matrix, Q: n x n matrix per shape group
- Additional memory for matrix inverses during the step

**Interaction with existing NorMuon variance reduction:**
- FISMO's preconditioners P and Q already provide per-direction scaling
- NorMuon's per-neuron variance reduction may be redundant or could complement FISMO
- Would need ablation to determine if both are needed

### Ideas for Experimentation

1. **Direct FISMO implementation:** Replace the Muon step with the full FISMO algorithm. The main challenge is the cost of P^{-1/2} and Q^{-1/2} computation at every step.

2. **Lightweight FISMO variant:** Update P and Q less frequently (e.g., every K steps) to amortize the cost of matrix inverse square roots. Between updates, use the cached P^{-1/2} and Q^{-1/2}.

3. **Diagonal FISMO:** Use diagonal approximations of P and Q (keeping only diagonal entries). This reduces to per-row and per-column scaling of the gradient before orthogonalization -- much cheaper and still captures some anisotropy. This is somewhat analogous to what NorMuon already does, but principled.

4. **FISMO without full inverse square root:** Instead of P^{-1/2}, approximate using a few Newton-Schulz iterations (similar to how Polar Express approximates the polar decomposition). This could be fused into the existing compiled kernel.

5. **Condition number monitoring:** Add logging of the condition number of Muon's updates in nanochat to empirically verify whether the current isotropic updates are suboptimal for nanochat's specific architecture and data.

6. **Hybrid approach:** Use FISMO for attention layers (where curvature structure may matter more) and standard Muon for MLP layers (where isotropy may be sufficient), or vice versa.

### Key Takeaway

FISMO's central message for nanochat is that **Muon's strict isotropy may be leaving performance on the table**. The gradient's singular value spectrum contains curvature information that uniform normalization discards. By incorporating even a lightweight form of Fisher-structured preconditioning before orthogonalization, nanochat could potentially achieve faster convergence and better final loss -- at the cost of some additional computation and memory for maintaining the preconditioners P and Q.

The practical question is whether the overhead of maintaining and applying P and Q (matrix inversions, square roots) can be kept small enough relative to the training step to be worthwhile, especially in nanochat's speed-optimized single-node setting. A diagonal or infrequently-updated variant seems like the most promising starting point.

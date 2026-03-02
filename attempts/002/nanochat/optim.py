"""
A nice and efficient mixed AdamW/Muon/Hyperball Combined Optimizer (attempt-local override).

Changes from root version:
- Item 5:  Cautious weight decay for AdamW groups
           Source: modded-nanogpt record #43 PR #154 by @varunneal
- Item 9:  Hyperball optimizer (norm projection replaces WD for matrix params)
           Source: nanochat PR #498 + modded-nanogpt record #45 PR #160 by @KellerJordan
- Item 11: Flipped MLP LR multiplier for Muon
           Source: nanochat PR #492 by @chrisjmccormick
- Item 13: Chebyshev-optimal Polar Express coefficients
           Source: Polar Express arXiv:2409.20325 + CANS (ICLR 2026)
- Item 14: Mantissa tracking for BF16 parameter precision
           Source: modded-nanogpt record #57 PR #190 by @classiclarryd
- Item 22: Heterogeneous batch sizes (AdamW every other step)
           Source: modded-nanogpt record #39 PR #136 by @classiclarryd
- Item 27: Parameter reshaping for shared reduce_scatter
           Source: modded-nanogpt record #36 PR #132 by @classiclarryd
- Item 28: Explicit communication ordering
           Source: modded-nanogpt records #22-#24 by @KonstantinWilleke, @alexrgilbert, @ryanyang0, @vagrawal

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

# Source: modded-nanogpt record #43 PR #154 — cautious weight decay for AdamW (item 5)
# Standard decoupled WD is replaced with cautious WD: only decay weights in the
# same direction as the update, preventing decay from undoing the optimizer's work.
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step with cautious weight decay (item 5).
    """
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    update = exp_avg / denom * step_size
    # Source: modded-nanogpt record #43 PR #154 — cautious weight decay (item 5)
    # Only apply WD in directions where the update and parameter agree in sign.
    # This prevents weight decay from fighting the optimizer's intended update direction.
    mask = (update * p) > 0
    p.sub_(update)
    p.sub_(p * mask * (lr_t * wd_t))

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932

NorMuon variance reduction:
https://arxiv.org/pdf/2510.05491
"""

# Source: Polar Express arXiv:2409.20325 + CANS ICLR 2026 — Chebyshev-optimal coefficients (item 13)
# These replace the original heuristic coefficients with Chebyshev-optimal values
# that converge faster in fewer iterations, from the CANS paper.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def _nesterov_momentum(stacked_grads: Tensor, momentum_buffer: Tensor, momentum_t: Tensor) -> Tensor:
    """Nesterov momentum. Modifies momentum_buffer and stacked_grads in-place. Returns g."""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    return stacked_grads.lerp_(momentum_buffer, momentum)


def _polar_express(g: Tensor, ns_steps: int) -> Tensor:
    """Polar Express orthogonalization with Chebyshev-optimal coefficients (item 13). Returns orthogonalized g."""
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def _normuon_variance(g: Tensor, second_momentum_buffer: Tensor, beta2_t: Tensor, red_dim: int) -> Tensor:
    """NorMuon per-neuron variance reduction. Modifies buffer in-place. Returns scaled g."""
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    return g * final_scale.to(g.dtype)


def _nesterov_polar_variance(
    stacked_grads: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> Tensor:
    """
    Shared pipeline: Nesterov momentum -> Polar Express -> NorMuon variance reduction.
    Called from within @torch.compile functions (will be traced and inlined).

    Modifies momentum_buffer and second_momentum_buffer in-place.
    Returns the orthogonalized, variance-reduced update direction g.
    """
    g = _nesterov_momentum(stacked_grads, momentum_buffer, momentum_t)
    g = _polar_express(g, ns_steps)
    g = _normuon_variance(g, second_momentum_buffer, beta2_t, red_dim)
    return g


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update"""
    g = _nesterov_polar_variance(
        stacked_grads, momentum_buffer, second_momentum_buffer,
        momentum_t, beta2_t, ns_steps, red_dim,
    )
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Source: nanochat PR #498 + modded-nanogpt record #45 PR #160 — Hyperball optimizer (item 9)
# After the Muon update, project weights back to their initial Frobenius norm.
# This replaces cautious weight decay for matrix params — weights live on a hypersphere.
@torch.compile(dynamic=False, fullgraph=True)
def hyperball_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
    cached_norms: Tensor,  # (K, 1, 1) - initial Frobenius norms per param
) -> None:
    """
    Hyperball step: like Muon but replaces weight decay with norm projection.
    After the update, each parameter is rescaled to its initial Frobenius norm.
    """
    g = _nesterov_polar_variance(
        stacked_grads, momentum_buffer, second_momentum_buffer,
        momentum_t, beta2_t, ns_steps, red_dim,
    )
    # Update params (no weight decay — Hyperball uses norm projection instead)
    lr = lr_t.to(g.dtype)
    stacked_params.sub_(lr * g)

    # Source: nanochat PR #498 — Hyperball norm projection (item 9)
    # Project each parameter back to its initial Frobenius norm (hypersphere constraint).
    current_norms = stacked_params.norm(dim=(-2, -1), keepdim=True)
    stacked_params.mul_(cached_norms / current_norms.clamp_min(1e-10))


# -----------------------------------------------------------------------------
"""
TEON (Tensorized Orthonormalization) optimizer kernel.
Generalizes Muon by stacking gradients from paired consecutive layers (K=2)
into a concatenated matrix and orthogonalizing them jointly, capturing
cross-layer correlations for improved sample efficiency.
Source: TEON paper (Cross-Layer Muon)
"""

@torch.compile(dynamic=False, fullgraph=True)
def teon_step_fused(
    stacked_grads_concat: Tensor,      # (N_pairs, m, 2n) - mode-1 concatenated
    stacked_params_concat: Tensor,     # (N_pairs, m, 2n)
    momentum_buffer_concat: Tensor,    # (N_pairs, m, 2n)
    second_momentum_buffer: Tensor,    # (2*N_pairs, m, 1) or (2*N_pairs, 1, n) - per-layer
    momentum_t: Tensor,                # () - 0-D CPU tensor
    lr_t: Tensor,                      # () - 0-D CPU tensor
    wd_t: Tensor,                      # () - 0-D CPU tensor
    beta2_t: Tensor,                   # () - 0-D CPU tensor
    ns_steps: int,
    red_dim: int,
    n_orig: int,                       # original column dimension (for split)
) -> None:
    """TEON step: joint momentum + polar express on concatenated pairs, per-layer NorMuon + update."""
    # 1. Nesterov momentum on concatenated tensors (N_pairs, m, 2n)
    g = _nesterov_momentum(stacked_grads_concat, momentum_buffer_concat, momentum_t)
    # 2. Polar Express on concatenated (uses wide path: A = X @ X.mT is (m,m) — same cost as square)
    g = _polar_express(g, ns_steps)
    # 3. Unconcatenate: (N_pairs, m, 2n) -> (2*N_pairs, m, n) for per-layer processing
    n_pairs = g.size(0)
    g_layers = torch.cat([g[..., :n_orig], g[..., n_orig:]], dim=0)  # (2*N_pairs, m, n)
    # 4. NorMuon variance reduction per-layer
    g_layers = _normuon_variance(g_layers, second_momentum_buffer, beta2_t, red_dim)
    # 5. Re-split processed gradients
    g_even = g_layers[:n_pairs]   # (N_pairs, m, n)
    g_odd = g_layers[n_pairs:]    # (N_pairs, m, n)
    # 6. Cautious WD + update, writing directly into stacked_params_concat views
    lr = lr_t.to(g_even.dtype)
    wd = wd_t.to(g_even.dtype)
    p_even = stacked_params_concat[..., :n_orig]
    mask_even = (g_even * p_even) >= 0
    p_even.sub_(lr * g_even + lr * wd * p_even * mask_even)
    p_odd = stacked_params_concat[..., n_orig:]
    mask_odd = (g_odd * p_odd) >= 0
    p_odd.sub_(lr * g_odd + lr * wd * p_odd * mask_odd)


# Source: modded-nanogpt record #57 PR #190 — mantissa tracking for BF16 precision (item 14)
# Reconstructs FP32 from BF16 param + stored uint16 mantissa bits. Saves 2 bytes/param
# vs full FP32 master weights while preserving training stability.
@torch.compile(dynamic=False, fullgraph=True)
def mantissa_reconstruct_fp32(p_bf16: Tensor, mantissa: Tensor) -> Tensor:
    """Reconstruct FP32 from BF16 parameter + uint16 mantissa bits."""
    p_uint32 = p_bf16.view(torch.uint16).to(torch.uint32) << 16
    p_fp32 = (p_uint32 | mantissa.to(torch.uint32)).view(torch.float32)
    return p_fp32


@torch.compile(dynamic=False, fullgraph=True)
def mantissa_split_bf16(p_fp32: Tensor) -> tuple:
    """Split FP32 into BF16 param + uint16 mantissa bits."""
    p_uint32 = p_fp32.view(torch.uint32)
    p_bf16 = (p_uint32 >> 16).to(torch.uint16).view(torch.bfloat16)
    mantissa = (p_uint32 & 0xFFFF).to(torch.uint16)
    return p_bf16, mantissa


# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon/Hyperball for 2D matrix params, AdamW for others.
    Single GPU version. See DistMuonAdamW for multi-GPU.

    Supports:
    - Hyperball mode for matrix params (item 9)
    - Cautious AdamW weight decay (item 5)
    - Flipped MLP LR multiplier (item 11)
    - Mantissa tracking for BF16 (item 14)
    - Heterogeneous batch sizes — skip Adam on even steps (item 22)
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        # Source: nanochat PR #492 — flipped MLP LR multiplier (item 11)
        # Corrects Muon LR scaling for non-square matrices: tall matrices (M>N) get no boost,
        # wide matrices (M<N) get scaled by (N/M)^0.5
        ratio = shape[-2] / shape[-1]
        lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
        self._muon_lr_t.fill_(group["lr"] * lr_mult)
        self._muon_wd_t.fill_(group["weight_decay"])

        use_hyperball = group.get("use_hyperball", False)

        if use_hyperball:
            # Source: nanochat PR #498 + modded-nanogpt #45 PR #160 — Hyperball (item 9)
            if "cached_norms" not in state:
                state["cached_norms"] = stacked_params.norm(dim=(-2, -1), keepdim=True).clone()
            hyperball_step_fused(
                stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                self._muon_momentum_t, self._muon_lr_t, self._muon_beta2_t,
                group["ns_steps"], red_dim, state["cached_norms"],
            )
        else:
            muon_step_fused(
                stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_teon(self, group: dict) -> None:
        """TEON step: consecutive params in the flat list form pairs for cross-layer orthogonalization."""
        params: list[Tensor] = group['params']
        if not params:
            return
        num_pairs = len(params) // 2
        p0 = params[0]
        m, n = p0.shape
        device, dtype = p0.device, p0.dtype

        state = self.state[p0]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_pairs, m, 2 * n, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            # Per-layer NorMuon buffer: 2*N_pairs entries (each concat pair splits into 2 layers)
            if m >= n:
                state["second_momentum_buffer"] = torch.zeros(2 * num_pairs, m, 1, dtype=dtype, device=device)
            else:
                state["second_momentum_buffer"] = torch.zeros(2 * num_pairs, 1, n, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if m >= n else -2

        # Mode-1 concatenation: stack pairs along columns
        stacked_grads_concat = torch.stack([
            torch.cat([params[2*i].grad, params[2*i+1].grad], dim=-1) for i in range(num_pairs)
        ])  # (num_pairs, m, 2n)
        stacked_params_concat = torch.stack([
            torch.cat([params[2*i], params[2*i+1]], dim=-1) for i in range(num_pairs)
        ])  # (num_pairs, m, 2n)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        # LR scaling uses original per-layer shape, not concatenated
        ratio = m / n
        lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
        self._muon_lr_t.fill_(group["lr"] * lr_mult)
        self._muon_wd_t.fill_(group["weight_decay"])

        teon_step_fused(
            stacked_grads_concat, stacked_params_concat,
            momentum_buffer, second_momentum_buffer,
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim, n,
        )

        # Copy updated params back (unconcatenate from stacked_params_concat)
        for i in range(num_pairs):
            params[2*i].copy_(stacked_params_concat[i, :, :n])
            params[2*i+1].copy_(stacked_params_concat[i, :, n:])

    @torch.no_grad()
    def step(self, do_adam=True):
        """
        Args:
            do_adam: if False, skip AdamW groups (item 22 — heterogeneous batch sizes).
                     Muon/TEON groups always step.
        """
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                # Source: modded-nanogpt record #39 PR #136 — skip Adam on even steps (item 22)
                if do_adam:
                    self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'teon':
                self._step_teon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon/Hyperball for 2D matrix params, AdamW for others.

    Supports all items from MuonAdamW plus:
    - Item 28: Explicit communication ordering for better NCCL overlap
    - Item 27: Parameter reshaping for shared reduce_scatter (requires caller to
               reshape params to common (n, d, d) before creating param groups)
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        # Source: modded-nanogpt records #22-#24 — explicit communication ordering (item 28)
        # Pre-compute scatter_order (large params first) and work_order (small params first)
        # so that large reduce_scatters start early and complete while small updates are computed.
        self._scatter_order = None  # lazily initialized
        self._work_order = None

    def _init_comm_ordering(self):
        """Initialize explicit communication ordering for better NCCL overlap (item 28)."""
        # Sort groups by total parameter size: launch large reduces first
        group_sizes = []
        for i, group in enumerate(self.param_groups):
            total_size = sum(p.numel() for p in group['params'])
            group_sizes.append((total_size, i))
        # scatter_order: large groups first (launch large NCCL ops first)
        self._scatter_order = [i for _, i in sorted(group_sizes, reverse=True)]
        # work_order: small groups first (process small updates while large NCCL ops complete)
        self._work_order = [i for _, i in sorted(group_sizes)]

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _reduce_teon(self, group: dict, world_size: int) -> dict:
        """TEON reduce: mode-1 concatenate grad pairs, then reduce_scatter concatenated matrices."""
        params = group['params']
        num_pairs = len(params) // 2
        p0 = params[0]
        m, n = p0.shape
        device, dtype = p0.device, p0.dtype
        concat_shape = (m, 2 * n)

        chunk_size = (num_pairs + world_size - 1) // world_size
        padded_num = chunk_size * world_size

        # Mode-1 concatenation of grad pairs
        grad_concat = torch.stack([
            torch.cat([params[2*i].grad, params[2*i+1].grad], dim=-1) for i in range(num_pairs)
        ])  # (num_pairs, m, 2n)
        stacked_grads = torch.empty(padded_num, *concat_shape, dtype=dtype, device=device)
        stacked_grads[:num_pairs].copy_(grad_concat)
        if num_pairs < padded_num:
            stacked_grads[num_pairs:].zero_()

        grad_chunk = torch.empty(chunk_size, *concat_shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads,
                    chunk_size=chunk_size, n_orig=n, num_pairs=num_pairs)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            # Source: nanochat PR #492 — flipped MLP LR multiplier (item 11)
            ratio = shape[-2] / shape[-1]
            lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
            self._muon_lr_t.fill_(group["lr"] * lr_mult)
            self._muon_wd_t.fill_(group["weight_decay"])

            use_hyperball = group.get("use_hyperball", False)

            if use_hyperball:
                # Source: nanochat PR #498 — Hyperball in distributed mode (item 9)
                if "cached_norms" not in state:
                    state["cached_norms"] = stacked_owned.norm(dim=(-2, -1), keepdim=True).clone()
                    # Pad to chunk_size if needed
                    if num_owned < chunk_size:
                        padded_norms = torch.ones(chunk_size, 1, 1, dtype=dtype, device=device)
                        padded_norms[:num_owned] = state["cached_norms"]
                        state["cached_norms"] = padded_norms
                hyperball_step_fused(
                    grad_chunk[:num_owned], stacked_owned,
                    state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                    self._muon_momentum_t, self._muon_lr_t, self._muon_beta2_t,
                    group["ns_steps"], red_dim, state["cached_norms"][:num_owned],
                )
            else:
                muon_step_fused(
                    grad_chunk[:num_owned], stacked_owned,
                    state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                    self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                    group["ns_steps"], red_dim,
                )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _compute_teon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """TEON compute: run teon_step_fused on this rank's chunk of concatenated pairs."""
        info['future'].wait()
        params = group['params']
        num_pairs = info['num_pairs']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        n_orig = info['n_orig']
        p0 = params[0]
        m, n = p0.shape
        device, dtype = p0.device, p0.dtype
        concat_shape = (m, 2 * n)

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, num_pairs - start_idx))

        state = self.state[p0]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *concat_shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            # Per-layer variance: 2*chunk_size entries (each concat pair -> 2 individual layers)
            if m >= n:
                state["second_momentum_buffer"] = torch.zeros(2 * chunk_size, m, 1, dtype=dtype, device=device)
            else:
                state["second_momentum_buffer"] = torch.zeros(2 * chunk_size, 1, n, dtype=dtype, device=device)
        red_dim = -1 if m >= n else -2

        updated_params = torch.empty(chunk_size, *concat_shape, dtype=dtype, device=device)

        if num_owned > 0:
            # Stack the owned concatenated params
            owned_params = []
            for i in range(num_owned):
                pair_idx = start_idx + i
                owned_params.append(torch.cat([params[2*pair_idx], params[2*pair_idx+1]], dim=-1))
            stacked_owned = torch.stack(owned_params)  # (num_owned, m, 2n)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            # LR scaling uses original per-layer shape
            ratio = m / n
            lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
            self._muon_lr_t.fill_(group["lr"] * lr_mult)
            self._muon_wd_t.fill_(group["weight_decay"])

            teon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:2 * num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim, n_orig,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]  # Reuse buffer for all_gather
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=None,
                                teon_params=params, n_orig=n_orig, num_pairs=num_pairs))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if "teon_params" in info:
                # TEON: unconcatenate and copy back to individual params
                teon_params = info["teon_params"]
                n_orig = info["n_orig"]
                num_pairs = info["num_pairs"]
                stacked = info["stacked_params"][:num_pairs]
                for i in range(num_pairs):
                    teon_params[2*i].copy_(stacked[i, :, :n_orig])
                    teon_params[2*i+1].copy_(stacked[i, :, n_orig:])
            elif info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self, do_adam=True):
        """
        Args:
            do_adam: if False, skip AdamW groups (item 22).
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Source: modded-nanogpt records #22-#24 — explicit communication ordering (item 28)
        if self._scatter_order is None:
            self._init_comm_ordering()

        # Phase 1: launch all async reduce ops in scatter_order (large first)
        reduce_infos: dict[int, dict] = {}
        for idx in self._scatter_order:
            group = self.param_groups[idx]
            if group['kind'] == 'adamw':
                if do_adam:
                    reduce_infos[idx] = self._reduce_adamw(group, world_size)
            elif group['kind'] == 'muon':
                reduce_infos[idx] = self._reduce_muon(group, world_size)
            elif group['kind'] == 'teon':
                reduce_infos[idx] = self._reduce_teon(group, world_size)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait + compute + launch gathers in work_order (small first)
        gather_list: list[dict] = []
        for idx in self._work_order:
            if idx not in reduce_infos:
                continue
            group = self.param_groups[idx]
            info = reduce_infos[idx]
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            elif group['kind'] == 'teon':
                self._compute_teon(group, info, gather_list, rank)

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)

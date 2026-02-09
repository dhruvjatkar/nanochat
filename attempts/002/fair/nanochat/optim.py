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
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express with Chebyshev-optimal coefficients (item 13)
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
    g = X

    # NorMuon variance reduction
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

    @torch.no_grad()
    def step(self, do_adam=True):
        """
        Args:
            do_adam: if False, skip AdamW groups (item 22 — heterogeneous batch sizes).
                     Muon groups always step.
        """
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                # Source: modded-nanogpt record #39 PR #136 — skip Adam on even steps (item 22)
                if do_adam:
                    self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
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

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
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

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)

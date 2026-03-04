"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.

Ablation version: adds optional features behind flags (all default OFF = baseline behavior).

Optional features:
- Item 5:  Cautious weight decay for AdamW (--use-cautious-wd)
- Item 9:  Hyperball optimizer (use_hyperball in param_groups)
- Item 11: Flipped MLP LR multiplier (bundled with --use-polar-express)
- Item 13: Flipped LR multiplier for Muon (--use-polar-express)
- Item 14: Mantissa tracking for BF16 precision (--use-mantissa-tracking)
- Item 22: Heterogeneous batch sizes (do_adam flag in step())
- Item 27: TEON cross-layer orthogonalization (kind='teon' in param_groups)
- Item 28: Explicit communication ordering (bundled with TEON)

Adapted from: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# AdamW step: baseline (standard decoupled WD)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr_t: Tensor, beta1_t: Tensor, beta2_t: Tensor,
    eps_t: Tensor, wd_t: Tensor,
) -> None:
    """Fused AdamW step with standard decoupled weight decay (baseline)."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


# Item 5: Cautious weight decay variant
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_cautious(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr_t: Tensor, beta1_t: Tensor, beta2_t: Tensor,
    eps_t: Tensor, wd_t: Tensor,
) -> None:
    """Fused AdamW step with cautious weight decay (item 5)."""
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    update = exp_avg / denom * step_size
    mask = (update * p) > 0
    p.sub_(update)
    p.sub_(p * mask * (lr_t * wd_t))


# -----------------------------------------------------------------------------
# Polar Express coefficients (same in baseline and attempt 001)

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# Shared helper functions (factored out for TEON reuse)
def _nesterov_momentum(stacked_grads: Tensor, momentum_buffer: Tensor, momentum_t: Tensor) -> Tensor:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    return stacked_grads.lerp_(momentum_buffer, momentum)


def _polar_express(g: Tensor, ns_steps: int) -> Tensor:
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def _normuon_variance(g: Tensor, second_momentum_buffer: Tensor, beta2_t: Tensor, red_dim: int) -> Tensor:
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


def _nesterov_polar_variance(stacked_grads, momentum_buffer, second_momentum_buffer,
                              momentum_t, beta2_t, ns_steps, red_dim):
    g = _nesterov_momentum(stacked_grads, momentum_buffer, momentum_t)
    g = _polar_express(g, ns_steps)
    g = _normuon_variance(g, second_momentum_buffer, beta2_t, red_dim)
    return g


# Baseline Muon step (inlined, matching baseline optim.py)
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor, stacked_params: Tensor,
    momentum_buffer: Tensor, second_momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, wd_t: Tensor, beta2_t: Tensor,
    ns_steps: int, red_dim: int,
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update"""
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
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
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Item 9: Hyperball step (norm projection instead of WD)
@torch.compile(dynamic=False, fullgraph=True)
def hyperball_step_fused(
    stacked_grads: Tensor, stacked_params: Tensor,
    momentum_buffer: Tensor, second_momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, beta2_t: Tensor,
    ns_steps: int, red_dim: int,
    cached_norms: Tensor,
) -> None:
    """Hyperball step: Muon update + norm projection (no weight decay)."""
    g = _nesterov_polar_variance(
        stacked_grads, momentum_buffer, second_momentum_buffer,
        momentum_t, beta2_t, ns_steps, red_dim,
    )
    lr = lr_t.to(g.dtype)
    stacked_params.sub_(lr * g)
    current_norms = stacked_params.norm(dim=(-2, -1), keepdim=True)
    stacked_params.mul_(cached_norms / current_norms.clamp_min(1e-10))


# Item 27: TEON step (cross-layer orthogonalization)
@torch.compile(dynamic=False, fullgraph=True)
def teon_step_fused(
    stacked_grads_concat: Tensor, stacked_params_concat: Tensor,
    momentum_buffer_concat: Tensor, second_momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, wd_t: Tensor, beta2_t: Tensor,
    ns_steps: int, red_dim: int, n_orig: int,
) -> None:
    """TEON step: joint momentum + polar express on concatenated pairs, per-layer NorMuon + update."""
    g = _nesterov_momentum(stacked_grads_concat, momentum_buffer_concat, momentum_t)
    g = _polar_express(g, ns_steps)
    n_pairs = g.size(0)
    g_layers = torch.cat([g[..., :n_orig], g[..., n_orig:]], dim=0)
    g_layers = _normuon_variance(g_layers, second_momentum_buffer, beta2_t, red_dim)
    g_even = g_layers[:n_pairs]
    g_odd = g_layers[n_pairs:]
    lr = lr_t.to(g_even.dtype)
    wd = wd_t.to(g_even.dtype)
    p_even = stacked_params_concat[..., :n_orig]
    mask_even = (g_even * p_even) >= 0
    p_even.sub_(lr * g_even + lr * wd * p_even * mask_even)
    p_odd = stacked_params_concat[..., n_orig:]
    mask_odd = (g_odd * p_odd) >= 0
    p_odd.sub_(lr * g_odd + lr * wd * p_odd * mask_odd)


# Item 14: Mantissa tracking utilities
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
# Single GPU MuonAdamW optimizer

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others. Single GPU version.

    Supports optional features via param_group flags:
    - use_cautious_wd: cautious weight decay for AdamW (item 5)
    - use_hyperball: Hyperball norm projection for Muon (item 9)
    - use_flipped_lr: flipped MLP LR multiplier for Muon (item 13)
    - use_mantissa_tracking: BF16 mantissa tracking for AdamW (item 14)
    - kind='teon': TEON cross-layer orthogonalization (item 27)
    - do_adam=False in step(): skip AdamW groups (item 22)
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
        use_cautious = group.get('use_cautious_wd', False)
        use_mantissa = group.get('use_mantissa_tracking', False)
        step_fn = adamw_step_fused_cautious if use_cautious else adamw_step_fused

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if use_mantissa and p.dtype == torch.bfloat16:
                    # Initialize mantissa from current param
                    state['mantissa'] = torch.zeros(p.shape, dtype=torch.uint16, device=p.device)

            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            if use_mantissa and 'mantissa' in state:
                # Reconstruct FP32, run step, split back
                p_fp32 = mantissa_reconstruct_fp32(p.data, state['mantissa'])
                exp_avg_fp32 = state['exp_avg'].float() if state['exp_avg'].dtype != torch.float32 else state['exp_avg']
                exp_avg_sq_fp32 = state['exp_avg_sq'].float() if state['exp_avg_sq'].dtype != torch.float32 else state['exp_avg_sq']
                grad_fp32 = grad.float()
                step_fn(p_fp32, grad_fp32, exp_avg_fp32, exp_avg_sq_fp32,
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
                p_bf16, mantissa = mantissa_split_bf16(p_fp32)
                p.data.copy_(p_bf16)
                state['mantissa'] = mantissa
                if state['exp_avg'].dtype != torch.float32:
                    state['exp_avg'].copy_(exp_avg_fp32.to(state['exp_avg'].dtype))
                if state['exp_avg_sq'].dtype != torch.float32:
                    state['exp_avg_sq'].copy_(exp_avg_sq_fp32.to(state['exp_avg_sq'].dtype))
            else:
                step_fn(p, grad, state['exp_avg'], state['exp_avg_sq'],
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

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
        self._muon_wd_t.fill_(group["weight_decay"])

        # Item 13/11: flipped LR multiplier
        use_flipped_lr = group.get("use_flipped_lr", False)
        if use_flipped_lr:
            ratio = shape[-2] / shape[-1]
            lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
        else:
            lr_mult = max(1.0, shape[-2] / shape[-1])**0.5
        self._muon_lr_t.fill_(group["lr"] * lr_mult)

        use_hyperball = group.get("use_hyperball", False)
        if use_hyperball:
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
        """TEON step: consecutive params form pairs for cross-layer orthogonalization."""
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
        if "second_momentum_buffer" not in state:
            if m >= n:
                state["second_momentum_buffer"] = torch.zeros(2 * num_pairs, m, 1, dtype=dtype, device=device)
            else:
                state["second_momentum_buffer"] = torch.zeros(2 * num_pairs, 1, n, dtype=dtype, device=device)
        red_dim = -1 if m >= n else -2

        stacked_grads_concat = torch.stack([
            torch.cat([params[2*i].grad, params[2*i+1].grad], dim=-1) for i in range(num_pairs)
        ])
        stacked_params_concat = torch.stack([
            torch.cat([params[2*i], params[2*i+1]], dim=-1) for i in range(num_pairs)
        ])

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        use_flipped_lr = group.get("use_flipped_lr", False)
        if use_flipped_lr:
            ratio = m / n
            lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
        else:
            lr_mult = max(1.0, m / n)**0.5
        self._muon_lr_t.fill_(group["lr"] * lr_mult)
        self._muon_wd_t.fill_(group["weight_decay"])

        teon_step_fused(
            stacked_grads_concat, stacked_params_concat,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim, n,
        )

        for i in range(num_pairs):
            params[2*i].copy_(stacked_params_concat[i, :, :n])
            params[2*i+1].copy_(stacked_params_concat[i, :, n:])

    @torch.no_grad()
    def step(self, do_adam=True):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                if do_adam:
                    self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'teon':
                self._step_teon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


# -----------------------------------------------------------------------------
# Distributed MuonAdamW optimizer

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.
    Supports all features from MuonAdamW plus distributed communication.
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
        self._scatter_order = None
        self._work_order = None

    def _init_comm_ordering(self):
        group_sizes = []
        for i, group in enumerate(self.param_groups):
            total_size = sum(p.numel() for p in group['params'])
            group_sizes.append((total_size, i))
        self._scatter_order = [i for _, i in sorted(group_sizes, reverse=True)]
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
        params = group['params']
        num_pairs = len(params) // 2
        p0 = params[0]
        m, n = p0.shape
        device, dtype = p0.device, p0.dtype
        concat_shape = (m, 2 * n)

        chunk_size = (num_pairs + world_size - 1) // world_size
        padded_num = chunk_size * world_size

        grad_concat = torch.stack([
            torch.cat([params[2*i].grad, params[2*i+1].grad], dim=-1) for i in range(num_pairs)
        ])
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
        use_cautious = group.get('use_cautious_wd', False)
        use_mantissa = group.get('use_mantissa_tracking', False)
        step_fn = adamw_step_fused_cautious if use_cautious else adamw_step_fused

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
                if use_mantissa and p_slice.dtype == torch.bfloat16:
                    state['mantissa'] = torch.zeros(p_slice.shape, dtype=torch.uint16, device=p_slice.device)
            state['step'] += 1

            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            if use_mantissa and 'mantissa' in state:
                p_fp32 = mantissa_reconstruct_fp32(p_slice.data, state['mantissa'])
                exp_avg_fp32 = state['exp_avg'].float() if state['exp_avg'].dtype != torch.float32 else state['exp_avg']
                exp_avg_sq_fp32 = state['exp_avg_sq'].float() if state['exp_avg_sq'].dtype != torch.float32 else state['exp_avg_sq']
                grad_fp32 = grad_slice.float()
                step_fn(p_fp32, grad_fp32, exp_avg_fp32, exp_avg_sq_fp32,
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
                p_bf16, mantissa = mantissa_split_bf16(p_fp32)
                p_slice.data.copy_(p_bf16)
                state['mantissa'] = mantissa
            else:
                step_fn(p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                        self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                        self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

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
            self._muon_wd_t.fill_(group["weight_decay"])

            use_flipped_lr = group.get("use_flipped_lr", False)
            if use_flipped_lr:
                ratio = shape[-2] / shape[-1]
                lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
            else:
                lr_mult = max(1.0, shape[-2] / shape[-1])**0.5
            self._muon_lr_t.fill_(group["lr"] * lr_mult)

            use_hyperball = group.get("use_hyperball", False)
            if use_hyperball:
                if "cached_norms" not in state:
                    state["cached_norms"] = stacked_owned.norm(dim=(-2, -1), keepdim=True).clone()
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
            if m >= n:
                state["second_momentum_buffer"] = torch.zeros(2 * chunk_size, m, 1, dtype=dtype, device=device)
            else:
                state["second_momentum_buffer"] = torch.zeros(2 * chunk_size, 1, n, dtype=dtype, device=device)
        red_dim = -1 if m >= n else -2

        updated_params = torch.empty(chunk_size, *concat_shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = []
            for i in range(num_owned):
                pair_idx = start_idx + i
                owned_params.append(torch.cat([params[2*pair_idx], params[2*pair_idx+1]], dim=-1))
            stacked_owned = torch.stack(owned_params)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            use_flipped_lr = group.get("use_flipped_lr", False)
            if use_flipped_lr:
                ratio = m / n
                lr_mult = 1.0 if ratio >= 1 else ratio**-0.5
            else:
                lr_mult = max(1.0, m / n)**0.5
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

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=None,
                                teon_params=params, n_orig=n_orig, num_pairs=num_pairs))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if "teon_params" in info:
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
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self._scatter_order is None:
            self._init_comm_ordering()

        # Phase 1: launch all async reduce ops
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

        # Phase 2: wait for reduces, compute updates, launch gathers
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

"""
GPT model — ablation version.

Baseline (all flags OFF) matches the root nanochat/gpt.py exactly.
Each feature from the 30-item experiment plan is behind a config toggle (default OFF).

Architecture features:
- Item 1:  Chunk-based RoPE (use_chunk_rope)
- Item 6:  Fused cross-entropy via cut-cross-entropy (use_fused_ce)
- Item 10a: Parameterized RMSNorm with learnable gamma (use_param_rmsnorm)
- Item 10b: Projection scalars on attn+MLP output (use_proj_scalars)
- Item 15: Backout mechanism (use_backout)
- Item 16a: Smear module (use_smear)
- Item 16b: Sparse attention gate (use_attn_gate)
- Item 18a: Half-truncated RoPE (use_half_truncated_rope)
- Item 18b: Partial key offset (use_partial_key_offset)
- Item 19: Bigram hash embedding (bigram_vocab_multiplier > 0)
- Item 20: U-net skip connections (use_unet_skip)
- Item 25: Fused Triton MLP kernel (use_fused_mlp)
- Item 30: Paired head attention (use_paired_heads)
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

# Item 6: Fused cross-entropy (conditional import)
try:
    from cut_cross_entropy import linear_cross_entropy
    linear_cross_entropy = torch.compiler.disable(linear_cross_entropy)
    torch.backends.cuda.matmul.allow_tf32 = True
    HAS_CUT_CE = True
except ImportError:
    HAS_CUT_CE = False

# Item 25: Fused MLP kernel (conditional import)
try:
    from triton_kernels import fused_mlp
    HAS_FUSED_MLP = True
except ImportError:
    HAS_FUSED_MLP = False

# Item 19: Bigram hash (conditional import)
from nanochat.dataloader import compute_bigram_hash_ids


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    window_pattern: str = "SSSL"
    # --- Ablation toggles (all default OFF = baseline behavior) ---
    use_chunk_rope: bool = False            # Item 1
    use_fused_ce: bool = False              # Item 6
    use_param_rmsnorm: bool = False         # Item 10a
    use_proj_scalars: bool = False          # Item 10b
    use_backout: bool = False               # Item 15
    use_smear: bool = False                 # Item 16a
    use_attn_gate: bool = False             # Item 16b
    use_half_truncated_rope: bool = False   # Item 18a
    use_partial_key_offset: bool = False    # Item 18b
    bigram_vocab_multiplier: int = 0        # Item 19 (0 = off)
    use_unet_skip: bool = False             # Item 20
    use_fused_mlp: bool = False             # Item 25
    use_paired_heads: bool = False          # Item 30


def norm(x):
    # Purely functional rmsnorm with no learnable params (baseline)
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# Baseline: indexing-based split
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# Item 1: chunk-based split (better backward kernel)
def apply_rotary_emb_chunk(x, cos, sin):
    assert x.ndim == 4
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

        # Item 10b: projection scalar
        self.c_proj_scalar = nn.Parameter(torch.zeros(1)) if config.use_proj_scalars else None

        # Item 16b: sparse attention gate
        self.attn_gate = nn.Linear(12, self.n_head, bias=False) if config.use_attn_gate else None

        # Item 18b: partial key offset
        self.use_partial_key_offset = config.use_partial_key_offset

        # Item 30: paired head attention
        self.use_paired_heads = config.use_paired_heads and (config.n_head % 2 == 0)

        # Item 1: chunk-based RoPE
        self.use_chunk_rope = config.use_chunk_rope

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Item 18b: partial key offset
        if self.use_partial_key_offset and T > 1 and kv_cache is None:
            half_dim = self.head_dim // 2
            k_lower = k[:, :, :, :half_dim]
            k_upper_shifted = torch.cat([k[:, :1, :, half_dim:], k[:, :-1, :, half_dim:]], dim=1)
            k = torch.cat([k_lower, k_upper_shifted], dim=-1)

        # Apply Rotary Embeddings
        cos, sin = cos_sin
        rope_fn = apply_rotary_emb_chunk if self.use_chunk_rope else apply_rotary_emb
        q, k = rope_fn(q, cos, sin), rope_fn(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Item 30: paired head attention
        if self.use_paired_heads and kv_cache is None:
            q = q.view(B, T, self.n_head // 2, self.head_dim * 2)
            k = k.view(B, T, self.n_kv_head // 2, self.head_dim * 2)
            v = v.view(B, T, self.n_kv_head // 2, self.head_dim * 2)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.view(B, T, self.n_head, self.head_dim)
        else:
            # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
            if kv_cache is None:
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

        # Item 16b: sparse attention gate
        if self.attn_gate is not None:
            attn_gate_val = torch.sigmoid(self.attn_gate(x[..., :12]))
            y = y * attn_gate_val.unsqueeze(-1)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)

        # Item 10b: projection scalar
        if self.c_proj_scalar is not None:
            y = y * (1 + self.c_proj_scalar)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        # Item 10b: projection scalar
        self.c_proj_scalar = nn.Parameter(torch.zeros(1)) if config.use_proj_scalars else None

        # Item 25: fused MLP
        self.use_fused_mlp = config.use_fused_mlp and HAS_FUSED_MLP

    def forward(self, x):
        if self.use_fused_mlp:
            y = fused_mlp(x, self.c_fc.weight, self.c_proj.weight)
        else:
            y = self.c_fc(x)
            y = F.relu(y).square()
            y = self.c_proj(y)
        # Item 10b: projection scalar
        if self.c_proj_scalar is not None:
            y = y * (1 + self.c_proj_scalar)
        return y


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        # Item 10a: parameterized RMSNorm vs functional norm
        self.use_param_rmsnorm = config.use_param_rmsnorm
        if self.use_param_rmsnorm:
            self.attn_norm = nn.RMSNorm(config.n_embd)
            self.mlp_norm = nn.RMSNorm(config.n_embd)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if self.use_param_rmsnorm:
            x = x + self.attn(self.attn_norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(self.mlp_norm(x))
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings (ResFormer-style)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # Item 16a: smear module
        if config.use_smear:
            self.smear_gate = nn.Linear(12, 1, bias=False)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.smear_gate = None
            self.smear_lambda = None

        # Item 15: backout mechanism
        if config.use_backout:
            self.backout_layer = 2 * config.n_layer // 3
            self.backout_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.backout_lambda = None

        # Item 20: U-net skip connections
        if config.use_unet_skip:
            self.skip_source_layer = config.n_layer // 3
            self.skip_target_layer = 2 * config.n_layer // 3
            self.skip_gate = nn.Linear(12, 1, bias=False)
            self.skip_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.skip_gate = None
            self.skip_lambda = None

        # Item 19: bigram hash embedding
        if config.bigram_vocab_multiplier > 0:
            bigram_vocab_size = config.vocab_size * config.bigram_vocab_multiplier
            self.bigram_embed = nn.Embedding(bigram_vocab_size, config.n_embd)
            self.bigram_vocab_size = bigram_vocab_size
            self.bigram_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.bigram_embed = None
            self.bigram_vocab_size = 0
            self.bigram_lambda = None

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # Item 10b: projection scalars
            if block.attn.c_proj_scalar is not None:
                torch.nn.init.zeros_(block.attn.c_proj_scalar)
            if block.mlp.c_proj_scalar is not None:
                torch.nn.init.zeros_(block.mlp.c_proj_scalar)
            # Item 16b: attention gate
            if block.attn.attn_gate is not None:
                torch.nn.init.zeros_(block.attn.attn_gate.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Item 16a: smear
        if self.smear_gate is not None:
            torch.nn.init.zeros_(self.smear_gate.weight)
            torch.nn.init.zeros_(self.smear_lambda)

        # Item 15: backout
        if self.backout_lambda is not None:
            torch.nn.init.zeros_(self.backout_lambda)

        # Item 20: U-net skip
        if self.skip_gate is not None:
            torch.nn.init.zeros_(self.skip_gate.weight)
            torch.nn.init.zeros_(self.skip_lambda)

        # Item 19: bigram
        if self.bigram_embed is not None:
            torch.nn.init.normal_(self.bigram_embed.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.bigram_lambda)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
            if self.bigram_embed is not None:
                self.bigram_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Item 18a: half-truncated RoPE
        if self.config.use_half_truncated_rope:
            half = head_dim // 4
            inv_freq[half:] = 0.0
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    # Item 17: dynamic window warmup (YaRN)
    def resize_windows(self, step, num_iterations, short_init=128, long_init=384,
                       short_final=None, long_final=None):
        if short_final is None:
            short_final = self.config.sequence_len // 2
        if long_final is None:
            long_final = self.config.sequence_len
        progress = min(step / max(num_iterations, 1), 1.0)
        short_w = int(short_init + (short_final - short_init) * progress)
        long_w = int(long_init + (long_final - long_init) * progress)
        pattern = self.config.window_pattern.upper()
        char_to_window = {"L": (long_w, 0), "S": (short_w, 0)}
        new_windows = []
        for layer_idx in range(self.config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            new_windows.append(char_to_window[char])
        new_windows[-1] = (long_w, 0)
        self.window_sizes = new_windows

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        # Exclude optional module params from matmul FLOPs
        if self.bigram_embed is not None:
            nparams_exclude += self.bigram_embed.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # Count optional modules separately
        other = 0
        if self.bigram_embed is not None:
            other += self.bigram_embed.weight.numel()
        if self.bigram_lambda is not None:
            other += self.bigram_lambda.numel()
        if self.smear_gate is not None:
            other += sum(p.numel() for p in self.smear_gate.parameters())
        if self.smear_lambda is not None:
            other += self.smear_lambda.numel()
        if self.backout_lambda is not None:
            other += self.backout_lambda.numel()
        if self.skip_gate is not None:
            other += sum(p.numel() for p in self.skip_gate.parameters())
        if self.skip_lambda is not None:
            other += self.skip_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + other
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                       weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                       use_hyperball=False, use_teon=False,
                       use_cautious_wd=False, use_polar_express=False,
                       use_mantissa_tracking=False):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Collect optional scalar/linear params
        new_scalar_params = []
        new_linear_params = []
        bigram_params = []
        if self.smear_lambda is not None:
            new_scalar_params.append(self.smear_lambda)
        if self.backout_lambda is not None:
            new_scalar_params.append(self.backout_lambda)
        if self.skip_lambda is not None:
            new_scalar_params.append(self.skip_lambda)
        if self.bigram_lambda is not None:
            new_scalar_params.append(self.bigram_lambda)
        if self.smear_gate is not None:
            new_linear_params.extend(list(self.smear_gate.parameters()))
        if self.skip_gate is not None:
            new_linear_params.extend(list(self.skip_gate.parameters()))
        if self.bigram_embed is not None:
            bigram_params = list(self.bigram_embed.parameters())

        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Common flags for AdamW groups
        adamw_extra = {}
        if use_cautious_wd:
            adamw_extra['use_cautious_wd'] = True
        if use_mantissa_tracking:
            adamw_extra['use_mantissa_tracking'] = True

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0, **adamw_extra),
        ]

        # Add optional param groups
        if new_scalar_params:
            param_groups.append(dict(kind='adamw', params=new_scalar_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra))
        if new_linear_params:
            param_groups.append(dict(kind='adamw', params=new_linear_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra))
        if bigram_params:
            param_groups.append(dict(kind='adamw', params=bigram_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra))

        # Muon flags
        muon_extra = {}
        if use_hyperball:
            muon_extra['use_hyperball'] = True
        if use_polar_express:
            muon_extra['use_flipped_lr'] = True

        if use_teon:
            # TEON: type-aware grouping for QKV attention params
            qkv_by_type = {'c_q': [], 'c_k': [], 'c_v': []}
            other_matrix = []
            leftover_1d = []
            for name, p in self.transformer.h.named_parameters():
                if p.ndim < 2:
                    leftover_1d.append(p)
                    continue
                matched = False
                for key in qkv_by_type:
                    if f'.attn.{key}.weight' in name:
                        layer_idx = int(name.split('.')[0])
                        qkv_by_type[key].append((layer_idx, p))
                        matched = True
                        break
                if not matched:
                    other_matrix.append(p)

            for key in qkv_by_type:
                sorted_params = sorted(qkv_by_type[key], key=lambda x: x[0])
                paired_params = []
                for i in range(0, len(sorted_params) - 1, 2):
                    paired_params.append(sorted_params[i][1])
                    paired_params.append(sorted_params[i+1][1])
                if paired_params:
                    param_groups.append(dict(
                        kind='teon', params=paired_params,
                        lr=matrix_lr, momentum=0.95, ns_steps=5,
                        beta2=0.95, weight_decay=weight_decay,
                        **muon_extra,
                    ))
                if len(sorted_params) % 2 == 1:
                    other_matrix.append(sorted_params[-1][1])

            for shape in sorted({p.shape for p in other_matrix}):
                group_params = [p for p in other_matrix if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                    **muon_extra,
                ))

            if leftover_1d:
                param_groups.append(dict(
                    kind='adamw', params=leftover_1d, lr=scalar_lr,
                    betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra,
                ))

            n_teon_params = sum(len(g['params']) for g in param_groups if g['kind'] == 'teon')
            n_muon_params = sum(len(g['params']) for g in param_groups if g['kind'] == 'muon')
            print0(f"TEON enabled: {n_teon_params} QKV params in TEON groups, {n_muon_params} remaining in Muon groups")
        else:
            # Baseline shape-only grouping
            all_h_params = list(self.transformer.h.parameters())
            matrix_params = [p for p in all_h_params if p.ndim >= 2]
            leftover_1d = [p for p in all_h_params if p.ndim < 2]

            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                    **muon_extra,
                ))

            if leftover_1d:
                param_groups.append(dict(
                    kind='adamw', params=leftover_1d, lr=scalar_lr,
                    betas=adam_betas, eps=1e-10, weight_decay=0.0, **adamw_extra,
                ))

        # Verify all params are accounted for
        grouped_params = set()
        for g in param_groups:
            for p in g['params']:
                grouped_params.add(id(p))
        all_params = set(id(p) for p in self.parameters())
        assert grouped_params == all_params, f"Parameter grouping mismatch: {len(grouped_params)} grouped vs {len(all_params)} total"

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # Item 24: tied/untied embedding support
    def tie_embeddings(self):
        """Tie wte and lm_head weights."""
        self.lm_head.weight = nn.Parameter(self.transformer.wte.weight)
        self._embeddings_tied = True

    def untie_embeddings(self):
        """Untie embeddings: create separate lm_head weight from current tied state."""
        if hasattr(self, '_embeddings_tied') and self._embeddings_tied:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone())
            self._embeddings_tied = False

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                mtp_weights=None, bigram_ids=None):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual

        # Item 19: bigram hash embedding
        if self.bigram_embed is not None and kv_cache is None:
            if bigram_ids is None:
                bigram_ids = compute_bigram_hash_ids(idx, self.bigram_vocab_size)
            bigram_emb = self.bigram_embed(bigram_ids)
        else:
            bigram_emb = None

        # Item 16a: smear module
        if self.smear_gate is not None and kv_cache is None and T > 1:
            smear_input = x[:, 1:, :12]
            smear_gate_val = torch.sigmoid(self.smear_gate(smear_input))
            smear_addition = self.smear_lambda * smear_gate_val * x[:, :-1]
            x = torch.cat([x[:, :1], x[:, 1:] + smear_addition], dim=1)

        # Transformer blocks
        x_backout = None
        x_skip = None

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Item 19: blend in bigram embedding
            if bigram_emb is not None:
                x = x + self.bigram_lambda * bigram_emb

            # Item 20: save skip source
            if self.skip_gate is not None and i == self.skip_source_layer:
                x_skip = x.clone()

            # Item 20: inject skip at target
            if self.skip_gate is not None and i == self.skip_target_layer and x_skip is not None:
                skip_gate_val = torch.sigmoid(self.skip_gate(x_skip[..., :12]))
                x = x + torch.sigmoid(self.skip_lambda) * 2 * skip_gate_val * x_skip

            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

            # Item 15: save backout residual
            if self.backout_lambda is not None and i == self.backout_layer:
                x_backout = x.clone()

        # Item 15: apply backout
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 20  # baseline softcap

        if targets is not None:
            # Training: compute loss
            if self.config.use_fused_ce and HAS_CUT_CE and loss_reduction != 'none':
                # Item 6: fused cross-entropy (avoids materializing logits)
                loss = linear_cross_entropy(
                    x.bfloat16(), self.lm_head.weight[:self.config.vocab_size],
                    targets,
                    softcap=softcap,
                    reduction=loss_reduction,
                    ignore_index=-1,
                )
            else:
                # Baseline: standard materialized logits path
                logits = self.lm_head(x)
                logits = logits[..., :self.config.vocab_size]
                logits = logits.float()
                logits = softcap * torch.tanh(logits / softcap)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                      ignore_index=-1, reduction=loss_reduction)

            # Item 23: Multi-token prediction
            if mtp_weights is not None and len(mtp_weights) > 1:
                total_loss = mtp_weights[0] * loss
                for offset_idx in range(1, len(mtp_weights)):
                    w = mtp_weights[offset_idx]
                    if w <= 0:
                        continue
                    offset = offset_idx + 1
                    if T <= offset:
                        continue
                    mtp_targets = torch.full_like(targets, -1)
                    mtp_targets[:, :-offset+1] = targets[:, offset-1:]
                    if self.config.use_fused_ce and HAS_CUT_CE:
                        mtp_loss = linear_cross_entropy(
                            x.bfloat16(), self.lm_head.weight[:self.config.vocab_size],
                            mtp_targets, softcap=softcap,
                            reduction=loss_reduction, ignore_index=-1,
                        )
                    else:
                        mtp_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                   mtp_targets.view(-1),
                                                   ignore_index=-1, reduction=loss_reduction)
                    total_loss = total_loss + w * mtp_loss
                loss = total_loss

            return loss
        else:
            # Inference: return logits
            logits = self.lm_head(x)
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive streaming inference."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

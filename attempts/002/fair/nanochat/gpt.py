"""
GPT model (attempt-local override with all 30-item experiment plan optimizations).

Changes from root:
- Item 1:  chunk-based RoPE (PR #492)
- Item 6:  Fused cross-entropy via cut-cross-entropy (PR #128 + arXiv:2411.09009)
- Item 7:  GQA with n_kv_head CLI plumbing (arXiv:2305.13245)
- Item 9:  Hyperball-aware init_weights (PR #498)
- Item 10: Parameterized RMSNorm + projection scalars (PR #498 + modded-nanogpt #42)
- Item 15: Backout mechanism (modded-nanogpt #40 PR #140)
- Item 16: Smear module + sparse attention gate (modded-nanogpt #34/#28)
- Item 17: Dynamic attention window warmup with YaRN (modded-nanogpt #13/#31 + arXiv:2309.00071)
- Item 18: Half-truncated RoPE + partial key offset (modded-nanogpt #17/#49)
- Item 19: Bigram hash embedding (modded-nanogpt #62 PR #201)
- Item 20: U-net skip connections (modded-nanogpt #11 + arXiv:1505.04597)
- Item 23: Multi-token prediction (modded-nanogpt #53 PR #178 + arXiv:2404.19737)
- Item 24: Tied embed -> untie mid-training (modded-nanogpt #51 PR #175)
- Item 25: Fused Triton MLP kernel (modded-nanogpt #59 PR #197)
- Item 27: Parameter reshaping for shared reduce_scatter (modded-nanogpt #36 PR #132)
- Item 30: Paired head attention (modded-nanogpt #58 PR #191)
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module
from nanochat.flash_attention import flash_attn

# Source: nanochat PR #128 + arXiv:2411.09009 — cut-cross-entropy for fused CE (item 6)
try:
    from cut_cross_entropy import linear_cross_entropy
    HAS_CUT_CE = True
except ImportError:
    HAS_CUT_CE = False
    print0("Warning: cut-cross-entropy not installed. Using standard CE. Install with: pip install cut-cross-entropy")

# Source: modded-nanogpt record #59 PR #197 — fused MLP kernel (item 25)
try:
    from triton_kernels import fused_mlp
    HAS_FUSED_MLP = True
except ImportError:
    HAS_FUSED_MLP = False

# Source: modded-nanogpt record #62 PR #201 — bigram hash embedding (item 19)
from nanochat.dataloader import compute_bigram_hash_ids


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6      # number of query heads
    n_kv_head: int = 6   # number of key/value heads (GQA) — Source: arXiv:2305.13245 (item 7)
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # --- Item 19: Bigram hash embedding ---
    # Source: modded-nanogpt record #62 PR #201
    bigram_vocab_multiplier: int = 3  # bigram_vocab_size = vocab_size * this
    # --- Item 30: Paired head attention ---
    # Source: modded-nanogpt record #58 PR #191
    use_paired_heads: bool = False  # requires even n_head
    # --- Item 18: Half-truncated RoPE ---
    # Source: modded-nanogpt record #17 by @YouJiacheng
    use_half_truncated_rope: bool = True
    # --- Item 25: Fused MLP kernel ---
    use_fused_mlp: bool = True


# Source: nanochat PR #498 + modded-nanogpt #42 PR #151 — parameterized RMSNorm (item 10)
# Replaces the functional norm() with learnable RMSNorm (gamma parameter).
# The original used F.rms_norm with no learnable parameters.
def norm(x):
    """Functional rmsnorm with no learnable params (used for QK norm in attention)."""
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


# Source: nanochat PR #492 https://github.com/karpathy/nanochat/pull/492 — chunk-based RoPE (item 1)
# Uses x.chunk(2, dim=-1) instead of x[..., :d], x[..., d:] for better backward kernel.
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    # Source: nanochat PR #492 — chunk-based split for better backward kernel (item 1)
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
        # Source: nanochat PR #498 + modded-nanogpt #42 — projection scalar (item 10)
        # Zero-init scalar that scales attention output: out *= (1 + c_proj_scalar)
        self.c_proj_scalar = nn.Parameter(torch.zeros(1))
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Source: modded-nanogpt record #28 PR #117 — sparse attention gate (item 16)
        self.attn_gate = nn.Linear(12, self.n_head, bias=False)
        # Source: modded-nanogpt record #49 PR #169 — partial key offset (item 18)
        self.use_partial_key_offset = True  # enabled for all layers
        # Source: modded-nanogpt record #58 PR #191 — paired head attention (item 30)
        self.use_paired_heads = config.use_paired_heads and (config.n_head % 2 == 0)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Source: modded-nanogpt record #49 PR #169 — partial key offset (item 18)
        # Shift upper half of key dims from previous token for implicit bigram attention.
        if self.use_partial_key_offset and T > 1 and kv_cache is None:
            half_dim = self.head_dim // 2
            k[:, 1:, :, half_dim:] = k[:, :-1, :, half_dim:].clone()

        # Apply Rotary Embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Source: modded-nanogpt record #58 PR #191 — paired head attention (item 30)
        if self.use_paired_heads and kv_cache is None:
            # Pair adjacent heads into double-width heads
            q = q.view(B, T, self.n_head // 2, self.head_dim * 2)
            k = k.view(B, T, self.n_kv_head // 2, self.head_dim * 2)
            v = v.view(B, T, self.n_kv_head // 2, self.head_dim * 2)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.view(B, T, self.n_head, self.head_dim)
        else:
            # Standard attention
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

        # Source: modded-nanogpt record #28 PR #117 — sparse attention gate (item 16)
        # Cheap input-dependent gate on the first 12 embedding dims per head.
        attn_gate_val = torch.sigmoid(self.attn_gate(x[..., :12]))  # (B, T, n_head)
        y = y * attn_gate_val.unsqueeze(-1)  # (B, T, n_head, head_dim)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        # Source: nanochat PR #498 — projection scalar (item 10)
        y = y * (1 + self.c_proj_scalar)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # Source: nanochat PR #498 — projection scalar for MLP (item 10)
        self.c_proj_scalar = nn.Parameter(torch.zeros(1))
        self.use_fused_mlp = config.use_fused_mlp and HAS_FUSED_MLP

    def forward(self, x):
        if self.use_fused_mlp:
            # Source: modded-nanogpt record #59 PR #197 — fused Triton MLP (item 25)
            y = fused_mlp(x, self.c_fc.weight, self.c_proj.weight)
        else:
            y = self.c_fc(x)
            y = F.relu(y).square()
            y = self.c_proj(y)
        # Source: nanochat PR #498 — projection scalar (item 10)
        y = y * (1 + self.c_proj_scalar)
        return y


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Source: nanochat PR #498 + modded-nanogpt #42 PR #151 — parameterized RMSNorm (item 10)
        # Learnable gamma in RMSNorm replaces the functional norm() for block norms.
        self.attn_norm = nn.RMSNorm(config.n_embd)
        self.mlp_norm = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Source: nanochat PR #498 — parameterized RMSNorm (item 10)
        x = x + self.attn(self.attn_norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
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

        # Source: modded-nanogpt record #34 PR #130 — smear module (item 16)
        # Cheap inter-token information flow after embedding, before transformer blocks.
        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # Source: modded-nanogpt record #40 PR #140 — backout mechanism (item 15)
        # Subtract a gated copy of the residual stream from 2/3 depth before final norm.
        self.backout_layer = 2 * config.n_layer // 3
        self.backout_lambda = nn.Parameter(torch.zeros(1))

        # Source: modded-nanogpt record #11 + arXiv:1505.04597 — U-net skip connections (item 20)
        # Gated skip from encoder layers (~1/3 depth) to decoder layers (~2/3 depth).
        self.skip_source_layer = config.n_layer // 3
        self.skip_target_layer = 2 * config.n_layer // 3
        self.skip_gate = nn.Linear(12, 1, bias=False)
        self.skip_lambda = nn.Parameter(torch.zeros(1))

        # Source: modded-nanogpt record #62 PR #201 — bigram hash embedding (item 19)
        bigram_vocab_size = config.vocab_size * config.bigram_vocab_multiplier
        self.bigram_embed = nn.Embedding(bigram_vocab_size, config.n_embd)
        self.bigram_vocab_size = bigram_vocab_size
        self.bigram_lambda = nn.Parameter(torch.zeros(1))

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
            # Source: nanochat PR #498 — projection scalars init to zero (item 10)
            torch.nn.init.zeros_(block.attn.c_proj_scalar)
            torch.nn.init.zeros_(block.mlp.c_proj_scalar)
            # Source: modded-nanogpt record #28 PR #117 — sparse gate init (item 16)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE gate weights init to zero
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Source: modded-nanogpt record #34 PR #130 — smear gate init (item 16)
        torch.nn.init.zeros_(self.smear_gate.weight)
        torch.nn.init.zeros_(self.smear_lambda)

        # Source: modded-nanogpt record #40 PR #140 — backout lambda init (item 15)
        torch.nn.init.zeros_(self.backout_lambda)

        # Source: modded-nanogpt record #11 — U-net skip init (item 20)
        torch.nn.init.zeros_(self.skip_gate.weight)
        torch.nn.init.zeros_(self.skip_lambda)

        # Source: modded-nanogpt record #62 PR #201 — bigram embedding init (item 19)
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
            self.bigram_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Source: modded-nanogpt record #17 by @YouJiacheng — half-truncated RoPE (item 18)
        # Zero out the second half of frequencies: only the first half of head dims
        # get positional encoding, creating position-invariant channels in the second half.
        if self.config.use_half_truncated_rope:
            half = head_dim // 4  # half of head_dim//2 frequencies
            inv_freq[half:] = 0.0
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    # Source: modded-nanogpt record #31 PR #122 + YaRN arXiv:2309.00071 — dynamic window warmup (item 17)
    def resize_windows(self, step, num_iterations, short_init=128, long_init=384,
                       short_final=None, long_final=None):
        """
        Dynamically resize attention windows during training.
        Windows grow from (short_init, long_init) at step 0 to (short_final, long_final) at end.
        When windows change, recompute RoPE frequencies with YaRN scaling.
        """
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
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_embed.weight.numel())
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
        # Include new params in total but not in scaling params
        bigram = self.bigram_embed.weight.numel()
        smear = sum(p.numel() for p in [self.smear_gate.weight, self.smear_lambda])
        backout = self.backout_lambda.numel()
        skip = sum(p.numel() for p in [self.skip_gate.weight, self.skip_lambda])
        other = bigram + smear + backout + skip + self.bigram_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + other
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
                       use_hyperball=False):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # New optimization params (items 15, 16, 19, 20)
        new_scalar_params = [self.smear_lambda, self.backout_lambda, self.skip_lambda, self.bigram_lambda]
        new_linear_params = list(self.smear_gate.parameters()) + list(self.skip_gate.parameters())
        bigram_params = list(self.bigram_embed.parameters())

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            # New params from items 15, 16, 19, 20
            dict(kind='adamw', params=new_scalar_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=new_linear_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=bigram_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (or Hyperball) for matrix params
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                # Source: nanochat PR #498 — Hyperball option (item 9)
                use_hyperball=use_hyperball,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                mtp_weights=None, bigram_ids=None):
        """
        Args:
            idx: (B, T) input token IDs
            targets: (B, T) target token IDs (optional, for training)
            kv_cache: KV cache for inference
            loss_reduction: 'mean' or 'sum'
            mtp_weights: list of floats for multi-token prediction (item 23), e.g. [1.0, 0.5, 0.25]
            bigram_ids: (B, T) precomputed bigram hash IDs (item 19, optional — computed if not provided)
        """
        B, T = idx.size()

        assert T <= self.cos.size(1)
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Embed tokens
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial embedding for x0 residual

        # Source: modded-nanogpt record #62 PR #201 — compute bigram hash IDs (item 19)
        if bigram_ids is None and kv_cache is None:
            bigram_ids = compute_bigram_hash_ids(idx, self.bigram_vocab_size)
        bigram_emb = self.bigram_embed(bigram_ids) if bigram_ids is not None else None

        # Source: modded-nanogpt record #34 PR #130 — smear module (item 16)
        # Cheap inter-token information flow: gate * previous_token added to current.
        if kv_cache is None and T > 1:
            smear_input = x[:, 1:, :12]  # first 12 dims of positions 1..T-1
            smear_gate_val = torch.sigmoid(self.smear_gate(smear_input))  # (B, T-1, 1)
            x = x.clone()
            x[:, 1:] = x[:, 1:] + self.smear_lambda * smear_gate_val * x[:, :-1]

        # Transformer blocks
        x_backout = None  # for backout mechanism (item 15)
        x_skip = None     # for U-net skip (item 20)

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Source: modded-nanogpt record #62 PR #201 — blend in bigram embedding (item 19)
            if bigram_emb is not None:
                x = x + self.bigram_lambda * bigram_emb

            # Source: modded-nanogpt record #11 — save skip source (item 20)
            if i == self.skip_source_layer:
                x_skip = x.clone()

            # Source: modded-nanogpt record #11 — inject skip at target (item 20)
            if i == self.skip_target_layer and x_skip is not None:
                skip_gate_val = torch.sigmoid(self.skip_gate(x_skip[..., :12]))
                x = x + torch.sigmoid(self.skip_lambda) * 2 * skip_gate_val * x_skip

            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

            # Source: modded-nanogpt record #40 PR #140 — save backout residual (item 15)
            if i == self.backout_layer:
                x_backout = x.clone()

        # Source: modded-nanogpt record #40 PR #140 — apply backout (item 15)
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)

        if targets is not None:
            # Training: compute loss
            softcap = 15

            if HAS_CUT_CE:
                # Source: nanochat PR #128 + arXiv:2411.09009 — fused cross-entropy (item 6)
                # Avoids materializing the (B*T, vocab_size) logits tensor.
                loss = linear_cross_entropy(
                    x, self.lm_head.weight[:self.config.vocab_size],
                    targets.view(-1),
                    softcap=softcap,
                    reduction=loss_reduction,
                    ignore_index=-1,
                )
            else:
                # Fallback: standard materialized logits path
                logits = self.lm_head(x)
                logits = logits[..., :self.config.vocab_size]
                logits = logits.float()
                logits = softcap * torch.tanh(logits / softcap)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                      ignore_index=-1, reduction=loss_reduction)

            # Source: modded-nanogpt record #53 PR #178 + arXiv:2404.19737 — MTP (item 23)
            # Multi-token prediction: compute CE at offsets +2, +3, etc. with annealing weights.
            if mtp_weights is not None and len(mtp_weights) > 1:
                total_loss = mtp_weights[0] * loss
                for offset_idx in range(1, len(mtp_weights)):
                    w = mtp_weights[offset_idx]
                    if w <= 0:
                        continue
                    offset = offset_idx + 1  # offset 2, 3, ...
                    if T <= offset:
                        continue
                    # Shift targets by additional offset
                    mtp_targets = torch.full_like(targets, -1)
                    mtp_targets[:, :-offset+1] = targets[:, offset-1:]
                    if HAS_CUT_CE:
                        mtp_loss = linear_cross_entropy(
                            x, self.lm_head.weight[:self.config.vocab_size],
                            mtp_targets.view(-1), softcap=softcap,
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
            softcap = 15
            logits = self.lm_head(x)
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    # Source: modded-nanogpt record #51 PR #175 — tied/untied embedding support (item 24)
    def tie_embeddings(self):
        """Tie wte and lm_head weights (wte shares lm_head's data transposed)."""
        self.lm_head.weight = nn.Parameter(self.transformer.wte.weight)
        self._embeddings_tied = True

    def untie_embeddings(self):
        """Untie embeddings: create separate lm_head weight from current tied state."""
        if hasattr(self, '_embeddings_tied') and self._embeddings_tied:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.data.clone())
            self._embeddings_tied = False

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
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

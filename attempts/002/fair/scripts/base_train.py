"""
Train model (attempt-local override with all experiment plan optimizations).

Changes from root:
- Item 7:  --n-kv-head CLI arg for GQA (arXiv:2305.13245)
- Item 12: Decoupled warmdown (PR #498 + modded-nanogpt #44 PR #158)
- Item 17: Dynamic attention window warmup with YaRN (modded-nanogpt #13/#31)
- Item 21: Batch size schedule (modded-nanogpt #46 PR #163)
- Item 22: Heterogeneous batch sizes — Adam every other step (modded-nanogpt #39 PR #136)
- Item 23: Multi-token prediction schedule (modded-nanogpt #53 PR #178)
- Item 24: Tied embed -> untie mid-training (modded-nanogpt #51 PR #175)
- Compile mode support (from previous fair attempt)

Run as:
python -m scripts.base_train
torchrun --nproc_per_node=8 -m scripts.base_train
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")
# Compilation
parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
# Source: arXiv:2305.13245 (Ainslie et al., GQA) — grouped-query attention (item 7)
parser.add_argument("--n-kv-head", type=int, default=-1, help="number of KV heads for GQA (-1 = same as n_head)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for Muon")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix params (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step")
# Source: nanochat PR #498 + modded-nanogpt #44 PR #158 — decoupled warmdown (item 12)
parser.add_argument("--matrix-warmdown-frac", type=float, default=1.0, help="LR multiplier floor for matrix params during warmdown (1.0 = no decay)")
parser.add_argument("--adamw-warmdown-frac", type=float, default=0.3, help="LR multiplier floor for AdamW params during warmdown")
# Source: nanochat PR #498 + modded-nanogpt #45 PR #160 — Hyperball optimizer (item 9)
parser.add_argument("--use-hyperball", action="store_true", help="use Hyperball optimizer for matrix params (norm projection instead of WD)")
# Source: modded-nanogpt record #39 PR #136 — heterogeneous batch sizes (item 22)
parser.add_argument("--adam-every-n", type=int, default=2, help="run AdamW groups every N steps (1=every step, 2=every other)")
# Source: modded-nanogpt record #53 PR #178 + arXiv:2404.19737 — multi-token prediction (item 23)
parser.add_argument("--mtp-schedule", type=str, default="", help="MTP weight schedule as comma-sep stages e.g. '1,0.5,0.25;1,0.5;1' (;-separated stages)")
# Source: modded-nanogpt record #51 PR #175 — tied embed -> untie mid-training (item 24)
parser.add_argument("--tie-embed-until", type=float, default=0.0, help="tie embed weights until this fraction of training (0.0 = disabled, 0.67 = untie at 2/3)")
# Source: modded-nanogpt record #46 PR #163 — batch size schedule (item 21)
parser.add_argument("--batch-schedule", type=str, default="", help="batch size schedule as comma-sep stages e.g. '262144:0.1,524288:0.5,1048576:1.0' (size:until_frac)")
# Source: modded-nanogpt #13/#31 + arXiv:2309.00071 — dynamic window warmup (item 17)
parser.add_argument("--dynamic-window", action="store_true", help="enable dynamic attention window warmup with YaRN")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens for val loss")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Compute init and wandb logging
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

if HAS_FA3:
    print0("Using Flash Attention 3 (Hopper GPU detected).")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("!" * 80)

# -----------------------------------------------------------------------------
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    # Source: arXiv:2305.13245 — GQA with CLI-specified n_kv_head (item 7)
    n_kv_head = args.n_kv_head if args.n_kv_head > 0 else num_heads
    assert num_heads % n_kv_head == 0, f"n_head ({num_heads}) must be divisible by n_kv_head ({n_kv_head})"
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=n_kv_head, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

# Source: modded-nanogpt record #51 PR #175 — tie embeddings at start (item 24)
if args.tie_embed_until > 0:
    model.tie_embeddings()
    print0(f"Embeddings tied until {args.tie_embed_until*100:.0f}% of training")

# Resume checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# FP8 training
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True
        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"FP8 training enabled ({args.fp8_recipe}) - converted {num_fp8_layers} layers, skipped {num_skipped}")

@contextmanager
def disable_fp8(model):
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))
    if not fp8_locations:
        yield
        return
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(fp8_module.in_features, fp8_module.out_features,
                          bias=fp8_module.bias is not None, device=fp8_module.weight.device,
                          dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile
orig_model = model
print0(f"torch.compile mode: {args.compile_mode}")
if args.compile_mode == "default":
    model = torch.compile(model, dynamic=False)
else:
    model = torch.compile(model, mode=args.compile_mode, dynamic=False)

# -----------------------------------------------------------------------------
# Scaling laws
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    return params_counts['transformer_matrices'] + params_counts['lm_head']

num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    # Source: nanochat PR #498 — Hyperball optimizer option (item 9)
    use_hyperball=args.use_hyperball,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# DataLoader
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Training horizon and schedulers
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated iterations from target ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")

# Source: nanochat PR #498 + modded-nanogpt #44 PR #158 — decoupled warmdown (item 12)
# Matrix params (Muon/Hyperball) keep higher LR during warmdown than AdamW params.
def get_lr_multiplier(it, is_matrix=False):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        if is_matrix:
            # Source: nanochat PR #498 — matrix params keep higher LR during warmdown (item 12)
            floor = args.matrix_warmdown_frac
        else:
            # Source: nanochat PR #498 — AdamW params decay to lower floor (item 12)
            floor = args.adamw_warmdown_frac
        return progress * 1.0 + (1 - progress) * floor

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Source: modded-nanogpt record #53 PR #178 — MTP weight schedule (item 23)
def get_mtp_weights(it):
    """Parse MTP schedule and return weights for current step."""
    if not args.mtp_schedule:
        return None
    stages = args.mtp_schedule.split(';')
    n_stages = len(stages)
    stage_idx = min(int(it / num_iterations * n_stages), n_stages - 1)
    weights_str = stages[stage_idx]
    return [float(w) for w in weights_str.split(',')]

# Source: modded-nanogpt record #46 PR #163 — batch size schedule (item 21)
def get_batch_config(it):
    """Parse batch schedule and return (batch_size, lr_scale) for current step."""
    if not args.batch_schedule:
        return total_batch_size, 1.0
    stages = args.batch_schedule.split(',')
    current_batch = total_batch_size
    for stage in stages:
        size_str, frac_str = stage.split(':')
        size = int(size_str)
        until_frac = float(frac_str)
        if it / num_iterations < until_frac:
            current_batch = size
            break
    # Source: arXiv:1711.00489 — scale LR proportionally to sqrt(batch/base_batch) (item 21)
    lr_scale = (current_batch / total_batch_size) ** 0.5
    return current_batch, lr_scale

# -----------------------------------------------------------------------------
# Training loop

if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Eval
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "val/bpb": val_bpb})
        model.train()

    # CORE metric
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "core_metric": results["core_metric"]})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = ["The capital of France is", "The opposite of hot is", "If 5*x + 3 = 13, then x is"]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), {
            "step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs,
            "user_config": user_config, "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len, "dataloader_state_dict": dataloader_state_dict,
            "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},
        }, rank=ddp_rank)

    if last_step:
        break

    # Source: modded-nanogpt record #51 PR #175 — untie embeddings mid-training (item 24)
    if args.tie_embed_until > 0:
        untie_step = int(args.tie_embed_until * num_iterations)
        if step == untie_step:
            print0(f"Step {step}: Untying embeddings (was tied until {args.tie_embed_until*100:.0f}%)")
            orig_model.untie_embeddings()

    # Source: modded-nanogpt #13/#31 + arXiv:2309.00071 — dynamic window warmup (item 17)
    if args.dynamic_window:
        orig_model.resize_windows(step, num_iterations)

    # -------------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()

    # Source: modded-nanogpt record #53 PR #178 — get MTP weights for this step (item 23)
    mtp_weights = get_mtp_weights(step)

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y, mtp_weights=mtp_weights)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Source: nanochat PR #498 — decoupled warmdown: different LR floors for matrix vs AdamW (item 12)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    # Source: modded-nanogpt record #46 PR #163 — batch size schedule LR adjustment (item 21)
    _current_batch, batch_lr_adj = get_batch_config(step)
    for group in optimizer.param_groups:
        is_matrix = group['kind'] == 'muon'
        lrm = get_lr_multiplier(step, is_matrix=is_matrix)
        group["lr"] = group["initial_lr"] * lrm * batch_lr_adj
        if is_matrix:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # Source: modded-nanogpt record #39 PR #136 — skip Adam on even steps (item 22)
    do_adam = (step % args.adam_every_n == args.adam_every_n - 1)
    optimizer.step(do_adam=do_adam)
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {get_lr_multiplier(step):.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        wandb_run.log({
            "step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss, "train/lrm": get_lr_multiplier(step),
            "train/dt": dt, "train/tok_per_sec": tok_per_sec, "train/mfu": mfu, "train/epoch": epoch,
        })

    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# Stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,
    {"Number of parameters": num_params, "FLOPs per token": f"{num_flops_per_token:e}",
     "Iterations": num_iterations, "Training tokens": total_tokens,
     "Token:Param ratio": total_batch_size * num_iterations / num_scaling_params,
     "DDP world size": ddp_world_size},
    {"Min val bpb": min_val_bpb if val_bpb is not None else None, "Final val bpb": val_bpb,
     "CORE metric": results.get("core_metric", None), "MFU": f"{mfu:.2f}%",
     "Total flops": f"{flops_so_far:e}", "Training time": f"{total_training_time/60:.2f}m",
     "Peak memory": f"{get_max_memory() / 1024 / 1024:.2f}MiB"},
])

wandb_run.finish()
compute_cleanup()

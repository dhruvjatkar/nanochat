# Attempts

Personal attempts at optimizing nanochat wallclock time to beat GPT-2.

## Structure

Each attempt is a numbered directory (e.g. `001-baseline`, `002-my-idea`):

```
attempts/
├── 001-baseline/
│   ├── speedrun.sh        # The launch script (submit this to Slurm)
│   ├── CHANGELOG.md       # What changed vs previous attempt
│   ├── scripts/           # (optional) Modified Python scripts override repo defaults
│   └── results.txt        # Auto-appended run timestamps
```

## How script overrides work

If an attempt has a `scripts/` subdirectory, it is prepended to `PYTHONPATH`.
This means `python -m scripts.base_train` will pick up the local copy first.
Only copy scripts you actually modify — unmodified ones fall through to the repo's `scripts/`.

## Submitting a run

```bash
# From explorer:
sbatch -p multigpu -t 3:00:00 --gres=gpu:h200:8 \
    --output=$HOME/nanochat/slurm-%j.out \
    --wrap "export PATH=\$HOME/.local/bin:\$PATH && cd ~/nanochat && bash attempts/001-baseline/speedrun.sh"
```

## Key files to optimize (wallclock time)

- `scripts/base_train.py` — training loop, optimizer setup, batch size, grad accum
- `nanochat/gpt.py` — model architecture (depth, attention, MLP)
- `nanochat/optim.py` — Muon/AdamW optimizer kernels
- `nanochat/dataloader.py` — data loading and bin packing
- `nanochat/flash_attention.py` — FA3 integration
- `runs/speedrun.sh` — upstream reference (do not modify, use as baseline)

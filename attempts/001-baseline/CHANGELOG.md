# Attempt 001: Baseline

## Description
Reproduces karpathy leaderboard Run 3 (commit 2c062aa) as-is. No modifications.

## Goal
- Target: CORE > 0.256525 (beat GPT-2)
- Reference time: ~2.76h on 8xH100

## Hyperparameters
- `depth`: 26
- `device-batch-size`: 16
- `total-batch-size`: auto (1M tokens for d26)
- `target-param-data-ratio`: 8.25
- `fp8`: enabled

## Reference (karpathy Run 3)
- Time: 2.76h
- CORE: 0.2602
- val_bpb: 0.74645
- Commit: 2c062aa

## Notes
This is the unmodified upstream speedrun. Future attempts will fork from here.

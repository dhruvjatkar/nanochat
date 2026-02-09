"""
Attempt-local nanochat package override.

Extends __path__ to include the root nanochat/ directory so that
non-overridden modules (e.g., common.py, tokenizer.py, flash_attention.py)
are imported from the repo root.

Overridden modules (gpt.py, optim.py, dataloader.py) are found here first
because $ATTEMPT_DIR is prepended to PYTHONPATH in speedrun.sh.
"""
import os as _os

# Navigate from attempts/002/fair/nanochat/ up to repo root nanochat/
_root_nanochat = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', 'nanochat')
)
if _os.path.isdir(_root_nanochat) and _root_nanochat not in __path__:
    __path__.append(_root_nanochat)

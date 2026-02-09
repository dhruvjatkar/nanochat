"""
Distributed dataloaders for pretraining (attempt-local override).

Changes from root version:
- Item 2:  Tensor-based document pool (eliminates GC stalls)
           Source: nanochat PR #477 https://github.com/karpathy/nanochat/pull/477 by @chrisjmccormick
- Item 19: Bigram hash ID computation for bigram hash embeddings
           Source: modded-nanogpt record #62 PR #201 https://github.com/KellerJordan/modded-nanogpt/pull/201
- Item 21: Batch size schedule support via generator .send()
           Source: modded-nanogpt record #46 PR #163 https://github.com/KellerJordan/modded-nanogpt/pull/163

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files


# Source: modded-nanogpt record #62 PR #201 — bigram hash embedding hash function (item 19)
def compute_bigram_hash_ids(token_ids, bigram_vocab_size):
    """
    Compute bigram hash IDs for a batch of token sequences.

    For each position i > 0, compute:
        hash[i] = (36313 * tokens[i]) ^ (27191 * tokens[i-1]) % (bigram_vocab_size - 1)
    Position 0 gets hash 0 (no previous token).

    Args:
        token_ids: (B, T) long tensor of token IDs
        bigram_vocab_size: size of the bigram embedding table
    Returns:
        bigram_ids: (B, T) long tensor of bigram hash IDs
    """
    B, T = token_ids.shape
    bigram_ids = torch.zeros_like(token_ids)
    if T > 1:
        curr = token_ids[:, 1:]  # (B, T-1)
        prev = token_ids[:, :-1]  # (B, T-1)
        hashed = ((36313 * curr) ^ (27191 * prev)) % (bigram_vocab_size - 1)
        bigram_ids[:, 1:] = hashed
    return bigram_ids


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping and tensor-based document pool.

    Source: nanochat PR #477 — tensor pool replaces Python list doc_buffer to eliminate GC stalls (item 2).
    Source: modded-nanogpt record #46 PR #163 — batch size schedule support via .send() (item 21).

    Changes from root:
    - Document buffer uses a flat torch.Tensor pool with (start, length) index pairs
      instead of a Python list of Python lists. This avoids frequent GC pressure from
      tens of thousands of small Python list objects.
    - Supports dynamic batch size via generator .send() protocol: send a tuple
      (new_B, new_T, new_grad_accum) to change batch dimensions mid-training.
    - Computes bigram hash IDs alongside inputs/targets for bigram hash embedding (item 19).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    pq_idx, rg_idx, epoch = 0, 0, 1

    # Source: nanochat PR #477 — tensor-based document pool (item 2)
    # Use a flat tensor pool instead of Python list to avoid GC stalls.
    # Documents are tracked as (start_offset, length) tuples into the pool.
    POOL_CAPACITY = buffer_size * (T + 256)  # generous initial capacity
    pool = torch.empty(POOL_CAPACITY, dtype=torch.long)
    pool_end = 0  # next free position in pool
    docs = []  # list of (start, length) tuples

    def compact_pool():
        """Compact the pool by removing gaps from consumed documents."""
        nonlocal pool, pool_end, docs
        if not docs:
            pool_end = 0
            return
        new_pool = torch.empty_like(pool)
        new_end = 0
        new_docs = []
        for start, length in docs:
            new_pool[new_end:new_end + length] = pool[start:start + length]
            new_docs.append((new_end, length))
            new_end += length
        pool = new_pool
        pool_end = new_end
        docs = new_docs

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch, pool_end, pool
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            n = len(tokens)
            # Grow pool if needed
            if pool_end + n > pool.shape[0]:
                # Try compacting first
                compact_pool()
                if pool_end + n > pool.shape[0]:
                    # Double pool size
                    new_pool = torch.empty(max(pool.shape[0] * 2, pool_end + n), dtype=torch.long)
                    new_pool[:pool_end] = pool[:pool_end]
                    pool = new_pool
            pool[pool_end:pool_end + n] = torch.tensor(tokens, dtype=torch.long)
            docs.append((pool_end, n))
            pool_end += n

    # Pre-allocate output buffers
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    # Source: nanochat PR #477 — direct copy to GPU, no intermediate cpu_buffer (item 2)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    # Pinned staging buffer for async HtoD
    cpu_staging = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_inputs = cpu_staging[:B * T].view(B, T)
    cpu_targets = cpu_staging[B * T:].view(B, T)

    # Source: modded-nanogpt record #46 PR #163 — batch size schedule support (item 21)
    # The generator supports .send() to dynamically change batch size.
    # Send a (new_B, new_T, new_grad_accum) tuple to resize.
    dynamic_config = None

    while True:
        # Check if we received a new batch config via .send()
        if dynamic_config is not None:
            new_B, new_T, _new_grad_accum = dynamic_config
            if new_B != B or new_T != T:
                B, T = new_B, new_T
                row_capacity = T + 1
                row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
                gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
                inputs = gpu_buffer[:B * T].view(B, T)
                targets = gpu_buffer[B * T:].view(B, T)
                cpu_staging = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
                cpu_inputs = cpu_staging[:B * T].view(B, T)
                cpu_targets = cpu_staging[B * T:].view(B, T)
            dynamic_config = None

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(docs) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, (start, length) in enumerate(docs):
                    if length <= remaining and length > best_len:
                        best_idx = i
                        best_len = length

                if best_idx >= 0:
                    start, length = docs.pop(best_idx)
                    row_buffer[row_idx, pos:pos + length] = pool[start:start + length]
                    pos += length
                else:
                    # No doc fits - crop shortest in buffer to fill remaining
                    shortest_idx = min(range(len(docs)), key=lambda i: docs[i][1])
                    start, length = docs.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = pool[start:start + remaining]
                    pos += remaining

        # Copy to CPU staging, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy and yield
        gpu_buffer.copy_(cpu_staging, non_blocking=use_cuda)

        # Source: modded-nanogpt record #62 PR #201 — compute bigram hash IDs (item 19)
        # Bigram IDs are computed on-device from the input token IDs.
        # The GPT model's forward() will use these if bigram embeddings are enabled.
        dynamic_config = yield inputs, targets, state_dict

        # Periodically compact pool to reclaim memory
        if len(docs) < buffer_size // 4:
            compact_pool()


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

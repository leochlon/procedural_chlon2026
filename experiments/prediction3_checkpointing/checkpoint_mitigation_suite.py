#!/usr/bin/env python3
# Research-quality experiment files used to generate the results for procedural_chlon2026.
"""
checkpoint_mitigation_suite.py

Option B: Evaluate checkpointing + answer-gating as *validated interventions*.

What this script measures
-------------------------
For each (model, task, filler_type, k) condition it evaluates:
  1) Baseline prompt (no checkpoint)
  2) Checkpointed prompt (oracle checkpoint inserted periodically into the *tail filler* region)
And for each prompt variant it reports:
  - acc_global: argmax over full vocab
  - acc_candidate_only: answer-gated decoding (argmax over candidate set)
  - frac_stage2a: wrong & non-candidate (format/continuation)
  - frac_stage2b: wrong & candidate (wrong binding among candidates)
  - mean_gate_gap: (best candidate logit) - (best non-candidate logit)
  - mean_value_gap: (target logit) - (best wrong-candidate logit)

This directly addresses the reviewer: "you propose mitigations but don't evaluate them."
Answer-gating is evaluated via candidate-only decoding; checkpointing is evaluated via prompt-level intervention.

Design choices (deliberate)
---------------------------
- Oracle checkpoint: we insert a compact state serialization that includes the *true* binding(s).
  This matches the paper's "checkpointing" design principle: externalize state to reduce effective distance.
- Tail-only checkpointing: we insert checkpoints only into the final filler region before the question
  (avoids weirdness like mentioning KEY2 before it's defined).
- Paired evaluation: baseline and checkpointed prompts use the *same* underlying sampled example specs.
- Memory-safe batching: examples are generated lazily per batch; no storing all examples.

Multi-GPU
---------
Run with torchrun for DDP-style data parallelism (each GPU evaluates a disjoint subset of examples).
We all-reduce scalar stats and gather small top-wrong histograms.

Example (8 GPUs):
  torchrun --nproc_per_node 8 checkpoint_mitigation_suite.py \
    --models Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct \
    --tasks decoy_injection competing_vars primacy_recency \
    --k_values 128 256 512 1024 \
    --filler_types repeat decoy_heavy \
    --trials_per_k 2000 \
    --batch_size 4 \
    --checkpoint_every 64 128 \
    --dtype bf16 \
    --attn_impl sdpa \
    --out_csv checkpoint_eval.csv \
    --out_json checkpoint_eval.json

Notes
-----
- For very large k, reduce --batch_size. Attention is O(n^2) unless using flash attention.
- Set env var to reduce fragmentation:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# DDP utilities
# -------------------------

def ddp_init() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl")
        return rank, world, local
    return 0, 1, 0

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_barrier():
    if is_dist():
        dist.barrier()

def all_reduce_(t: torch.Tensor, op=dist.ReduceOp.SUM):
    if is_dist():
        dist.all_reduce(t, op=op)
    return t

def all_gather_object(obj):
    if not is_dist():
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


# -------------------------
# General utilities
# -------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def try_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def safe_decode(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        return f"<tok:{tid}>"


# -------------------------
# Pool + fillers
# -------------------------

DEFAULT_WORDS = [
    "apple","banana","cherry","grape","lemon","mango","peach","pear","plum","kiwi","melon","berry",
    "alpha","beta","gamma","delta","theta","lambda","omega","sigma","kappa",
    "red","blue","green","yellow","black","white","orange","purple",
    "cat","dog","mouse","lion","tiger","bear","wolf","fox","zebra","shark","whale",
    "yes","no","true","false","left","right","up","down","east","west",
]

COHERENT_TEXT = (
    "In the early morning the city was quiet and the streets were empty. "
    "People walked to work and the sky turned pale blue above the buildings. "
    "A small cafe opened its doors and served coffee to a few tired customers. "
)

@dataclass
class Pool:
    token_ids: List[int]
    filler_id: int
    random_ids: List[int]

def build_pool(tokenizer, num_values: int, seed: int) -> Pool:
    rng = random.Random(seed)
    special = set(tokenizer.all_special_ids)

    ids: List[int] = []
    words = list(DEFAULT_WORDS)
    rng.shuffle(words)
    for w in words:
        for form in [f" {w}", w]:
            tid = tokenizer.encode(form, add_special_tokens=False)
            if len(tid) == 1 and tid[0] not in special:
                if tid[0] not in ids:
                    ids.append(int(tid[0]))
                break
        if len(ids) >= num_values:
            break
    if len(ids) < max(16, num_values // 2):
        raise RuntimeError(f"Could not build enough single-token values. Got {len(ids)}")

    filler_id = None
    for cand in [" the", " and", " of", ".", ",", " "]:
        tid = tokenizer.encode(cand, add_special_tokens=False)
        if len(tid) == 1 and tid[0] not in special:
            filler_id = int(tid[0])
            break
    if filler_id is None:
        filler_id = ids[0]

    random_ids: List[int] = []
    for tid in range(tokenizer.vocab_size):
        if tid in special or tid in set(ids):
            continue
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s or len(s) > 12:
            continue
        if "\uFFFD" in s:
            continue
        random_ids.append(int(tid))
        if len(random_ids) >= 8000:
            break
    if len(random_ids) < 1000:
        random_ids = [t for t in range(tokenizer.vocab_size) if t not in special][:2000]

    return Pool(token_ids=ids, filler_id=filler_id, random_ids=random_ids)

def make_filler(tokenizer, pool: Pool, filler_type: str, k: int, rng: random.Random,
                *, decoy_id: Optional[int] = None, decoy_reps: int = 10) -> List[int]:
    filler_type = filler_type.lower()
    if k <= 0:
        return []
    if filler_type == "repeat":
        return [pool.filler_id] * k
    if filler_type == "coherent":
        ids = tokenizer.encode(COHERENT_TEXT, add_special_tokens=False)
        if not ids:
            return [pool.filler_id] * k
        out = []
        while len(out) < k:
            out.extend(ids)
        return out[:k]
    if filler_type == "random":
        return [rng.choice(pool.random_ids) for _ in range(k)]
    if filler_type == "decoy_heavy":
        out = [pool.filler_id] * k
        if decoy_id is None:
            return out
        reps = min(decoy_reps, k)
        for i in range(reps):
            pos = int((i + 1) * (k / (reps + 1)))
            pos = max(0, min(k - 1, pos))
            out[pos] = int(decoy_id)
        for _ in range(min(8, k // 16)):
            out[rng.randrange(0, k)] = rng.choice(pool.random_ids)
        return out
    raise ValueError(f"Unknown filler_type: {filler_type}")


# -------------------------
# Tasks + checkpointing
# -------------------------

@dataclass
class ExampleSpec:
    # shared
    task: str
    candidates: List[int]  # includes target
    target: int
    # task-specific
    key1: Optional[int] = None  # for competing_vars
    key2: Optional[int] = None
    key_first: Optional[int] = None  # for primacy_recency
    key_mid: Optional[int] = None
    key_last: Optional[int] = None
    decoy: Optional[int] = None  # for decoy_injection
    # fillers
    fillers: List[List[int]] = dataclasses.field(default_factory=list)  # segments

def sample_candidates(pool: Pool, rng: random.Random, include: List[int], n_distractors: int) -> List[int]:
    cand = [int(x) for x in include]
    extra = [t for t in pool.token_ids if t not in set(cand)]
    rng.shuffle(extra)
    cand += [int(x) for x in extra[:max(0, n_distractors)]]
    rng.shuffle(cand)
    return cand

def make_spec(task: str, tok, pool: Pool, k: int, filler_type: str, *,
              n_distractors: int, decoy_reps: int, seed: int) -> ExampleSpec:
    rng = random.Random(seed)
    task = task.lower().strip()

    if task == "competing_vars":
        v1 = rng.choice(pool.token_ids)
        v2 = rng.choice([t for t in pool.token_ids if t != v1])
        cand = sample_candidates(pool, rng, include=[v1, v2], n_distractors=n_distractors)
        filler1 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v2, decoy_reps=decoy_reps)
        filler2 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v2, decoy_reps=decoy_reps)
        return ExampleSpec(task=task, candidates=cand, target=int(v1), key1=int(v1), key2=int(v2), fillers=[filler1, filler2])

    if task == "primacy_recency":
        v1 = rng.choice(pool.token_ids)
        v2 = rng.choice([t for t in pool.token_ids if t != v1])
        v3 = rng.choice([t for t in pool.token_ids if t not in (v1, v2)])
        cand = sample_candidates(pool, rng, include=[v1, v2, v3], n_distractors=n_distractors)
        f = make_filler(tok, pool, filler_type, k, rng, decoy_id=v3, decoy_reps=decoy_reps)
        # 3 fillers between assignments + one tail filler
        f1 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v3, decoy_reps=decoy_reps)
        f2 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v3, decoy_reps=decoy_reps)
        f3 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v3, decoy_reps=decoy_reps)
        return ExampleSpec(task=task, candidates=cand, target=int(v1), key_first=int(v1), key_mid=int(v2), key_last=int(v3),
                           fillers=[f1, f2, f3])

    if task == "decoy_injection":
        v = rng.choice(pool.token_ids)
        decoy = rng.choice([t for t in pool.token_ids if t != v])
        cand = sample_candidates(pool, rng, include=[v, decoy], n_distractors=n_distractors)
        filler = make_filler(tok, pool, filler_type, k, rng, decoy_id=decoy, decoy_reps=decoy_reps)
        return ExampleSpec(task=task, candidates=cand, target=int(v), decoy=int(decoy), fillers=[filler])

    raise ValueError(f"Unknown task: {task}")

def checkpoint_tokens_for_spec(tok, spec: ExampleSpec, prompt_style: str) -> List[int]:
    """
    Compact oracle checkpoint serialization.
    """
    if prompt_style != "bracket":
        # we support eq too, but bracket is default; keep the checkpoint neutral.
        # Use a fixed natural-language-ish checkpoint to avoid relying on '=' formatting.
        if spec.task == "competing_vars":
            pre = "\nCHECKPOINT: KEY1 is"
            mid = " ; KEY2 is"
            suf = ".\n"
            return tok.encode(pre, add_special_tokens=False) + [int(spec.key1)] + tok.encode(mid, add_special_tokens=False) + [int(spec.key2)] + tok.encode(suf, add_special_tokens=False)
        if spec.task == "primacy_recency":
            pre = "\nCHECKPOINT: FIRST(KEY) is"
            mid = " ; LAST(KEY) is"
            suf = ".\n"
            return tok.encode(pre, add_special_tokens=False) + [int(spec.key_first)] + tok.encode(mid, add_special_tokens=False) + [int(spec.key_last)] + tok.encode(suf, add_special_tokens=False)
        if spec.task == "decoy_injection":
            pre = "\nCHECKPOINT: KEY is"
            suf = ".\n"
            return tok.encode(pre, add_special_tokens=False) + [int(spec.target)] + tok.encode(suf, add_special_tokens=False)

    # bracket style
    if spec.task == "competing_vars":
        pre = "\nCHECKPOINT: KEY1=["
        mid = "] KEY2=["
        suf = "]\n"
        return tok.encode(pre, add_special_tokens=False) + [int(spec.key1)] + tok.encode(mid, add_special_tokens=False) + [int(spec.key2)] + tok.encode(suf, add_special_tokens=False)

    if spec.task == "primacy_recency":
        pre = "\nCHECKPOINT: FIRST(KEY)=["
        mid = "] LAST(KEY)=["
        suf = "]\n"
        return tok.encode(pre, add_special_tokens=False) + [int(spec.key_first)] + tok.encode(mid, add_special_tokens=False) + [int(spec.key_last)] + tok.encode(suf, add_special_tokens=False)

    if spec.task == "decoy_injection":
        pre = "\nCHECKPOINT: KEY=["
        suf = "]\n"
        return tok.encode(pre, add_special_tokens=False) + [int(spec.target)] + tok.encode(suf, add_special_tokens=False)

    raise ValueError(f"Unknown task for checkpoint: {spec.task}")

def insert_periodic_checkpoint(filler: List[int], chk: List[int], every: int) -> List[int]:
    if every <= 0 or not chk or len(filler) <= every:
        return list(filler)
    out: List[int] = []
    i = 0
    n = len(filler)
    while i < n:
        j = min(n, i + every)
        out.extend(filler[i:j])
        i = j
        if i < n:
            out.extend(chk)
    return out

def build_prompt(tok, spec: ExampleSpec, prompt_style: str, checkpoint_every: int) -> List[int]:
    """
    Build token IDs for the prompt. Checkpointing is applied to the *tail* filler segment only.
    """
    ids: List[int] = []
    if tok.bos_token_id is not None:
        ids.append(int(tok.bos_token_id))

    style = prompt_style.lower().strip()
    task = spec.task

    if task == "competing_vars":
        assert spec.key1 is not None and spec.key2 is not None
        f1, f2 = spec.fillers
        # Apply checkpoints only to tail filler (f2), after both keys are defined.
        if checkpoint_every > 0:
            chk = checkpoint_tokens_for_spec(tok, spec, prompt_style=style)
            f2 = insert_periodic_checkpoint(f2, chk, checkpoint_every)

        if style == "bracket":
            ids += tok.encode("KEY1=[", add_special_tokens=False)
            ids.append(int(spec.key1))
            ids += tok.encode("]\n", add_special_tokens=False)
            ids += f1
            ids += tok.encode("\nKEY2=[", add_special_tokens=False)
            ids.append(int(spec.key2))
            ids += tok.encode("]\n", add_special_tokens=False)
            ids += f2
            ids += tok.encode("\nQuestion: What is KEY1?\nKEY1=[", add_special_tokens=False)
        else:
            ids += tok.encode("KEY1 =", add_special_tokens=False)
            ids.append(int(spec.key1))
            ids += f1
            ids += tok.encode("\nKEY2 =", add_special_tokens=False)
            ids.append(int(spec.key2))
            ids += f2
            ids += tok.encode("\nQuestion: What is KEY1?\nKEY1 =", add_special_tokens=False)
        return ids

    if task == "primacy_recency":
        assert spec.key_first is not None and spec.key_mid is not None and spec.key_last is not None
        f1, f2, f3 = spec.fillers
        # Apply checkpoints only to tail filler (f3), after the last assignment.
        if checkpoint_every > 0:
            chk = checkpoint_tokens_for_spec(tok, spec, prompt_style=style)
            f3 = insert_periodic_checkpoint(f3, chk, checkpoint_every)

        if style == "bracket":
            ids += tok.encode("KEY=[", add_special_tokens=False); ids.append(int(spec.key_first)); ids += tok.encode("]\n", add_special_tokens=False)
            ids += f1
            ids += tok.encode("\nKEY=[", add_special_tokens=False); ids.append(int(spec.key_mid)); ids += tok.encode("]\n", add_special_tokens=False)
            ids += f2
            ids += tok.encode("\nKEY=[", add_special_tokens=False); ids.append(int(spec.key_last)); ids += tok.encode("]\n", add_special_tokens=False)
            ids += f3
            ids += tok.encode("\nQuestion: What was the FIRST value of KEY?\nKEY=[", add_special_tokens=False)
        else:
            ids += tok.encode("KEY =", add_special_tokens=False); ids.append(int(spec.key_first))
            ids += f1
            ids += tok.encode("\nKEY =", add_special_tokens=False); ids.append(int(spec.key_mid))
            ids += f2
            ids += tok.encode("\nKEY =", add_special_tokens=False); ids.append(int(spec.key_last))
            ids += f3
            ids += tok.encode("\nQuestion: What was the FIRST value of KEY?\nKEY =", add_special_tokens=False)
        return ids

    if task == "decoy_injection":
        assert spec.decoy is not None
        f = spec.fillers[0]
        # For decoy_injection, checkpoint inside the only filler segment.
        if checkpoint_every > 0:
            chk = checkpoint_tokens_for_spec(tok, spec, prompt_style=style)
            f = insert_periodic_checkpoint(f, chk, checkpoint_every)

        if style == "bracket":
            ids += tok.encode("KEY=[", add_special_tokens=False); ids.append(int(spec.target)); ids += tok.encode("]\n", add_special_tokens=False)
            ids += f
            ids += tok.encode("\nQuestion: What is KEY?\nKEY=[", add_special_tokens=False)
        else:
            ids += tok.encode("KEY =", add_special_tokens=False); ids.append(int(spec.target))
            ids += f
            ids += tok.encode("\nQuestion: What is KEY?\nKEY =", add_special_tokens=False)
        return ids

    raise ValueError(f"Unknown task in build_prompt: {task}")


# -------------------------
# Evaluation
# -------------------------

@dataclass
class Stats:
    n_total: int = 0
    n_correct: int = 0
    n_correct_gated: int = 0
    n_stage2a: int = 0
    n_stage2b: int = 0
    sum_gate_gap: float = 0.0
    sum_value_gap: float = 0.0
    wrong_non_cand: Dict[int, int] = dataclasses.field(default_factory=dict)
    wrong_cand: Dict[int, int] = dataclasses.field(default_factory=dict)

def collate(tokenizer, seqs: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_len = max(len(s) for s in seqs)
    bsz = len(seqs)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :L] = 1
    return input_ids, attn

@torch.no_grad()
def forward_last_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_attentions=False)
    return out.logits[:, -1, :].detach()  # [B,V]

def update_stats(stats: Stats, logits: torch.Tensor, spec: ExampleSpec):
    """
    logits: [V] (single example)
    """
    cand = [int(x) for x in spec.candidates]
    cand_set = set(cand)

    pred = int(torch.argmax(logits).item())

    # gated prediction: argmax over candidates
    cand_logits = logits[cand]
    cand_arg = int(torch.argmax(cand_logits).item())
    pred_gated = int(cand[cand_arg])

    target = int(spec.target)

    # gate gap: best candidate vs best non-candidate
    best_cand_logit = float(torch.max(cand_logits).item())

    # best non-candidate: mask candidate indices
    masked = logits.clone()
    masked[cand] = -1e9
    best_non_cand = float(torch.max(masked).item())
    gate_gap = best_cand_logit - best_non_cand

    # value gap: target vs best wrong-candidate
    # (if target not in candidates, this is ill-defined, but target is always included)
    tpos = None
    for i, tid in enumerate(cand):
        if tid == target:
            tpos = i
            break
    assert tpos is not None
    cand_logits_wo = cand_logits.clone()
    cand_logits_wo[tpos] = -1e9
    best_wrong_cand = float(torch.max(cand_logits_wo).item())
    value_gap = float(logits[target].item()) - best_wrong_cand

    stats.n_total += 1
    stats.sum_gate_gap += gate_gap
    stats.sum_value_gap += value_gap

    if pred == target:
        stats.n_correct += 1
    else:
        if pred in cand_set:
            stats.n_stage2b += 1
            stats.wrong_cand[pred] = stats.wrong_cand.get(pred, 0) + 1
        else:
            stats.n_stage2a += 1
            stats.wrong_non_cand[pred] = stats.wrong_non_cand.get(pred, 0) + 1

    if pred_gated == target:
        stats.n_correct_gated += 1

def merge_hist_dicts(dicts: List[Dict[int, int]], topk: int = 20) -> Dict[int, int]:
    merged: Dict[int, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[int(k)] = merged.get(int(k), 0) + int(v)
    # prune
    items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return {k: v for k, v in items}

def stats_to_row(tok, model_name: str, task: str, filler_type: str, k: int,
                 checkpoint_every: int, variant: str, stats: Stats) -> dict:
    n = max(1, stats.n_total)
    top_non = sorted(stats.wrong_non_cand.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_cand = sorted(stats.wrong_cand.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_non_str = [(safe_decode(tok, tid), int(c)) for tid, c in top_non]
    top_cand_str = [(safe_decode(tok, tid), int(c)) for tid, c in top_cand]
    return {
        "model": model_name,
        "task": task,
        "prompt_style": None,  # filled by caller
        "filler_type": filler_type,
        "k": int(k),
        "checkpoint_every": int(checkpoint_every),
        "variant": variant,  # "baseline" or "checkpoint"
        "n_total": int(stats.n_total),
        "acc_global": float(stats.n_correct / n),
        "acc_candidate_only": float(stats.n_correct_gated / n),
        "n_stage2a": int(stats.n_stage2a),
        "n_stage2b": int(stats.n_stage2b),
        "frac_stage2a": float(stats.n_stage2a / max(1, stats.n_stage2a + stats.n_stage2b)),
        "frac_stage2b": float(stats.n_stage2b / max(1, stats.n_stage2a + stats.n_stage2b)),
        "mean_gate_gap": float(stats.sum_gate_gap / n),
        "mean_value_gap": float(stats.sum_value_gap / n),
        "top_wrong_non_candidates": json.dumps(top_non_str),
        "top_wrong_candidates": json.dumps(top_cand_str),
    }

@torch.no_grad()
def eval_condition_paired(model, tok, pool: Pool, device: torch.device,
                          *, model_name: str, task: str, prompt_style: str,
                          filler_type: str, k: int,
                          trials: int, batch_size: int,
                          n_distractors: int, decoy_reps: int,
                          checkpoint_every: int,
                          seed_base: int) -> Tuple[Stats, Optional[Stats]]:
    """
    Returns: (baseline_stats, checkpoint_stats or None if checkpoint_every<=0)
    Each rank evaluates a shard of examples by index.
    """
    rank = dist.get_rank() if is_dist() else 0
    world = dist.get_world_size() if is_dist() else 1

    # shard indices
    indices = list(range(rank, trials, world))

    base = Stats()
    chk = Stats() if checkpoint_every > 0 else None

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        specs: List[ExampleSpec] = []
        for gi in batch_idx:
            spec = make_spec(task, tok, pool, k, filler_type,
                             n_distractors=n_distractors,
                             decoy_reps=decoy_reps,
                             seed=seed_base + gi * 1009)
            specs.append(spec)

        # baseline
        seqs_base = [build_prompt(tok, s, prompt_style, checkpoint_every=0) for s in specs]
        input_ids, attn = collate(tok, seqs_base, device=device)
        logits = forward_last_logits(model, input_ids, attn)  # [B,V]
        for bi, spec in enumerate(specs):
            update_stats(base, logits[bi], spec)
        del input_ids, attn, logits, seqs_base
        gc.collect()
        try_empty_cache()

        # checkpointed
        if chk is not None:
            seqs_chk = [build_prompt(tok, s, prompt_style, checkpoint_every=checkpoint_every) for s in specs]
            input_ids, attn = collate(tok, seqs_chk, device=device)
            logits = forward_last_logits(model, input_ids, attn)
            for bi, spec in enumerate(specs):
                update_stats(chk, logits[bi], spec)
            del input_ids, attn, logits, seqs_chk
            gc.collect()
            try_empty_cache()

    return base, chk

def reduce_stats(stats: Stats) -> Stats:
    """
    All-reduce scalar fields; gather hist dicts (top-20 per rank), merge on rank0, broadcast merged.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    t = torch.tensor([
        stats.n_total,
        stats.n_correct,
        stats.n_correct_gated,
        stats.n_stage2a,
        stats.n_stage2b,
    ], dtype=torch.float64, device=device)
    s = torch.tensor([
        stats.sum_gate_gap,
        stats.sum_value_gap,
    ], dtype=torch.float64, device=device)

    all_reduce_(t)
    all_reduce_(s)

    out = Stats(
        n_total=int(t[0].item()),
        n_correct=int(t[1].item()),
        n_correct_gated=int(t[2].item()),
        n_stage2a=int(t[3].item()),
        n_stage2b=int(t[4].item()),
        sum_gate_gap=float(s[0].item()),
        sum_value_gap=float(s[1].item()),
        wrong_non_cand={},
        wrong_cand={},
    )

    # gather hist dicts (prune locally first)
    local_non = merge_hist_dicts([stats.wrong_non_cand], topk=20)
    local_cand = merge_hist_dicts([stats.wrong_cand], topk=20)
    gathered_non = all_gather_object(local_non)
    gathered_cand = all_gather_object(local_cand)

    if (dist.get_rank() if is_dist() else 0) == 0:
        merged_non = merge_hist_dicts(gathered_non, topk=20)
        merged_cand = merge_hist_dicts(gathered_cand, topk=20)
    else:
        merged_non = None
        merged_cand = None

    if is_dist():
        obj_list = [merged_non, merged_cand]
        dist.broadcast_object_list(obj_list, src=0)
        merged_non, merged_cand = obj_list

    out.wrong_non_cand = merged_non or {}
    out.wrong_cand = merged_cand or {}
    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--tasks", type=str, nargs="+", default=["decoy_injection", "competing_vars", "primacy_recency"])
    ap.add_argument("--prompt_style", type=str, default="bracket", choices=["bracket", "eq"])
    ap.add_argument("--k_values", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--filler_types", type=str, nargs="+", default=["repeat", "decoy_heavy"])
    ap.add_argument("--trials_per_k", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--num_values", type=int, default=128)
    ap.add_argument("--n_distractors", type=int, default=48)
    ap.add_argument("--decoy_reps", type=int, default=12)

    ap.add_argument("--checkpoint_every", type=int, nargs="+", default=[0, 64, 128])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2", "auto"])
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="checkpoint_eval.csv")
    ap.add_argument("--out_json", type=str, default="checkpoint_eval.json")
    ap.add_argument("--outdir", type=str, default="checkpoint_eval_out")

    args = ap.parse_args()

    rank, world, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed + 1000 * rank)

    ensure_dir(args.outdir)

    dtype = torch_dtype_from_str(args.dtype)

    all_rows: List[dict] = []

    for model_name in args.models:
        if rank == 0:
            print(f"\n=== Loading model: {model_name} ===")
        ddp_barrier()

        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        model_kwargs = dict(trust_remote_code=True)
        if args.attn_impl != "auto":
            model_kwargs["attn_implementation"] = args.attn_impl
        if not args.load_in_8bit:
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        model.to(device)

        pool = build_pool(tok, num_values=args.num_values, seed=args.seed + 77)

        for task in args.tasks:
            for filler_type in args.filler_types:
                for k in args.k_values:
                    for chk_every in args.checkpoint_every:
                        if rank == 0:
                            print(f"[{model_name}] task={task} filler={filler_type} k={k} chk_every={chk_every} trials={args.trials_per_k}")

                        base_local, chk_local = eval_condition_paired(
                            model, tok, pool, device,
                            model_name=model_name,
                            task=task,
                            prompt_style=args.prompt_style,
                            filler_type=filler_type,
                            k=int(k),
                            trials=int(args.trials_per_k),
                            batch_size=int(args.batch_size),
                            n_distractors=int(args.n_distractors),
                            decoy_reps=int(args.decoy_reps),
                            checkpoint_every=int(chk_every),
                            seed_base=args.seed + 1234567,
                        )

                        base = reduce_stats(base_local)
                        chk = reduce_stats(chk_local) if chk_local is not None else None

                        if rank == 0:
                            row_b = stats_to_row(tok, model_name, task, filler_type, k, chk_every, "baseline", base)
                            row_b["prompt_style"] = args.prompt_style
                            all_rows.append(row_b)

                            if chk is not None:
                                row_c = stats_to_row(tok, model_name, task, filler_type, k, chk_every, "checkpoint", chk)
                                row_c["prompt_style"] = args.prompt_style
                                all_rows.append(row_c)

                                # paired deltas (for convenience)
                                delta = {
                                    "model": model_name,
                                    "task": task,
                                    "prompt_style": args.prompt_style,
                                    "filler_type": filler_type,
                                    "k": int(k),
                                    "checkpoint_every": int(chk_every),
                                    "variant": "delta(checkpoint-baseline)",
                                    "n_total": row_b["n_total"],
                                    "acc_global": row_c["acc_global"] - row_b["acc_global"],
                                    "acc_candidate_only": row_c["acc_candidate_only"] - row_b["acc_candidate_only"],
                                    "frac_stage2a": row_c["frac_stage2a"] - row_b["frac_stage2a"],
                                    "frac_stage2b": row_c["frac_stage2b"] - row_b["frac_stage2b"],
                                    "mean_gate_gap": row_c["mean_gate_gap"] - row_b["mean_gate_gap"],
                                    "mean_value_gap": row_c["mean_value_gap"] - row_b["mean_value_gap"],
                                    "n_stage2a": row_c["n_stage2a"] - row_b["n_stage2a"],
                                    "n_stage2b": row_c["n_stage2b"] - row_b["n_stage2b"],
                                    "top_wrong_non_candidates": "",
                                    "top_wrong_candidates": "",
                                }
                                all_rows.append(delta)

                        # cleanup between conditions
                        gc.collect()
                        try_empty_cache()

        # unload model
        del model
        gc.collect()
        try_empty_cache()
        ddp_barrier()

    if rank == 0:
        df = pd.DataFrame(all_rows)
        out_csv = os.path.join(args.outdir, args.out_csv) if not os.path.isabs(args.out_csv) else args.out_csv
        out_json = os.path.join(args.outdir, args.out_json) if not os.path.isabs(args.out_json) else args.out_json
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nWrote: {out_csv}")
        print(f"Wrote: {out_json}")

    ddp_barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

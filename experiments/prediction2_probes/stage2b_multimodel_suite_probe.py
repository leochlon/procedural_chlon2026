#!/usr/bin/env python3
# Research-quality experiment files used to generate the results for procedural_chlon2026.
"""
stage2b_multimodel_suite_probe.py

Stage-2B (binding failure) benchmark suite: designed to produce errors where the model outputs
a WRONG CANDIDATE VALUE (Stage 2B), not just format/filler tokens (Stage 2A gating).

Implements experiments requested:

1) Competing Defined Variables (KEY1 vs KEY2)
2) Adversarial Decoy Injection (target vs high-frequency decoy)
3) Strawberry-style WORD vs TOTAL vs SUFFIX binding (baseline + factorized variant)
4) Primacy vs Recency binding ("FIRST value of KEY")
5) Bracket + long distance: achieved by choosing large --k_values and adversarial/decoy fillers

Multi-GPU + multi-model support:
- Run with torchrun.
- Optionally shard GPUs by model: different process groups run different models concurrently.

Outputs:
- CSV summary with:
    acc_global, acc_candidate_only,
    frac_errors_stage2a (pred not in candidates),
    frac_errors_stage2b (pred in candidates but wrong),
    mean gate_gap and value_gap,
    top wrong non-candidates and top wrong candidates (decoded)
- JSON with richer per-condition metadata.

Example (8 GPUs, 4 models concurrently with 2 GPUs each):
  torchrun --nproc_per_node 8 stage2b_multimodel_suite.py \
    --models Qwen/Qwen2.5-3B-Instruct google/gemma-2-2b-it meta-llama/Llama-3.2-3B microsoft/Phi-3-mini-4k-instruct \
    --shard_by_model \
    --tasks competing_vars decoy_injection primacy_recency strawberry_baseline strawberry_factorized \
    --k_values 128 256 512 1024 2048 \
    --filler_types repeat coherent random decoy_heavy \
    --trials_per_condition 2000 \
    --batch_size 8 \
    --decoy_reps 12 \
    --n_distractors 6 \
    --dtype bf16 \
    --attn_impl sdpa \
    --outdir stage2b_results \
    --out_csv stage2b_summary.csv \
    --out_json stage2b_summary.json

Notes:
- This suite is intentionally "closed-system": we pre-build traces so Stage 1 is satisfied for strawberry.
- For robust Stage-2B, prefer:
    --prompt_style bracket (default)
    --filler_types decoy_heavy or random with --decoy_reps >= 10
    large --k_values
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Distributed utils
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


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_tensor(t: torch.Tensor, group=None, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(t, op=op, group=group)
    return t


def all_gather_object(obj, group=None):
    if not is_dist():
        return [obj]
    out = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(out, obj, group=group)
    return out


# -------------------------
# General utils
# -------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def try_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        return f"<tok:{token_id}>"


# -------------------------
# Candidate pools
# -------------------------

DEFAULT_WORDS = [
    # fruits
    "apple","banana","cherry","grape","lemon","mango","peach","pear","plum","kiwi","melon","berry",
    # greek
    "alpha","beta","gamma","delta","theta","lambda","omega","sigma","kappa",
    # colors
    "red","blue","green","yellow","black","white","orange","purple",
    # animals
    "cat","dog","mouse","lion","tiger","bear","wolf","fox","zebra","shark","whale",
    # misc short tokens
    "yes","no","true","false","left","right","up","down","east","west",
]

@dataclass
class Pool:
    token_ids: List[int]
    id_to_str: Dict[int, str]
    filler_id: int
    random_ids: List[int]


def build_word_pool(tokenizer, num_values: int, seed: int) -> Optional[Pool]:
    rng = random.Random(seed)
    special = set(tokenizer.all_special_ids)
    ids: List[int] = []
    id_to_str: Dict[int, str] = {}

    # Prefer leading-space single tokens (BPE common)
    words = list(DEFAULT_WORDS)
    rng.shuffle(words)

    for w in words:
        for form in [f" {w}", w]:
            tid = tokenizer.encode(form, add_special_tokens=False)
            if len(tid) == 1 and tid[0] not in special:
                t = tid[0]
                if t not in id_to_str:
                    ids.append(t)
                    id_to_str[t] = w
                break
        if len(ids) >= num_values:
            break

    if len(ids) < max(8, num_values // 4):
        return None

    filler_id = None
    for cand in [" the", " and", " of", ".", ","]:
        tid = tokenizer.encode(cand, add_special_tokens=False)
        if len(tid) == 1 and tid[0] not in special:
            filler_id = tid[0]
            break
    if filler_id is None:
        filler_id = ids[0]

    # random pool
    random_ids = []
    for tid in range(tokenizer.vocab_size):
        if tid in special or tid in set(ids):
            continue
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s or len(s) > 12:
            continue
        if "\uFFFD" in s or "\x00" in s:
            continue
        random_ids.append(tid)
        if len(random_ids) >= 5000:
            break
    if len(random_ids) < 1000:
        random_ids = [t for t in range(tokenizer.vocab_size) if t not in special][:2000]

    return Pool(ids, id_to_str, filler_id, random_ids)


def build_alnum_pool(tokenizer, num_values: int, max_len: int = 4) -> Pool:
    special = set(tokenizer.all_special_ids)
    pat = re.compile(rf"^\s*[A-Za-z0-9]{{1,{max_len}}}$")
    ids: List[int] = []
    id_to_str: Dict[int, str] = {}

    for tid in range(tokenizer.vocab_size):
        if tid in special:
            continue
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s or not pat.match(s):
            continue
        # prefer leading space
        if len(s) >= 2 and s[0].isspace():
            ids.append(tid)
            id_to_str[tid] = s.strip()
        if len(ids) >= num_values:
            break

    if len(ids) < num_values:
        for tid in range(tokenizer.vocab_size):
            if tid in special or tid in set(ids):
                continue
            s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if not s or not pat.match(s):
                continue
            ids.append(tid)
            id_to_str[tid] = s.strip()
            if len(ids) >= num_values:
                break

    if len(ids) < max(16, num_values // 2):
        raise RuntimeError(f"Could not build value pool; found {len(ids)} tokens.")

    filler_id = None
    for cand in [" the", " and", " of", ".", ","]:
        tid = tokenizer.encode(cand, add_special_tokens=False)
        if len(tid) == 1 and tid[0] not in special:
            filler_id = tid[0]
            break
    if filler_id is None:
        filler_id = ids[0]

    random_ids = []
    for tid in range(tokenizer.vocab_size):
        if tid in special or tid in set(ids):
            continue
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s or len(s) > 12:
            continue
        random_ids.append(tid)
        if len(random_ids) >= 5000:
            break
    if len(random_ids) < 1000:
        random_ids = [t for t in range(tokenizer.vocab_size) if t not in special][:2000]

    return Pool(ids, id_to_str, filler_id, random_ids)


def build_pool(tokenizer, num_values: int, seed: int) -> Pool:
    # Try nice word pool first; fallback to alnum.
    p = build_word_pool(tokenizer, num_values=num_values, seed=seed)
    if p is not None:
        return p
    return build_alnum_pool(tokenizer, num_values=num_values)


# -------------------------
# Filler builders
# -------------------------

COHERENT_TEXT = (
    "In the early morning the city was quiet and the streets were empty. "
    "People walked to work and the sky turned pale blue above the buildings. "
    "A small cafe opened its doors and served coffee to a few tired customers. "
)

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
        # mostly filler, but insert decoy token many times (spaced)
        out = [pool.filler_id] * k
        if decoy_id is None:
            return out
        reps = min(decoy_reps, k)
        # place reps uniformly
        for i in range(reps):
            pos = int((i + 1) * (k / (reps + 1)))
            pos = max(0, min(k - 1, pos))
            out[pos] = int(decoy_id)
        # sprinkle some random tokens to avoid degeneracy
        for _ in range(min(8, k // 16)):
            out[rng.randrange(0, k)] = rng.choice(pool.random_ids)
        return out
    raise ValueError(f"Unknown filler_type {filler_type}")


# -------------------------
# Examples / Tasks
# -------------------------

@dataclass
class Example:
    input_ids: torch.Tensor      # [S]
    target_id: int               # correct next-token id
    candidate_ids: List[int]     # candidate set ids
    meta: Dict[str, str]         # for logging
    # Optional: "true competitor" id to highlight expected misbinding
    competitor_id: Optional[int] = None


class Task:
    name: str
    def __init__(self, tokenizer, pool: Pool, prompt_style: str):
        self.tok = tokenizer
        self.pool = pool
        self.style = prompt_style

    def build_one(self, k: int, filler_type: str, rng: random.Random, *,
                  decoy_reps: int, n_distractors: int) -> Example:
        raise NotImplementedError


class CompetingVars(Task):
    name = "competing_vars"
    def build_one(self, k: int, filler_type: str, rng: random.Random, *, decoy_reps: int, n_distractors: int) -> Example:
        tok, pool = self.tok, self.pool
        v1 = rng.choice(pool.token_ids)
        v2 = rng.choice([t for t in pool.token_ids if t != v1])

        # candidates include the two values + optional distractors
        cand = [int(v1), int(v2)]
        if n_distractors > 0:
            extra = [t for t in pool.token_ids if t not in cand]
            rng.shuffle(extra)
            cand += [int(x) for x in extra[:n_distractors]]
        rng.shuffle(cand)

        # filler blocks; allow decoy_heavy to bias toward KEY2's value
        fill1 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v2, decoy_reps=decoy_reps)
        fill2 = make_filler(tok, pool, filler_type, k, rng, decoy_id=v2, decoy_reps=decoy_reps)

        if self.style == "bracket":
            p1 = "KEY1=["
            close = " ]\n"
            p2 = "KEY2=["
            q = "\nQuestion: What is KEY1?\nKEY1=["
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p1, add_special_tokens=False)
            ids.append(int(v1))
            ids += tok.encode(close, add_special_tokens=False)
            ids += fill1
            ids += tok.encode(p2, add_special_tokens=False)
            ids.append(int(v2))
            ids += tok.encode(close, add_special_tokens=False)
            ids += fill2
            ids += tok.encode(q, add_special_tokens=False)
        else:
            p1 = "KEY1 ="
            p2 = "\nKEY2 ="
            q = "\nQuestion: What is KEY1?\nKEY1 ="
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p1, add_special_tokens=False)
            ids.append(int(v1))
            ids += fill1
            ids += tok.encode(p2, add_special_tokens=False)
            ids.append(int(v2))
            ids += fill2
            ids += tok.encode(q, add_special_tokens=False)

        return Example(
            input_ids=torch.tensor(ids, dtype=torch.long),
            target_id=int(v1),
            candidate_ids=cand,
            meta={"task": self.name, "k": str(k), "filler": filler_type},
            competitor_id=int(v2),
        )


class DecoyInjection(Task):
    name = "decoy_injection"
    def build_one(self, k: int, filler_type: str, rng: random.Random, *, decoy_reps: int, n_distractors: int) -> Example:
        tok, pool = self.tok, self.pool
        target = rng.choice(pool.token_ids)
        decoy = rng.choice([t for t in pool.token_ids if t != target])

        cand = [int(target), int(decoy)]
        if n_distractors > 0:
            extra = [t for t in pool.token_ids if t not in cand]
            rng.shuffle(extra)
            cand += [int(x) for x in extra[:n_distractors]]
        rng.shuffle(cand)

        fill = make_filler(tok, pool, filler_type, k, rng, decoy_id=decoy, decoy_reps=decoy_reps)

        if self.style == "bracket":
            p = "KEY=["
            close = " ]\n"
            q = "\nQuestion: What is KEY?\nKEY=["
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p, add_special_tokens=False)
            ids.append(int(target))
            ids += tok.encode(close, add_special_tokens=False)
            ids += fill
            ids += tok.encode(q, add_special_tokens=False)
        else:
            p = "KEY ="
            q = "\nQuestion: What is KEY?\nKEY ="
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p, add_special_tokens=False)
            ids.append(int(target))
            ids += fill
            ids += tok.encode(q, add_special_tokens=False)

        return Example(
            input_ids=torch.tensor(ids, dtype=torch.long),
            target_id=int(target),
            candidate_ids=cand,
            meta={"task": self.name, "k": str(k), "filler": filler_type},
            competitor_id=int(decoy),
        )


class PrimacyRecency(Task):
    name = "primacy_recency"
    def build_one(self, k: int, filler_type: str, rng: random.Random, *, decoy_reps: int, n_distractors: int) -> Example:
        tok, pool = self.tok, self.pool
        v1 = rng.choice(pool.token_ids)
        v2 = rng.choice([t for t in pool.token_ids if t != v1])
        v3 = rng.choice([t for t in pool.token_ids if t not in (v1, v2)])
        cand = [int(v1), int(v2), int(v3)]
        if n_distractors > 0:
            extra = [t for t in pool.token_ids if t not in cand]
            rng.shuffle(extra)
            cand += [int(x) for x in extra[:n_distractors]]
        rng.shuffle(cand)

        fill_mid = make_filler(tok, pool, filler_type, k, rng, decoy_id=v3, decoy_reps=decoy_reps)

        if self.style == "bracket":
            p = "KEY=["
            close = " ]\n"
            q = "\nQuestion: What was the FIRST value of KEY?\nKEY=["
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p, add_special_tokens=False); ids.append(int(v1)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill_mid
            ids += tok.encode(p, add_special_tokens=False); ids.append(int(v2)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill_mid
            ids += tok.encode(p, add_special_tokens=False); ids.append(int(v3)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill_mid
            ids += tok.encode(q, add_special_tokens=False)
        else:
            p = "KEY ="
            q = "\nQuestion: What was the FIRST value of KEY?\nKEY ="
            ids = []
            if tok.bos_token_id is not None:
                ids.append(tok.bos_token_id)
            ids += tok.encode(p, add_special_tokens=False); ids.append(int(v1))
            ids += fill_mid
            ids += tok.encode("\nKEY =", add_special_tokens=False); ids.append(int(v2))
            ids += fill_mid
            ids += tok.encode("\nKEY =", add_special_tokens=False); ids.append(int(v3))
            ids += fill_mid
            ids += tok.encode(q, add_special_tokens=False)

        return Example(
            input_ids=torch.tensor(ids, dtype=torch.long),
            target_id=int(v1),
            candidate_ids=cand,
            meta={"task": self.name, "k": str(k), "filler": filler_type},
            competitor_id=int(v3),  # recency competitor
        )


def count_r_in_repeated_word_steps(word: str, steps: int) -> int:
    """Count 'r' in the first `steps` characters of infinite repetition of `word`."""
    w = word
    n = len(w)
    c_full = w.count("r")
    full_cycles = steps // n
    rem = steps % n
    return full_cycles * c_full + w[:rem].count("r")


def build_numeric_token(tokenizer, n: int) -> Optional[int]:
    """
    Try to find a single token that decodes to the integer n (possibly with leading space/newline).
    """
    for form in [f" {n}", f"{n}", f"\n{n}"]:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            s = tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip()
            if s == str(n):
                return int(tid)
    return None


class StrawberryBaseline(Task):
    name = "strawberry_baseline"
    def __init__(self, tokenizer, pool: Pool, prompt_style: str, suffix_window: int = 20):
        super().__init__(tokenizer, pool, prompt_style)
        self.suffix_window = suffix_window

    def build_one(self, k: int, filler_type: str, rng: random.Random, *, decoy_reps: int, n_distractors: int) -> Example:
        """
        Stage-2B anchor: WORD vs TRACE_TOTAL vs SUFFIX.
        We provide a faithful trace (Stage 1 satisfied), but query asks WORD count only.
        Candidate set includes {WORD(=3), TRACE_TOTAL, SUFFIX_WINDOW}.
        """
        tok = self.tok
        word = "strawberry"
        word_count = word.count("r")  # 3
        trace_total = count_r_in_repeated_word_steps(word, k)
        suffix_steps = min(self.suffix_window, k)
        suffix_count = count_r_in_repeated_word_steps(word, suffix_steps)  # since repetition is periodic, suffix window aligns
        # If k is not aligned, suffix should be computed on last window of the trace; for repetition, counts can be computed by
        # difference of prefixes:
        if k >= suffix_steps:
            # count in last suffix_steps positions = count(prefix k) - count(prefix k - suffix_steps) but because pattern repeats, compute directly:
            suffix_count = count_r_in_repeated_word_steps(word, k) - count_r_in_repeated_word_steps(word, k - suffix_steps)

        counts = [word_count, trace_total, suffix_count]
        # Remove duplicates while preserving order
        uniq_counts = []
        for c in counts:
            if c not in uniq_counts:
                uniq_counts.append(c)

        # Build candidate tokens: prefer numeric single-tokens; else codebook tokens from pool
        cand_ids: List[int] = []
        count_to_tok: Dict[int, int] = {}
        numeric_ok = True
        for c in uniq_counts:
            tid = build_numeric_token(tok, int(c))
            if tid is None:
                numeric_ok = False
                break
            count_to_tok[int(c)] = tid
        if numeric_ok:
            for c in uniq_counts:
                cand_ids.append(count_to_tok[int(c)])
            target_id = count_to_tok[int(word_count)]
        else:
            # codebook fallback: map each count to a unique pool token
            avail = [t for t in self.pool.token_ids]
            rng.shuffle(avail)
            for c in uniq_counts:
                count_to_tok[int(c)] = int(avail.pop())
            cand_ids = [count_to_tok[int(c)] for c in uniq_counts]
            target_id = count_to_tok[int(word_count)]

        # Compose trace text: compact, 1 line, cycling tokens
        # Example:
        # TRACE: s:0 t:0 r:1 a:1 ... y:3 | s:3 ...
        # We'll emit as single string to tokenize.
        running = 0
        parts = []
        for i in range(k):
            ch = word[i % len(word)]
            if ch == "r":
                running += 1
            parts.append(f"{ch}:{running}")
        trace_str = " ".join(parts)

        # Include explicit competing statistics as tokens (so they are salient competitors)
        # If numeric tokens exist, we just include the integer text. If codebook, include mapping.
        if numeric_ok:
            header = (
                'We are counting the letter "r" in the word "strawberry".\n'
                "The TRACE below cycles through characters for many steps and updates a running TRACE count.\n"
                "TRACE:\n"
            )
            summary = (
                f"\nTRACE_TOTAL={trace_total}\n"
                f"SUFFIX_{suffix_steps}={suffix_count}\n"
            )
            question = (
                "\nQuestion: How many r's are in the 10-letter word strawberry (WORD count only, not TRACE_TOTAL or SUFFIX)?\n"
                "Answer with a single token number:\n"
                "ANSWER:"
            )
        else:
            # Codebook: reveal mapping so model can output the code token; still tests binding between competing stats.
            # We provide mapping explicitly so Stage 1 is perfect and the only question is which statistic is selected.
            tok_word = safe_decode(tok, count_to_tok[word_count]).strip()
            tok_total = safe_decode(tok, count_to_tok[trace_total]).strip()
            tok_suf = safe_decode(tok, count_to_tok[suffix_count]).strip()
            header = (
                'We are counting the letter "r" in the word "strawberry".\n'
                "The TRACE below cycles through characters and updates a running TRACE count.\n"
                "TRACE:\n"
            )
            summary = (
                f"\nTRACE_TOTAL_CODE={tok_total}\n"
                f"SUFFIX_{suffix_steps}_CODE={tok_suf}\n"
                f"WORD_CODE={tok_word}\n"
            )
            question = (
                "\nQuestion: Return the WORD_CODE (the number of r's in the word strawberry only). "
                "Do NOT output TRACE_TOTAL_CODE or SUFFIX_CODE.\n"
                "ANSWER_CODE="
            )

        # Use fillers only to match other tasks; in strawberry the 'k' controls trace length so we keep filler small/neutral.
        # But we allow small filler to test robustness to extra context.
        filler = make_filler(tok, self.pool, filler_type if filler_type != "decoy_heavy" else "repeat", min(32, k // 4), rng)

        prompt = header + trace_str + summary
        ids = []
        if tok.bos_token_id is not None:
            ids.append(tok.bos_token_id)
        ids += tok.encode(prompt, add_special_tokens=False)
        ids += filler
        ids += tok.encode(question, add_special_tokens=False)

        # Candidate set = cand_ids (plus optional distractors from numeric? no)
        return Example(
            input_ids=torch.tensor(ids, dtype=torch.long),
            target_id=int(target_id),
            candidate_ids=[int(x) for x in cand_ids],
            meta={"task": self.name, "k": str(k), "filler": filler_type, "numeric_ok": str(int(numeric_ok)), "suffix": str(suffix_steps)},
            competitor_id=int(count_to_tok[int(trace_total)]) if int(trace_total) in count_to_tok else None,
        )


class StrawberryFactorized(StrawberryBaseline):
    name = "strawberry_factorized"
    def build_one(self, k: int, filler_type: str, rng: random.Random, *, decoy_reps: int, n_distractors: int) -> Example:
        """
        Factorized variant: same trace, but adds explicit variable factorization and a readout schema
        intended to reduce misbinding (binding intervention at the prompt level).
        """
        ex = super().build_one(k, filler_type, rng, decoy_reps=decoy_reps, n_distractors=n_distractors)
        tok = self.tok

        # Append a short factorization note that should reduce Stage 2B misbinding.
        factor_note = (
            "\nBinding instruction:\n"
            "- WORD statistic: count of 'r' in the 10-letter base word only.\n"
            "- TRACE_TOTAL statistic: count of 'r' across the full TRACE.\n"
            "- SUFFIX statistic: count of 'r' in the last window only.\n"
            "Return the WORD statistic ONLY.\n"
        )
        extra_ids = tok.encode(factor_note, add_special_tokens=False)
        ex.input_ids = torch.cat([ex.input_ids, torch.tensor(extra_ids, dtype=torch.long)], dim=0)
        ex.meta = dict(ex.meta)
        ex.meta["task"] = self.name
        return ex


def make_task(task_name: str, tokenizer, pool: Pool, prompt_style: str) -> Task:
    task_name = task_name.lower()
    if task_name == "competing_vars":
        return CompetingVars(tokenizer, pool, prompt_style)
    if task_name == "decoy_injection":
        return DecoyInjection(tokenizer, pool, prompt_style)
    if task_name == "primacy_recency":
        return PrimacyRecency(tokenizer, pool, prompt_style)
    if task_name == "strawberry_baseline":
        return StrawberryBaseline(tokenizer, pool, prompt_style)
    if task_name == "strawberry_factorized":
        return StrawberryFactorized(tokenizer, pool, prompt_style)
    raise ValueError(f"Unknown task: {task_name}")


# -------------------------
# Model loading
# -------------------------

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, attn_impl: str, load_in_8bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True)
    if attn_impl and attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tok


# -------------------------
# Evaluation
# -------------------------

@dataclass
class CondStats:
    n: int = 0
    n_correct: int = 0
    n_cand_correct: int = 0
    n_stage2a: int = 0
    n_stage2b: int = 0
    sum_gate_gap: float = 0.0
    sum_value_gap: float = 0.0
    wrong_non_cand: Dict[int, int] = dataclasses.field(default_factory=dict)
    wrong_cand: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Inline probe stats for mechanistic analysis
    probe_n_total: int = 0
    probe_n_correct_overall: int = 0
    probe_n_stage2b_trials: int = 0  # Stage 2B trials where we evaluated probe
    probe_n_correct_on_stage2b: int = 0  # Probe correctly predicts target on Stage 2B errors
    probe_n_correct_trials: int = 0  # Correct trials where we evaluated probe
    probe_n_correct_on_correct: int = 0  # Probe correct on correct trials (sanity)

def collate(exs: List[Example], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    lengths = {int(e.input_ids.numel()) for e in exs}
    if len(lengths) != 1:
        # For safety, do per-example evaluation if variable length sneaks in (shouldn't for our tasks).
        raise ValueError(f"Variable lengths in batch: {sorted(lengths)}")
    input_ids = torch.stack([e.input_ids for e in exs], dim=0).to(device)
    attn_mask = torch.ones_like(input_ids)
    targets = torch.tensor([e.target_id for e in exs], dtype=torch.long, device=device)
    cands = [e.candidate_ids for e in exs]
    return input_ids, attn_mask, targets, cands

@torch.no_grad()
def forward_last_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_attentions=False)
    return out.logits[:, -1, :]


@torch.no_grad()
def forward_with_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int = -1):
    """
    Returns (logits_last, hidden_last) where hidden_last is from specified layer at last position.
    layer_idx=-1 means final layer before unembedding.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
    )
    logits = out.logits[:, -1, :]  # [B, V]
    
    # hidden_states is tuple of (n_layers+1,) each [B, S, D]
    # Index 0 is embedding, 1..n_layers are transformer blocks
    hidden_states = out.hidden_states
    n_layers = len(hidden_states) - 1
    
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx + 1  # -1 -> final layer
    
    hidden = hidden_states[layer_idx][:, -1, :]  # [B, D]
    return logits, hidden


class InlineLinearProbe:
    """Simple linear probe for multi-class classification of target values."""
    
    def __init__(self, d_model: int, n_classes: int, class_ids: List[int]):
        self.d_model = d_model
        self.n_classes = n_classes
        self.class_ids = class_ids  # Maps class index -> token_id
        self.id_to_class = {tid: i for i, tid in enumerate(class_ids)}
        self.W = None  # [n_classes, d_model]
        self.b = None  # [n_classes]
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, reg: float = 1e-4):
        """
        X: [N, D] hidden states
        y: [N] class indices (not token ids)
        """
        N, D = X.shape
        W = np.random.randn(self.n_classes, D).astype(np.float32) * 0.01
        b = np.zeros(self.n_classes, dtype=np.float32)
        
        for _ in range(epochs):
            # Forward
            logits = X @ W.T + b  # [N, C]
            probs = self._softmax(logits)
            
            # Gradient (cross-entropy)
            grad = probs.copy()
            grad[np.arange(N), y] -= 1
            grad /= N
            
            dW = grad.T @ X + reg * W
            db = grad.sum(axis=0)
            
            W -= lr * dW
            b -= lr * db
        
        self.W = W
        self.b = b
    
    def _softmax(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns class indices."""
        logits = X @ self.W.T + self.b
        return logits.argmax(axis=-1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, C] probabilities."""
        logits = X @ self.W.T + self.b
        return self._softmax(logits)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())

def update_hist(d: Dict[int, int], key: int, inc: int = 1):
    d[int(key)] = int(d.get(int(key), 0)) + int(inc)

@torch.no_grad()
def eval_condition(
    model,
    tokenizer,
    task: Task,
    k: int,
    filler_type: str,
    trials: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    decoy_reps: int,
    n_distractors: int,
    cleanup_every_batches: int = 0,
    enable_probe: bool = False,
    probe_layer: int = -1,
    probe_train_frac: float = 0.5,
) -> CondStats:
    """
    Evaluate one (task, k, filler) condition with optional probe-on-wrong analysis.

    If enable_probe=True, trains a linear probe on hidden states to predict targets,
    then measures probe accuracy specifically on Stage 2B errors to answer:
    "When the model outputs a wrong candidate, is the correct target still decodable?"
    """
    rng = random.Random(seed)
    stats = CondStats()

    # For probe: collect hidden states, targets, and error types
    all_hiddens = []
    all_targets = []
    all_is_stage2b = []
    all_is_correct = []

    for batch_idx, start in enumerate(range(0, trials, batch_size)):
        bs = min(batch_size, trials - start)

        batch: List[Example] = [
            task.build_one(k, filler_type, rng, decoy_reps=decoy_reps, n_distractors=n_distractors)
            for _ in range(bs)
        ]
        Ls = {int(e.input_ids.numel()) for e in batch}
        if len(Ls) != 1:
            raise RuntimeError(
                f"Task {task.name} produced variable lengths within a batch: {sorted(Ls)}. "
                f"Consider padding + gathering last-nonpad logits if you extend tasks to variable-length prompts."
            )

        input_ids, attn_mask, targets, cands = collate(batch, device)

        # Use forward_with_hidden if probing, else use simpler forward
        if enable_probe:
            logits, hiddens = forward_with_hidden(model, input_ids, attn_mask, layer_idx=probe_layer)
        else:
            logits = forward_last_logits(model, input_ids, attn_mask)
            hiddens = None

        preds = logits.argmax(dim=-1)
        B = preds.shape[0]

        # --- Vectorized candidate-only evaluation ---
        Cmax = max(len(c) for c in cands)
        cand_ids_cpu = torch.zeros((B, Cmax), dtype=torch.long)
        cand_mask_cpu = torch.zeros((B, Cmax), dtype=torch.bool)
        for bi, cand in enumerate(cands):
            cand_ids_cpu[bi, : len(cand)] = torch.tensor(cand, dtype=torch.long)
            cand_mask_cpu[bi, : len(cand)] = True

        cand_ids = cand_ids_cpu.to(device, non_blocking=True)
        cand_mask = cand_mask_cpu.to(device, non_blocking=True)

        cand_logits = logits.gather(1, cand_ids)
        cand_logits_masked = cand_logits.masked_fill(~cand_mask, float("-inf"))

        cand_best_vals, cand_best_idx = cand_logits_masked.max(dim=-1)
        cand_pred = cand_ids.gather(1, cand_best_idx.unsqueeze(1)).squeeze(1)

        correct = preds.eq(targets)
        cand_correct = cand_pred.eq(targets)

        # value gap: logit(true) - max(wrong candidate)
        true_log = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        mask_wrong = cand_mask & cand_ids.ne(targets.unsqueeze(1))
        wrong_vals = cand_logits.masked_fill(~mask_wrong, float("-inf"))
        max_wrong = wrong_vals.max(dim=-1).values
        has_wrong = mask_wrong.any(dim=-1)
        value_gap = torch.where(has_wrong, true_log - max_wrong, torch.full_like(true_log, float("nan")))

        # gate gap: max(candidate) - max(non-candidate)
        K = int(min(logits.shape[-1], Cmax + 16))
        topk_vals, topk_ids = logits.topk(K, dim=-1)

        in_cand = ((topk_ids.unsqueeze(-1) == cand_ids.unsqueeze(1)) & cand_mask.unsqueeze(1)).any(dim=-1)
        non_cand_mask = ~in_cand
        has_non = non_cand_mask.any(dim=-1)
        first_non_idx = non_cand_mask.float().argmax(dim=-1)
        max_non = topk_vals.gather(1, first_non_idx.unsqueeze(1)).squeeze(1)

        if not bool(has_non.all().item()):
            bad_rows = (has_non == 0).nonzero(as_tuple=False).view(-1).tolist()
            for bi in bad_rows:
                cand_row = cand_ids[bi, cand_mask[bi]]
                masked = logits[bi].clone()
                masked.index_fill_(0, cand_row, float("-inf"))
                max_non[bi] = masked.max()

        gate_gap = cand_best_vals - max_non

        # Accumulate stats
        stats.n += B
        stats.n_correct += int(correct.sum().item())
        stats.n_cand_correct += int(cand_correct.sum().item())
        stats.sum_gate_gap += float(torch.nan_to_num(gate_gap, nan=0.0).sum().item())
        stats.sum_value_gap += float(torch.nan_to_num(value_gap, nan=0.0).sum().item())

        # Stage2A vs Stage2B error accounting
        errors = ~correct
        pred_in_cand = ((preds.unsqueeze(1) == cand_ids) & cand_mask).any(dim=1)
        stage2b = errors & pred_in_cand
        stage2a = errors & (~pred_in_cand)

        stats.n_stage2b += int(stage2b.sum().item())
        stats.n_stage2a += int(stage2a.sum().item())

        # Update histograms
        for tid in preds[stage2a].detach().cpu().tolist():
            update_hist(stats.wrong_non_cand, int(tid), 1)
        for tid in preds[stage2b].detach().cpu().tolist():
            update_hist(stats.wrong_cand, int(tid), 1)

        # Store for probe analysis
        if enable_probe and hiddens is not None:
            all_hiddens.append(hiddens.cpu().float().numpy())
            all_targets.append(targets.cpu().numpy())
            all_is_stage2b.append(stage2b.cpu().numpy())
            all_is_correct.append(correct.cpu().numpy())

        # Cleanup
        del batch, input_ids, attn_mask, targets, cands
        del logits, preds, cand_ids_cpu, cand_mask_cpu, cand_ids, cand_mask
        del cand_logits, cand_logits_masked, cand_best_vals, cand_best_idx, cand_pred
        del true_log, mask_wrong, wrong_vals, max_wrong, has_wrong, value_gap
        del topk_vals, topk_ids, in_cand, non_cand_mask, has_non, first_non_idx, max_non, gate_gap
        del correct, cand_correct, errors, pred_in_cand, stage2a, stage2b
        if hiddens is not None:
            del hiddens

        if cleanup_every_batches and ((batch_idx + 1) % cleanup_every_batches == 0):
            gc.collect()
            try_empty_cache()

    # Probe analysis
    if enable_probe and len(all_hiddens) > 0:
        X = np.concatenate(all_hiddens, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        is_stage2b = np.concatenate(all_is_stage2b, axis=0).astype(bool)
        is_correct = np.concatenate(all_is_correct, axis=0).astype(bool)

        # Build class set from unique targets
        unique_targets = sorted(set(targets_np.tolist()))
        if len(unique_targets) >= 2:
            probe = InlineLinearProbe(X.shape[1], len(unique_targets), unique_targets)

            # Convert targets to class indices
            y = np.array([probe.id_to_class[t] for t in targets_np])

            # Train/test split
            N = len(X)
            n_train = int(N * probe_train_frac)
            idx = np.random.RandomState(seed).permutation(N)
            train_idx, test_idx = idx[:n_train], idx[n_train:]

            if len(train_idx) >= 10 and len(test_idx) >= 10:
                probe.fit(X[train_idx], y[train_idx], epochs=100, lr=0.1)

                # Evaluate on test set
                test_preds = probe.predict(X[test_idx])
                test_correct = (test_preds == y[test_idx])

                stats.probe_n_total = len(test_idx)
                stats.probe_n_correct_overall = int(test_correct.sum())

                # Probe on Stage 2B errors (test set only)
                test_is_stage2b = is_stage2b[test_idx]
                stage2b_mask = test_is_stage2b
                if stage2b_mask.sum() > 0:
                    stats.probe_n_stage2b_trials = int(stage2b_mask.sum())
                    stats.probe_n_correct_on_stage2b = int(test_correct[stage2b_mask].sum())

                # Probe on correct trials (sanity check)
                test_is_correct = is_correct[test_idx]
                if test_is_correct.sum() > 0:
                    stats.probe_n_correct_trials = int(test_is_correct.sum())
                    stats.probe_n_correct_on_correct = int(test_correct[test_is_correct].sum())

    return stats

def merge_stats(stats_list: List[CondStats]) -> CondStats:
    out = CondStats()
    for s in stats_list:
        out.n += s.n
        out.n_correct += s.n_correct
        out.n_cand_correct += s.n_cand_correct
        out.n_stage2a += s.n_stage2a
        out.n_stage2b += s.n_stage2b
        out.sum_gate_gap += s.sum_gate_gap
        out.sum_value_gap += s.sum_value_gap
        out.probe_n_total += s.probe_n_total
        out.probe_n_correct_overall += s.probe_n_correct_overall
        out.probe_n_stage2b_trials += s.probe_n_stage2b_trials
        out.probe_n_correct_on_stage2b += s.probe_n_correct_on_stage2b
        out.probe_n_correct_trials += s.probe_n_correct_trials
        out.probe_n_correct_on_correct += s.probe_n_correct_on_correct
        for k, v in s.wrong_non_cand.items():
            out.wrong_non_cand[k] = out.wrong_non_cand.get(k, 0) + v
        for k, v in s.wrong_cand.items():
            out.wrong_cand[k] = out.wrong_cand.get(k, 0) + v
    return out


# -------------------------
# Probing (optional)
# -------------------------

def get_decoder_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Return the list of transformer block modules for common HF decoder-only LMs.
    Supports (tested in this codebase): Qwen2, Llama, Gemma2, Phi-3 style.
    """
    m = model
    if hasattr(m, "module"):
        m = m.module
    base = getattr(m, "model", m)

    # Common patterns
    if hasattr(base, "layers") and isinstance(base.layers, (list, torch.nn.ModuleList)):
        return list(base.layers)
    if hasattr(base, "h") and isinstance(base.h, (list, torch.nn.ModuleList)):
        return list(base.h)
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return list(base.decoder.layers)

    raise RuntimeError("Could not locate decoder layers for probing. Known attributes: model.layers, model.h, model.decoder.layers")


def parse_probe_layers(spec: str, n_layers: int) -> List[int]:
    """
    spec: comma-separated list of ints or keywords:
      - "first" -> 0
      - "mid" -> n_layers//2
      - "last" -> n_layers-1
    If spec is empty, defaults to first, mid, last.
    """
    if spec is None:
        spec = ""
    spec = str(spec).strip()
    if spec == "":
        return [0, n_layers // 2, n_layers - 1]
    out: List[int] = []
    for part in spec.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p == "first":
            out.append(0)
        elif p in ("mid", "middle"):
            out.append(n_layers // 2)
        elif p == "last":
            out.append(n_layers - 1)
        else:
            out.append(int(p))
    # unique + sorted
    out = sorted(set([i for i in out if 0 <= i < n_layers]))
    if not out:
        return [0, n_layers // 2, n_layers - 1]
    return out


@torch.no_grad()
def forward_last_logits_and_layer_last_h(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Returns:
      logits_last: [B, V]
      feats: dict[layer_idx] -> [B, D] (float32 on CPU)
    Uses forward hooks but CLONES the last-token slice to avoid holding onto full [B,S,D] tensors.
    """
    layers = get_decoder_layers(model)
    feats: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(li: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(hs):
                return
            # clone the last token so we don't retain the full activation tensor
            last = hs[:, -1, :].detach().clone()
            feats[li] = last.to(dtype=torch.float32, device="cpu")
        return hook

    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(make_hook(li)))

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_attentions=False)
    logits_last = out.logits[:, -1, :].detach()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # Ensure all requested layers present
    for li in layer_indices:
        if li not in feats:
            raise RuntimeError(f"Probe hook did not capture layer {li}.")
    return logits_last, feats


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
) -> Tuple[float, np.ndarray]:
    """
    Train a linear probe and return:
      - test accuracy (float)
      - predicted classes on test set (np.int64)
    """
    rng = np.random.default_rng(seed)
    d_in = int(X_train.shape[1])
    model = LinearProbe(d_in=d_in, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    n = Xt.shape[0]
    if n == 0:
        return float("nan"), np.zeros((0,), dtype=np.int64)

    for ep in range(int(epochs)):
        # shuffle
        perm = rng.permutation(n)
        Xt_ep = Xt[perm]
        yt_ep = yt[perm]
        for i in range(0, n, int(batch_size)):
            xb = Xt_ep[i:i+batch_size]
            yb = yt_ep[i:i+batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # eval
    Xv = torch.tensor(X_test, dtype=torch.float32, device=device)
    yv = torch.tensor(y_test, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(Xv).argmax(dim=-1)
        acc = (pred == yv).float().mean().item() if yv.numel() > 0 else float("nan")
    pred_cpu = pred.detach().to("cpu").numpy().astype(np.int64)

    # cleanup
    del Xt, yt, Xv, yv, pred
    del model
    gc.collect()
    try_empty_cache()
    return float(acc), pred_cpu


@torch.no_grad()
def collect_probe_dataset(
    model,
    tokenizer,
    task: Task,
    pool: Pool,
    *,
    k: int,
    filler_type: str,
    n_examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    layer_indices: List[int],
    decoy_reps: int,
    n_distractors: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build probe dataset for tasks whose target_id is in pool.token_ids.

    Returns:
      X_by_layer: dict[layer_idx] -> [N, D] float32
      y: [N] int64 class labels (index in pool.token_ids)
      is_wrong: [N] bool (model wrong on this example)
      is_stage2b_wrong: [N] bool (model wrong and pred in candidate set)
    """
    # mapping from token_id -> class index
    id_to_cls = {int(tid): i for i, tid in enumerate(pool.token_ids)}
    X_lists: Dict[int, List[np.ndarray]] = {li: [] for li in layer_indices}
    y_list: List[int] = []
    wrong_list: List[bool] = []
    stage2b_list: List[bool] = []

    rng = random.Random(seed)
    kept = 0

    pbar = tqdm(total=n_examples, desc=f"Probe data ({task.name},k={k},{filler_type})", leave=False)
    while kept < n_examples:
        bs = min(batch_size, n_examples - kept)
        batch = [task.build_one(k, filler_type, rng, decoy_reps=decoy_reps, n_distractors=n_distractors) for _ in range(bs)]
        input_ids, attn_mask, targets, cands = collate(batch, device)
        logits_last, feats = forward_last_logits_and_layer_last_h(model, input_ids, attn_mask, layer_indices)
        preds = logits_last.argmax(dim=-1).detach().to("cpu").tolist()

        # per-example bookkeeping
        for bi in range(bs):
            tgt = int(targets[bi].item())
            if tgt not in id_to_cls:
                continue  # unsupported label space
            cls = int(id_to_cls[tgt])
            y_list.append(cls)

            pred = int(preds[bi])
            cand_set = set(int(x) for x in batch[bi].candidate_ids)
            is_wrong = (pred != tgt)
            is_stage2b_wrong = is_wrong and (pred in cand_set)

            wrong_list.append(bool(is_wrong))
            stage2b_list.append(bool(is_stage2b_wrong))

            for li in layer_indices:
                X_lists[li].append(feats[li][bi].numpy())

            kept += 1
            pbar.update(1)
            if kept >= n_examples:
                break

        # cleanup
        del batch, input_ids, attn_mask, targets, cands, logits_last, feats
        try:
            del preds
        except Exception:
            pass

    pbar.close()

    # stack
    X_by_layer = {li: np.stack(X_lists[li], axis=0).astype(np.float32) for li in layer_indices}
    y = np.array(y_list, dtype=np.int64)
    is_wrong = np.array(wrong_list, dtype=np.bool_)
    is_stage2b_wrong = np.array(stage2b_list, dtype=np.bool_)
    return X_by_layer, y, is_wrong, is_stage2b_wrong


def run_probes_for_condition(
    model,
    tokenizer,
    task: Task,
    pool: Pool,
    *,
    k: int,
    filler_type: str,
    seed: int,
    device: torch.device,
    layer_indices: List[int],
    n_train: int,
    n_test: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    train_bs: int,
    decoy_reps: int,
    n_distractors: int,
) -> Dict[str, object]:
    """
    Train + evaluate probes for a single (task,k,filler) condition.
    Returns a JSON-serializable dict of metrics.
    """
    # Only meaningful when targets come from pool.token_ids
    supported = task.name in ("competing_vars", "decoy_injection", "primacy_recency")
    if not supported:
        return {"probe_ran": False, "probe_skip_reason": f"task {task.name} not supported for pool-label probe"}

    # collect
    Xtr_by, ytr, _, _ = collect_probe_dataset(
        model, tokenizer, task, pool,
        k=k, filler_type=filler_type, n_examples=int(n_train),
        batch_size=int(batch_size), seed=seed + 11, device=device,
        layer_indices=layer_indices, decoy_reps=decoy_reps, n_distractors=n_distractors,
    )
    Xte_by, yte, wrong_te, stage2b_wrong_te = collect_probe_dataset(
        model, tokenizer, task, pool,
        k=k, filler_type=filler_type, n_examples=int(n_test),
        batch_size=int(batch_size), seed=seed + 29, device=device,
        layer_indices=layer_indices, decoy_reps=decoy_reps, n_distractors=n_distractors,
    )

    n_classes = int(len(pool.token_ids))
    out: Dict[str, object] = {
        "probe_ran": True,
        "probe_layers": json.dumps([int(li) for li in layer_indices]),
        "probe_n_train": int(n_train),
        "probe_n_test": int(n_test),
        "probe_n_test_wrong": int(wrong_te.sum()),
        "probe_n_test_stage2b_wrong": int(stage2b_wrong_te.sum()),
    }

    acc_by: Dict[str, float] = {}
    acc_wrong_by: Dict[str, float] = {}
    acc_s2b_by: Dict[str, float] = {}

    for li in layer_indices:
        acc, pred_cls = train_linear_probe(
            Xtr_by[li], ytr, Xte_by[li], yte,
            n_classes=n_classes, seed=seed + 1000 + int(li),
            device=device, lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=train_bs,
        )
        acc_by[str(li)] = float(acc)

        # conditional accuracies
        if yte.size == 0:
            acc_wrong_by[str(li)] = float("nan")
            acc_s2b_by[str(li)] = float("nan")
        else:
            pred_ok = (pred_cls == yte)
            if wrong_te.any():
                acc_wrong_by[str(li)] = float(pred_ok[wrong_te].mean())
            else:
                acc_wrong_by[str(li)] = float("nan")
            if stage2b_wrong_te.any():
                acc_s2b_by[str(li)] = float(pred_ok[stage2b_wrong_te].mean())
            else:
                acc_s2b_by[str(li)] = float("nan")

        # cleanup per layer
        del pred_cls
        gc.collect()
        try_empty_cache()

    out["probe_acc_by_layer"] = json.dumps(acc_by)
    out["probe_acc_on_model_wrong_by_layer"] = json.dumps(acc_wrong_by)
    out["probe_acc_on_stage2b_wrong_by_layer"] = json.dumps(acc_s2b_by)

    # cleanup
    del Xtr_by, ytr, Xte_by, yte, wrong_te, stage2b_wrong_te
    gc.collect()
    try_empty_cache()
    return out




def topk_hist(d: Dict[int, int], k: int = 10) -> List[Tuple[int, int]]:
    return sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]


# -------------------------
# Model sharding by process groups
# -------------------------

@dataclass
class GroupInfo:
    group: Optional[dist.ProcessGroup]
    group_id: int
    group_rank: int
    group_world: int
    group_leader_global_rank: int


def make_model_groups(models: List[str], shard_by_model: bool, rank: int, world: int) -> Tuple[GroupInfo, str]:
    """
    If shard_by_model:
      - world_size must be divisible by len(models)
      - create groups; each group handles one model concurrently
    Else:
      - single group handles all models sequentially
    """
    if not is_dist() or not shard_by_model or len(models) == 1:
        return GroupInfo(group=None, group_id=0, group_rank=rank, group_world=world, group_leader_global_rank=0), models[0]

    m = len(models)
    if world % m != 0:
        raise RuntimeError(f"--shard_by_model requires WORLD_SIZE divisible by #models. world={world}, models={m}")
    group_world = world // m
    group_id = rank // group_world
    group_rank = rank % group_world
    leader = group_id * group_world
    ranks = list(range(group_id * group_world, (group_id + 1) * group_world))
    group = dist.new_group(ranks=ranks)
    model_name = models[group_id]
    return GroupInfo(group=group, group_id=group_id, group_rank=group_rank, group_world=group_world, group_leader_global_rank=leader), model_name


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--shard_by_model", action="store_true", help="Split ranks into groups; each group runs one model concurrently.")

    ap.add_argument("--tasks", nargs="+", default=["competing_vars", "decoy_injection", "primacy_recency", "strawberry_baseline"])
    ap.add_argument("--prompt_style", type=str, default="bracket", choices=["bracket", "eq"])
    ap.add_argument("--k_values", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--filler_types", nargs="+", type=str, default=["repeat", "decoy_heavy"])
    ap.add_argument("--trials_per_condition", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--cleanup_every_batches", type=int, default=0,
                    help="If >0, run gc.collect()+torch.cuda.empty_cache() every N batches inside eval_condition.")
    ap.add_argument("--cleanup_between_conditions", action="store_true",
                    help="Run gc.collect()+torch.cuda.empty_cache() after each (task,k,filler) condition.")
    ap.add_argument("--decoy_reps", type=int, default=12)
    ap.add_argument("--n_distractors", type=int, default=0)
    ap.add_argument("--num_values", type=int, default=128)

    # Probes (optional): decode which pool value should be output from intermediate activations.
    ap.add_argument("--do_probes", action="store_true", help="Run linear probes on selected conditions.")
    ap.add_argument("--probe_tasks", nargs="+", default=["competing_vars", "primacy_recency"],
                    help="Which tasks to run probes for (default: competing_vars primacy_recency).")
    ap.add_argument("--probe_k_values", nargs="+", type=int, default=[256],
                    help="k values to run probes at (default: 256).")
    ap.add_argument("--probe_filler_types", nargs="+", type=str, default=["decoy_heavy"],
                    help="filler types to run probes under (default: decoy_heavy).")
    ap.add_argument("--probe_layers", type=str, default="",
                    help="Comma-separated layer indices or keywords first,mid,last (default: first,mid,last).")
    ap.add_argument("--probe_n_train", type=int, default=1500)
    ap.add_argument("--probe_n_test", type=int, default=500)
    ap.add_argument("--probe_batch_size", type=int, default=4, help="Batch size for probe feature extraction.")
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-3)
    ap.add_argument("--probe_epochs", type=int, default=30)
    ap.add_argument("--probe_train_bs", type=int, default=256, help="Batch size for probe training.")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2", "auto"])
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # Inline probe-on-wrong analysis (optional)
    ap.add_argument("--enable_probe", action="store_true",
                    help="Enable inline probe analysis to measure decodability on Stage 2B errors")
    ap.add_argument("--probe_layer", type=int, default=-1,
                    help="Layer index for inline probe hidden states (-1 = final layer)")
    ap.add_argument("--probe_train_frac", type=float, default=0.5,
                    help="Fraction of data for inline probe training (rest for test)")

    ap.add_argument("--outdir", type=str, default="stage2b_out")
    ap.add_argument("--out_csv", type=str, default="stage2b_summary.csv")
    ap.add_argument("--out_json", type=str, default="stage2b_summary.json")

    args = ap.parse_args()

    rank, world, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed + rank * 10000)

    ensure_dir(args.outdir)

    dtype = torch_dtype_from_str(args.dtype)

    # Setup model group
    group_info, my_model = make_model_groups(args.models, args.shard_by_model, rank, world)
    group = group_info.group
    group_rank = group_info.group_rank
    group_world = group_info.group_world
    is_group_leader = (group_rank == 0)
    is_global_leader = (rank == 0)

    # Load model for this group
    if is_group_leader:
        print(f"\n[group {group_info.group_id}] Loading model: {my_model} (ranks: {group_world})")
    ddp_barrier()

    model, tok = load_model_and_tokenizer(my_model, dtype=dtype, attn_impl=args.attn_impl, load_in_8bit=args.load_in_8bit)
    model.to(device)
    pool = build_pool(tok, num_values=args.num_values, seed=args.seed + group_info.group_id)

    # Build tasks
    tasks = [make_task(t, tok, pool, args.prompt_style) for t in args.tasks]

    # Each rank evaluates a shard of trials for each condition
    trials_local = int(math.ceil(args.trials_per_condition / group_world))

    rows_local: List[dict] = []
    # Evaluate
    for task in tasks:
        for filler_type in args.filler_types:
            for k in args.k_values:
                # Deterministic seed per condition + rank-in-group
                cond_seed = (args.seed + 1000 * hash((my_model, task.name, filler_type, k)) % 1_000_000_000 + group_rank)
                stats = eval_condition(
                    model, tok, task, k, filler_type,
                    trials=trials_local,
                    batch_size=args.batch_size,
                    seed=cond_seed,
                    device=device,
                    decoy_reps=args.decoy_reps,
                    n_distractors=args.n_distractors,
                    cleanup_every_batches=args.cleanup_every_batches,
                    enable_probe=args.enable_probe,
                    probe_layer=args.probe_layer,
                    probe_train_frac=args.probe_train_frac,
                )

                if args.cleanup_between_conditions:
                    gc.collect()
                    try_empty_cache()

                # Reduce counts within group
                t = torch.tensor([
                    stats.n, stats.n_correct, stats.n_cand_correct, stats.n_stage2a, stats.n_stage2b,
                    stats.sum_gate_gap, stats.sum_value_gap,
                    stats.probe_n_total, stats.probe_n_correct_overall,
                    stats.probe_n_stage2b_trials, stats.probe_n_correct_on_stage2b,
                    stats.probe_n_correct_trials, stats.probe_n_correct_on_correct,
                ], device=device, dtype=torch.float64)
                all_reduce_tensor(t, group=group, op=dist.ReduceOp.SUM if is_dist() else None)

                # Gather hist dicts to group leader and merge
                h_non = all_gather_object(stats.wrong_non_cand, group=group)
                h_cand = all_gather_object(stats.wrong_cand, group=group)

                if is_group_leader:
                    merged_non: Dict[int, int] = {}
                    merged_cand: Dict[int, int] = {}
                    for d in h_non:
                        for kk, vv in d.items():
                            merged_non[int(kk)] = merged_non.get(int(kk), 0) + int(vv)
                    for d in h_cand:
                        for kk, vv in d.items():
                            merged_cand[int(kk)] = merged_cand.get(int(kk), 0) + int(vv)

                    n = int(t[0].item())
                    n_correct = int(t[1].item())
                    n_cand_correct = int(t[2].item())
                    n_s2a = int(t[3].item())
                    n_s2b = int(t[4].item())
                    sum_gate = float(t[5].item())
                    sum_val = float(t[6].item())

                    probe_n_total = int(t[7].item())
                    probe_n_correct_overall = int(t[8].item())
                    probe_n_stage2b_trials = int(t[9].item())
                    probe_n_correct_on_stage2b = int(t[10].item())
                    probe_n_correct_trials = int(t[11].item())
                    probe_n_correct_on_correct = int(t[12].item())

                    acc = n_correct / max(1, n)
                    acc_cand = n_cand_correct / max(1, n)

                    # Mean gaps; note: value gap may be nan if ill-defined; we still report mean over n
                    mean_gate_gap = sum_gate / max(1, n)
                    mean_value_gap = sum_val / max(1, n)

                    probe_acc_overall = probe_n_correct_overall / max(1, probe_n_total)
                    probe_acc_on_stage2b = probe_n_correct_on_stage2b / max(1, probe_n_stage2b_trials)
                    probe_acc_on_correct = probe_n_correct_on_correct / max(1, probe_n_correct_trials)

                    top_non = topk_hist(merged_non, k=10)
                    top_cand = topk_hist(merged_cand, k=10)
                    top_non_dec = [(safe_decode(tok, tid), int(cnt)) for tid, cnt in top_non]
                    top_cand_dec = [(safe_decode(tok, tid), int(cnt)) for tid, cnt in top_cand]

                    probe_payload: Dict[str, object] = {"probe_ran": False}
                    if args.do_probes:
                        # Only run probes on a small, user-specified subset of conditions.
                        if (task.name in set(args.probe_tasks)) and (filler_type in set(args.probe_filler_types)) and (k in set(args.probe_k_values)):
                            try:
                                layers = get_decoder_layers(model)
                                probe_layer_indices = parse_probe_layers(args.probe_layers, n_layers=len(layers))
                                probe_payload = run_probes_for_condition(
                                    model, tok, task, pool,
                                    k=k, filler_type=filler_type,
                                    seed=args.seed + 777,
                                    device=device,
                                    layer_indices=probe_layer_indices,
                                    n_train=args.probe_n_train,
                                    n_test=args.probe_n_test,
                                    batch_size=args.probe_batch_size,
                                    lr=args.probe_lr,
                                    weight_decay=args.probe_weight_decay,
                                    epochs=args.probe_epochs,
                                    train_bs=args.probe_train_bs,
                                    decoy_reps=args.decoy_reps,
                                    n_distractors=args.n_distractors,
                                )
                            except Exception as e:
                                probe_payload = {"probe_ran": False, "probe_error": str(e)}
                                gc.collect()
                                try_empty_cache()

                    row = {
                        "model": my_model,
                        "task": task.name,
                        "prompt_style": args.prompt_style,
                        "k": k,
                        "filler_type": filler_type,
                        "n_total": n,
                        "acc_global": acc,
                        "acc_candidate_only": acc_cand,
                        "n_stage2a": n_s2a,
                        "n_stage2b": n_s2b,
                        "frac_stage2a": n_s2a / max(1, (n - n_correct)),
                        "frac_stage2b": n_s2b / max(1, (n - n_correct)),
                        "mean_gate_gap": mean_gate_gap,
                        "mean_value_gap": mean_value_gap,
                        "probe_acc_overall": probe_acc_overall if args.enable_probe else None,
                        "probe_acc_on_stage2b": probe_acc_on_stage2b if args.enable_probe else None,
                        "probe_acc_on_correct": probe_acc_on_correct if args.enable_probe else None,
                        "probe_n_stage2b_trials": probe_n_stage2b_trials if args.enable_probe else None,
                        "top_wrong_non_candidates": json.dumps(top_non_dec, ensure_ascii=False),
                        "top_wrong_candidates": json.dumps(top_cand_dec, ensure_ascii=False),
                    }
                    row.update(probe_payload)
                    rows_local.append(row)

                    if args.enable_probe and is_group_leader:
                        print(f"  {task.name} k={k}: acc={acc:.3f} | probe_overall={probe_acc_overall:.3f} probe_on_stage2b={probe_acc_on_stage2b:.3f} (n={probe_n_stage2b_trials})")

    # Cleanup model memory early
    del model
    gc.collect()
    try_empty_cache()
    ddp_barrier()

    # Global gather: only group leaders contribute rows
    rows_to_send = rows_local if is_group_leader else []
    gathered = all_gather_object(rows_to_send, group=None)
    if is_global_leader:
        all_rows: List[dict] = []
        for part in gathered:
            if part:
                all_rows.extend(part)
        df = pd.DataFrame(all_rows).sort_values(["model", "task", "filler_type", "k"], kind="stable")
        out_csv = os.path.join(args.outdir, args.out_csv)
        out_json = os.path.join(args.outdir, args.out_json)
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"\nWrote:\n- {out_csv}\n- {out_json}")

    ddp_barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

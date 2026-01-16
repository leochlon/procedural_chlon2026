#!/usr/bin/env python3
# Research-quality experiment files used to generate the results for procedural_chlon2026.
"""
mech_stage2b_circuit_suite.py

One last "make reviewers happy" mechanistic experiment:
- Identify *which layers* and *which components* (ATTN vs MLP) causally matter for Stage-2B binding
  using activation patching (clean -> corrupted).
- Then identify *which attention heads* in the decisive layer(s) causally mediate the binding by:
    (a) head-output patching (clean slice -> corrupted slice at o_proj input)
    (b) head ablation (zero the head slice at o_proj input)

This is designed specifically for the Stage-2B regime you already found: wrong-candidate errors with
high probe-on-Stage2B accuracy (e.g., Qwen competing_vars / primacy_recency at k=256). fileciteturn1file0

Key idea: we patch the *input to o_proj*, which is the concatenation of per-head outputs.
This gives true head-level causal control without needing TransformerLens.

Supported architectures: Llama-style blocks (Llama/Gemma/Qwen2-like) where:
  model.model.layers[i].self_attn.o_proj exists,
  model.model.layers[i].mlp exists.

Run (single GPU):
  python mech_stage2b_circuit_suite.py \
    --model Qwen/Qwen2.5-3B \
    --task competing_vars \
    --k 256 \
    --prompt_style bracket \
    --clean_filler random \
    --corrupt_filler decoy_heavy \
    --decoy_reps 12 \
    --n_pairs 40 \
    --layers last8 \
    --top_layers 2 \
    --top_heads 12 \
    --dtype bf16 \
    --attn_impl sdpa \
    --outdir mech_circuit_out

Run (multi-GPU data parallel; each rank finds/patches its own pairs and we all-reduce):
  torchrun --nproc_per_node 8 mech_stage2b_circuit_suite.py ...same args...

Outputs:
  layer_component_restoration.csv   # causal patching for ATTN vs MLP per layer
  head_restoration.csv              # causal head patching/ablation for top layers
  top_heads.json                    # quick summary for paper
  attn_inspect.json (optional)      # correlational attention-to-source for the top heads on 1 pair
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
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
    raise ValueError(f"Unknown dtype {s}")

def try_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def safe_decode(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        return f"<tok:{tid}>"


# -------------------------
# Pool building (single-token words)
# -------------------------

DEFAULT_WORDS = [
    "apple","banana","cherry","grape","lemon","mango","peach","pear","plum","kiwi","melon","berry",
    "alpha","beta","gamma","delta","theta","lambda","omega","sigma","kappa",
    "red","blue","green","yellow","black","white","orange","purple",
    "cat","dog","mouse","lion","tiger","bear","wolf","fox","zebra","shark","whale",
    "yes","no","true","false","left","right","up","down","east","west",
]

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
    for cand in [" the", " and", " of", ".", ","]:
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
        if len(random_ids) >= 5000:
            break
    if len(random_ids) < 1000:
        random_ids = [t for t in range(tokenizer.vocab_size) if t not in special][:2000]

    return Pool(token_ids=ids, filler_id=filler_id, random_ids=random_ids)


# -------------------------
# Fillers
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
        out = [pool.filler_id] * k
        if decoy_id is None:
            return out
        reps = min(decoy_reps, k)
        for i in range(reps):
            pos = int((i + 1) * (k / (reps + 1)))
            pos = max(0, min(k - 1, pos))
            out[pos] = int(decoy_id)
        # light random sprinkling
        for _ in range(min(8, k // 16)):
            out[rng.randrange(0, k)] = rng.choice(pool.random_ids)
        return out
    raise ValueError(f"Unknown filler_type: {filler_type}")


# -------------------------
# Prompt builders + position bookkeeping
# -------------------------

@dataclass
class PairExample:
    clean_ids: torch.Tensor      # [S]
    corrupt_ids: torch.Tensor    # [S]
    target_id: int
    competitor_id: int
    candidate_ids: List[int]
    pos_key1: int
    pos_key2: int

def build_competing_vars_pair(tok, pool: Pool, *,
                              v1: int, v2: int,
                              k: int,
                              prompt_style: str,
                              clean_filler: str,
                              corrupt_filler: str,
                              decoy_reps: int,
                              n_distractors: int,
                              rng: random.Random) -> PairExample:
    # candidate set includes v1, v2 plus distractors
    cand = [int(v1), int(v2)]
    if n_distractors > 0:
        extra = [t for t in pool.token_ids if t not in cand]
        rng.shuffle(extra)
        cand += [int(x) for x in extra[:n_distractors]]
    rng.shuffle(cand)

    fill_clean_1 = make_filler(tok, pool, clean_filler, k, rng, decoy_id=None, decoy_reps=0)
    fill_clean_2 = make_filler(tok, pool, clean_filler, k, rng, decoy_id=None, decoy_reps=0)

    fill_cor_1 = make_filler(tok, pool, corrupt_filler, k, rng, decoy_id=v2, decoy_reps=decoy_reps)
    fill_cor_2 = make_filler(tok, pool, corrupt_filler, k, rng, decoy_id=v2, decoy_reps=decoy_reps)

    def build(fill1, fill2) -> Tuple[List[int], int, int]:
        ids: List[int] = []
        if tok.bos_token_id is not None:
            ids.append(int(tok.bos_token_id))
        if prompt_style == "bracket":
            # KEY1=[ v1 ] \n filler KEY2=[ v2 ] \n filler Q ... KEY1=[
            p1 = "KEY1=["
            close = " ]\n"
            p2 = "KEY2=["
            q = "\nQuestion: What is KEY1?\nKEY1=["
            ids += tok.encode(p1, add_special_tokens=False)
            pos_key1 = len(ids)  # position where v1 will be appended
            ids.append(int(v1))
            ids += tok.encode(close, add_special_tokens=False)
            ids += fill1
            ids += tok.encode(p2, add_special_tokens=False)
            pos_key2 = len(ids)
            ids.append(int(v2))
            ids += tok.encode(close, add_special_tokens=False)
            ids += fill2
            ids += tok.encode(q, add_special_tokens=False)
        else:
            p1 = "KEY1 ="
            p2 = "\nKEY2 ="
            q = "\nQuestion: What is KEY1?\nKEY1 ="
            ids += tok.encode(p1, add_special_tokens=False)
            pos_key1 = len(ids)
            ids.append(int(v1))
            ids += fill1
            ids += tok.encode(p2, add_special_tokens=False)
            pos_key2 = len(ids)
            ids.append(int(v2))
            ids += fill2
            ids += tok.encode(q, add_special_tokens=False)
        return ids, pos_key1, pos_key2

    clean_ids, pos1, pos2 = build(fill_clean_1, fill_clean_2)
    corrupt_ids, pos1b, pos2b = build(fill_cor_1, fill_cor_2)
    assert len(clean_ids) == len(corrupt_ids)
    assert pos1 == pos1b and pos2 == pos2b

    return PairExample(
        clean_ids=torch.tensor(clean_ids, dtype=torch.long),
        corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),
        target_id=int(v1),
        competitor_id=int(v2),
        candidate_ids=cand,
        pos_key1=pos1,
        pos_key2=pos2,
    )

def build_primacy_recency_pair(tok, pool: Pool, *,
                              v1: int, v2: int, v3: int,
                              k: int,
                              prompt_style: str,
                              clean_filler: str,
                              corrupt_filler: str,
                              decoy_reps: int,
                              n_distractors: int,
                              rng: random.Random) -> PairExample:
    # candidates include the three values (+ distractors)
    cand = [int(v1), int(v2), int(v3)]
    if n_distractors > 0:
        extra = [t for t in pool.token_ids if t not in cand]
        rng.shuffle(extra)
        cand += [int(x) for x in extra[:n_distractors]]
    rng.shuffle(cand)

    # clean fillers: neutral
    fill_clean = make_filler(tok, pool, clean_filler, k, rng, decoy_id=None, decoy_reps=0)
    # corrupt fillers: decoy heavy with last value v3 repeated
    fill_cor = make_filler(tok, pool, corrupt_filler, k, rng, decoy_id=v3, decoy_reps=decoy_reps)

    def build(fill) -> Tuple[List[int], int, int]:
        ids: List[int] = []
        if tok.bos_token_id is not None:
            ids.append(int(tok.bos_token_id))
        if prompt_style == "bracket":
            p = "KEY=["
            close = " ]\n"
            q = "\nQuestion: What was the FIRST value of KEY?\nKEY=["
            ids += tok.encode(p, add_special_tokens=False); pos_key1 = len(ids); ids.append(int(v1)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill
            ids += tok.encode(p, add_special_tokens=False); _ = len(ids); ids.append(int(v2)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill
            ids += tok.encode(p, add_special_tokens=False); pos_key2 = len(ids); ids.append(int(v3)); ids += tok.encode(close, add_special_tokens=False)
            ids += fill
            ids += tok.encode(q, add_special_tokens=False)
        else:
            p = "KEY ="
            q = "\nQuestion: What was the FIRST value of KEY?\nKEY ="
            ids += tok.encode(p, add_special_tokens=False); pos_key1 = len(ids); ids.append(int(v1))
            ids += fill
            ids += tok.encode("\nKEY =", add_special_tokens=False); _ = len(ids); ids.append(int(v2))
            ids += fill
            ids += tok.encode("\nKEY =", add_special_tokens=False); pos_key2 = len(ids); ids.append(int(v3))
            ids += fill
            ids += tok.encode(q, add_special_tokens=False)
        return ids, pos_key1, pos_key2

    clean_ids, pos_first, pos_last = build(fill_clean)
    corrupt_ids, pos_first_b, pos_last_b = build(fill_cor)
    assert len(clean_ids) == len(corrupt_ids)
    assert pos_first == pos_first_b and pos_last == pos_last_b

    return PairExample(
        clean_ids=torch.tensor(clean_ids, dtype=torch.long),
        corrupt_ids=torch.tensor(corrupt_ids, dtype=torch.long),
        target_id=int(v1),
        competitor_id=int(v3),  # recency competitor
        candidate_ids=cand,
        pos_key1=pos_first,
        pos_key2=pos_last,
    )


# -------------------------
# Model access helpers
# -------------------------

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Unsupported model: expected model.model.layers")

def get_attn_o_proj(layer):
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise RuntimeError("Layer has no self_attn")
    for name in ("o_proj", "out_proj", "dense"):
        if hasattr(attn, name):
            return getattr(attn, name)
    raise RuntimeError("Could not find attention output projection (o_proj/out_proj/dense)")

def get_mlp_module(layer):
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise RuntimeError("Layer has no mlp")
    return mlp

@torch.no_grad()
def forward_logits(model, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    input_ids = input_ids.unsqueeze(0).to(device)
    attn_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_attentions=False)
    return out.logits[0, -1, :].detach()  # [V]

def argmax_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())

def in_candidates(tid: int, cand: List[int]) -> bool:
    s = set(int(x) for x in cand)
    return int(tid) in s

def margin(logits: torch.Tensor, a: int, b: int) -> float:
    return float(logits[int(a)].item() - logits[int(b)].item())


# -------------------------
# Hook management for patching/ablation
# -------------------------

class HookManager:
    def __init__(self):
        self.handles = []
    def add(self, handle):
        self.handles.append(handle)
    def clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


# -------------------------
# Core patching logic
# -------------------------

@torch.no_grad()
def record_clean_vectors(model, layers, layer_ids: List[int], input_ids: torch.Tensor, device: torch.device) -> Dict[int, torch.Tensor]:
    """
    Record the input to o_proj at the final (answer) position for specified layers.
    Returns dict: layer_id -> tensor[hidden] on CPU (float16/bfloat16).
    """
    hm = HookManager()
    store: Dict[int, torch.Tensor] = {}

    for li in layer_ids:
        o_proj = get_attn_o_proj(layers[li])
        def make_hook(layer_index: int):
            def pre_hook(module, inputs):
                x = inputs[0]  # [B, S, H]
                # store last position only; move to CPU
                v = x[0, -1, :].detach().to("cpu")
                store[layer_index] = v
                return None
            return pre_hook
        hm.add(o_proj.register_forward_pre_hook(make_hook(li)))

    _ = forward_logits(model, input_ids, device=device)
    hm.clear()
    return store

@torch.no_grad()
def record_clean_mlp_outputs(model, layers, layer_ids: List[int], input_ids: torch.Tensor, device: torch.device) -> Dict[int, torch.Tensor]:
    """
    Record the output of the MLP module at the final position for specified layers.
    Returns dict: layer_id -> tensor[hidden] on CPU.
    """
    hm = HookManager()
    store: Dict[int, torch.Tensor] = {}

    for li in layer_ids:
        mlp = get_mlp_module(layers[li])
        def make_hook(layer_index: int):
            def fwd_hook(module, inputs, output):
                # output: [B,S,H]
                v = output[0, -1, :].detach().to("cpu")
                store[layer_index] = v
                return output
            return fwd_hook
        hm.add(mlp.register_forward_hook(make_hook(li)))

    _ = forward_logits(model, input_ids, device=device)
    hm.clear()
    return store

@torch.no_grad()
def run_with_attn_vec_patch(model, layers, li: int, clean_vec: torch.Tensor, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Patch entire attention concatenated-head vector (input to o_proj) at last position.
    """
    hm = HookManager()
    o_proj = get_attn_o_proj(layers[li])

    clean_vec = clean_vec.to(device)

    def pre_hook(module, inputs):
        x = inputs[0]  # [B,S,H]
        x2 = x.clone()
        x2[0, -1, :] = clean_vec
        return (x2,)
    hm.add(o_proj.register_forward_pre_hook(pre_hook))
    logits = forward_logits(model, input_ids, device=device)
    hm.clear()
    return logits

@torch.no_grad()
def run_with_mlp_out_patch(model, layers, li: int, clean_mlp: torch.Tensor, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Patch MLP output vector at last position (replace output).
    """
    hm = HookManager()
    mlp = get_mlp_module(layers[li])
    clean_mlp = clean_mlp.to(device)

    def fwd_hook(module, inputs, output):
        y = output.clone()
        y[0, -1, :] = clean_mlp
        return y
    hm.add(mlp.register_forward_hook(fwd_hook))
    logits = forward_logits(model, input_ids, device=device)
    hm.clear()
    return logits

@torch.no_grad()
def run_with_head_patch(model, layers, li: int, head: int, head_dim: int, clean_vec: torch.Tensor,
                        input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Patch a single head slice at o_proj input at last position.
    """
    hm = HookManager()
    o_proj = get_attn_o_proj(layers[li])

    clean_vec = clean_vec.to(device)
    a = int(head * head_dim)
    b = int((head + 1) * head_dim)

    def pre_hook(module, inputs):
        x = inputs[0]
        x2 = x.clone()
        x2[0, -1, a:b] = clean_vec[a:b]
        return (x2,)
    hm.add(o_proj.register_forward_pre_hook(pre_hook))
    logits = forward_logits(model, input_ids, device=device)
    hm.clear()
    return logits

@torch.no_grad()
def run_with_head_ablation(model, layers, li: int, head: int, head_dim: int,
                           input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ablate a head slice at o_proj input at last position by zeroing it.
    """
    hm = HookManager()
    o_proj = get_attn_o_proj(layers[li])
    a = int(head * head_dim)
    b = int((head + 1) * head_dim)

    def pre_hook(module, inputs):
        x = inputs[0]
        x2 = x.clone()
        x2[0, -1, a:b] = 0
        return (x2,)
    hm.add(o_proj.register_forward_pre_hook(pre_hook))
    logits = forward_logits(model, input_ids, device=device)
    hm.clear()
    return logits


# -------------------------
# Pair search (ensure clean correct and corrupt Stage-2B wrong)
# -------------------------

@torch.no_grad()
def find_pairs(model, tok, pool: Pool, device: torch.device,
               *, task: str, k: int, prompt_style: str,
               clean_filler: str, corrupt_filler: str, decoy_reps: int,
               n_distractors: int,
               n_pairs: int, max_tries: int,
               require_competitor: bool,
               seed: int,
               show_progress: bool = False) -> List[PairExample]:
    rng = random.Random(seed)
    pairs: List[PairExample] = []
    tries = 0
    pbar = tqdm(total=n_pairs, desc="Finding pairs", disable=not show_progress)
    while len(pairs) < n_pairs and tries < max_tries:
        tries += 1
        if task == "competing_vars":
            v1 = rng.choice(pool.token_ids)
            v2 = rng.choice([t for t in pool.token_ids if t != v1])
            ex = build_competing_vars_pair(tok, pool, v1=v1, v2=v2, k=k, prompt_style=prompt_style,
                                           clean_filler=clean_filler, corrupt_filler=corrupt_filler,
                                           decoy_reps=decoy_reps, n_distractors=n_distractors, rng=rng)
        elif task == "primacy_recency":
            v1 = rng.choice(pool.token_ids)
            v2 = rng.choice([t for t in pool.token_ids if t != v1])
            v3 = rng.choice([t for t in pool.token_ids if t not in (v1, v2)])
            ex = build_primacy_recency_pair(tok, pool, v1=v1, v2=v2, v3=v3, k=k, prompt_style=prompt_style,
                                            clean_filler=clean_filler, corrupt_filler=corrupt_filler,
                                            decoy_reps=decoy_reps, n_distractors=n_distractors, rng=rng)
        else:
            raise ValueError(f"Unknown task: {task}")

        # Check behavior
        logits_clean = forward_logits(model, ex.clean_ids, device=device)
        logits_cor = forward_logits(model, ex.corrupt_ids, device=device)
        pred_clean = argmax_token(logits_clean)
        pred_cor = argmax_token(logits_cor)

        if pred_clean != ex.target_id:
            continue
        if not in_candidates(pred_cor, ex.candidate_ids):
            continue
        if pred_cor == ex.target_id:
            continue  # want wrong candidate (Stage2B)
        if require_competitor and pred_cor != ex.competitor_id:
            continue

        pairs.append(ex)
        pbar.update(1)

    pbar.close()
    if len(pairs) < n_pairs:
        print(f"[WARN] Found only {len(pairs)}/{n_pairs} suitable pairs after {tries} tries. "
              f"Consider lowering --require_competitor or adjusting k/decoy_reps.")
    return pairs


# -------------------------
# Layer parsing
# -------------------------

def parse_layers(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec == "last8":
        return list(range(max(0, n_layers - 8), n_layers))
    if spec == "last4":
        return list(range(max(0, n_layers - 4), n_layers))
    if spec == "mid":
        return [n_layers // 2]
    # comma list
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    for x in out:
        if x < 0 or x >= n_layers:
            raise ValueError(f"Layer index {x} out of range [0,{n_layers-1}]")
    return out


# -------------------------
# Optional attention inspection (correlational) for paper figure
# -------------------------

@torch.no_grad()
def attention_inspect_one_pair(model, tok, ex: PairExample, *, layers_to_check: List[int], heads_to_check: List[int],
                               device: torch.device) -> dict:
    """
    Runs model with output_attentions=True on clean+corrupt for ONE pair and returns
    attn weight to KEY1 vs KEY2 positions for selected layer/head from the final query position.
    """
    clean = ex.clean_ids.unsqueeze(0).to(device)
    cor = ex.corrupt_ids.unsqueeze(0).to(device)
    mask_c = torch.ones_like(clean)
    mask_k = torch.ones_like(cor)

    out_clean = model(input_ids=clean, attention_mask=mask_c, use_cache=False, output_attentions=True)
    out_cor = model(input_ids=cor, attention_mask=mask_k, use_cache=False, output_attentions=True)

    # attentions list length = n_layers; each [B, n_heads, S, S]
    att_clean = out_clean.attentions
    att_cor = out_cor.attentions

    res = {"pos_key1": ex.pos_key1, "pos_key2": ex.pos_key2, "layers": {}}
    qpos = clean.shape[1] - 1
    for li in layers_to_check:
        a_c = att_clean[li][0]  # [H,S,S]
        a_k = att_cor[li][0]
        layer_entry = {}
        for h in heads_to_check:
            w1_c = float(a_c[h, qpos, ex.pos_key1].item())
            w2_c = float(a_c[h, qpos, ex.pos_key2].item())
            w1_k = float(a_k[h, qpos, ex.pos_key1].item())
            w2_k = float(a_k[h, qpos, ex.pos_key2].item())
            layer_entry[str(h)] = {
                "clean_attn_key1": w1_c,
                "clean_attn_key2": w2_c,
                "corrupt_attn_key1": w1_k,
                "corrupt_attn_key2": w2_k,
            }
        res["layers"][str(li)] = layer_entry
    return res


# -------------------------
# Main experiment
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--task", type=str, default="competing_vars", choices=["competing_vars", "primacy_recency"])
    ap.add_argument("--prompt_style", type=str, default="bracket", choices=["bracket", "eq"])
    ap.add_argument("--k", type=int, default=256)

    ap.add_argument("--clean_filler", type=str, default="random", choices=["repeat", "coherent", "random"])
    ap.add_argument("--corrupt_filler", type=str, default="decoy_heavy", choices=["repeat", "coherent", "random", "decoy_heavy"])
    ap.add_argument("--decoy_reps", type=int, default=12)
    ap.add_argument("--n_distractors", type=int, default=6)
    ap.add_argument("--num_values", type=int, default=128)

    ap.add_argument("--n_pairs", type=int, default=40)
    ap.add_argument("--max_tries", type=int, default=20000)
    ap.add_argument("--require_competitor", action="store_true", help="Only keep pairs where corrupt prediction equals competitor.")
    ap.add_argument("--layers", type=str, default="last8")
    ap.add_argument("--top_layers", type=int, default=2)
    ap.add_argument("--top_heads", type=int, default=12)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2", "auto"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--do_attn_inspect", action="store_true", help="Also dump attention-to-source for top heads on 1 pair.")
    ap.add_argument("--outdir", type=str, default="mech_circuit_out")

    args = ap.parse_args()

    rank, world, local = ddp_init()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed + 1000 * rank)

    ensure_dir(args.outdir)
    dtype = torch_dtype_from_str(args.dtype)

    # Load
    if rank == 0:
        print(f"Loading model {args.model} (dtype={args.dtype}, attn_impl={args.attn_impl})")
    ddp_barrier()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(trust_remote_code=True)
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
    model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    model.to(device)

    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = int(getattr(model.config, "num_attention_heads"))
    hidden = int(getattr(model.config, "hidden_size"))
    head_dim = hidden // n_heads

    layer_ids = parse_layers(args.layers, n_layers)
    if rank == 0:
        print(f"n_layers={n_layers} n_heads={n_heads} hidden={hidden} head_dim={head_dim}")
        print(f"Layer sweep set: {layer_ids}")

    pool = build_pool(tok, num_values=args.num_values, seed=args.seed + 77)

    # Find Stage-2B pairs on each rank
    pairs = find_pairs(
        model, tok, pool, device,
        task=args.task, k=args.k, prompt_style=args.prompt_style,
        clean_filler=args.clean_filler,
        corrupt_filler=args.corrupt_filler,
        decoy_reps=args.decoy_reps,
        n_distractors=args.n_distractors,
        n_pairs=math.ceil(args.n_pairs / max(1, world)),
        max_tries=args.max_tries,
        require_competitor=args.require_competitor,
        seed=args.seed + 12345 + rank * 17,
        show_progress=(rank == 0),
    )

    # Accumulators for layer-component patching
    # sum_restore_attn[layer_i], sum_restore_mlp[layer_i], counts
    sum_attn = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)
    sum_mlp = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)
    cnt_attn = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)
    cnt_mlp = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)

    # For selecting top layers: keep sum of delta margins too
    sum_dmargin_attn = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)
    sum_dmargin_mlp = torch.zeros((len(layer_ids),), dtype=torch.float64, device=device)

    # Store per-layer restoration list for ranking (on rank0 later)
    # We'll reconstruct from reduced tensors.

    # Run layer component patching for each pair (ATTN vec patch and MLP out patch)
    pairs_iter = tqdm(pairs, desc="Layer-component patching", disable=(rank != 0))
    for ex in pairs_iter:
        # Baseline logits
        logits_clean = forward_logits(model, ex.clean_ids, device=device)
        logits_cor = forward_logits(model, ex.corrupt_ids, device=device)

        m_clean = margin(logits_clean, ex.target_id, ex.competitor_id)
        m_cor = margin(logits_cor, ex.target_id, ex.competitor_id)
        denom = (m_clean - m_cor)
        if abs(denom) < 1e-6:
            continue

        clean_attn_vecs = record_clean_vectors(model, layers, layer_ids, ex.clean_ids, device=device)
        clean_mlp_vecs = record_clean_mlp_outputs(model, layers, layer_ids, ex.clean_ids, device=device)

        for j, li in enumerate(layer_ids):
            # Attention vector patch
            if li in clean_attn_vecs:
                logits_p = run_with_attn_vec_patch(model, layers, li, clean_attn_vecs[li], ex.corrupt_ids, device=device)
                m_p = margin(logits_p, ex.target_id, ex.competitor_id)
                rest = (m_p - m_cor) / (denom + 1e-9)
                sum_attn[j] += float(rest)
                cnt_attn[j] += 1.0
                sum_dmargin_attn[j] += float(m_p - m_cor)

            # MLP output patch
            if li in clean_mlp_vecs:
                logits_p = run_with_mlp_out_patch(model, layers, li, clean_mlp_vecs[li], ex.corrupt_ids, device=device)
                m_p = margin(logits_p, ex.target_id, ex.competitor_id)
                rest = (m_p - m_cor) / (denom + 1e-9)
                sum_mlp[j] += float(rest)
                cnt_mlp[j] += 1.0
                sum_dmargin_mlp[j] += float(m_p - m_cor)

        # periodic cleanup
        gc.collect()
        try_empty_cache()

    # Reduce across ranks
    all_reduce_(sum_attn); all_reduce_(cnt_attn); all_reduce_(sum_dmargin_attn)
    all_reduce_(sum_mlp);  all_reduce_(cnt_mlp);  all_reduce_(sum_dmargin_mlp)

    if rank == 0:
        # Compute means
        mean_attn = (sum_attn / torch.clamp(cnt_attn, min=1.0)).detach().cpu().numpy()
        mean_mlp = (sum_mlp / torch.clamp(cnt_mlp, min=1.0)).detach().cpu().numpy()
        mean_dattn = (sum_dmargin_attn / torch.clamp(cnt_attn, min=1.0)).detach().cpu().numpy()
        mean_dmlp = (sum_dmargin_mlp / torch.clamp(cnt_mlp, min=1.0)).detach().cpu().numpy()

        rows = []
        for idx, li in enumerate(layer_ids):
            rows.append({"layer": li, "component": "attn", "mean_restoration": float(mean_attn[idx]), "mean_delta_margin": float(mean_dattn[idx]), "n": int(cnt_attn[idx].item())})
            rows.append({"layer": li, "component": "mlp",  "mean_restoration": float(mean_mlp[idx]),  "mean_delta_margin": float(mean_dmlp[idx]),  "n": int(cnt_mlp[idx].item())})
        df_layer = pd.DataFrame(rows).sort_values(["layer", "component"])
        out_csv = os.path.join(args.outdir, "layer_component_restoration.csv")
        df_layer.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

        # Choose top layers by ATTN restoration (tie-break by delta margin)
        layer_scores = []
        for idx, li in enumerate(layer_ids):
            layer_scores.append((float(mean_attn[idx]), float(mean_dattn[idx]), int(li)))
        layer_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen_layers = [li for _, __, li in layer_scores[:max(1, args.top_layers)]]
        print(f"Top layers for head sweep (by ATTN patch restoration): {chosen_layers}")

    else:
        chosen_layers = None

    # Broadcast chosen_layers to all ranks
    if is_dist():
        obj_list = [chosen_layers]
        dist.broadcast_object_list(obj_list, src=0)
        chosen_layers = obj_list[0]

    # Head-level sweep only on chosen layers
    # Each rank reuses its local pairs.
    sum_head_patch = torch.zeros((len(chosen_layers), n_heads), dtype=torch.float64, device=device)
    cnt_head_patch = torch.zeros((len(chosen_layers), n_heads), dtype=torch.float64, device=device)
    sum_head_ablate = torch.zeros((len(chosen_layers), n_heads), dtype=torch.float64, device=device)
    cnt_head_ablate = torch.zeros((len(chosen_layers), n_heads), dtype=torch.float64, device=device)

    pairs_iter = tqdm(pairs, desc="Head-level patching", disable=(rank != 0))
    for ex in pairs_iter:
        logits_clean = forward_logits(model, ex.clean_ids, device=device)
        logits_cor = forward_logits(model, ex.corrupt_ids, device=device)
        m_clean = margin(logits_clean, ex.target_id, ex.competitor_id)
        m_cor = margin(logits_cor, ex.target_id, ex.competitor_id)
        denom = (m_clean - m_cor)
        if abs(denom) < 1e-6:
            continue

        clean_attn_vecs = record_clean_vectors(model, layers, chosen_layers, ex.clean_ids, device=device)

        for li_idx, li in enumerate(chosen_layers):
            if li not in clean_attn_vecs:
                continue
            clean_vec = clean_attn_vecs[li]
            for h in range(n_heads):
                # Patch
                logits_p = run_with_head_patch(model, layers, li, h, head_dim, clean_vec, ex.corrupt_ids, device=device)
                m_p = margin(logits_p, ex.target_id, ex.competitor_id)
                rest = (m_p - m_cor) / (denom + 1e-9)
                sum_head_patch[li_idx, h] += float(rest)
                cnt_head_patch[li_idx, h] += 1.0

                # Ablate
                logits_a = run_with_head_ablation(model, layers, li, h, head_dim, ex.corrupt_ids, device=device)
                m_a = margin(logits_a, ex.target_id, ex.competitor_id)
                delta = (m_a - m_cor)  # positive means ablation helps correct-vs-competitor margin
                sum_head_ablate[li_idx, h] += float(delta)
                cnt_head_ablate[li_idx, h] += 1.0

            gc.collect()
            try_empty_cache()

    all_reduce_(sum_head_patch); all_reduce_(cnt_head_patch)
    all_reduce_(sum_head_ablate); all_reduce_(cnt_head_ablate)

    if rank == 0:
        mean_patch = (sum_head_patch / torch.clamp(cnt_head_patch, min=1.0)).detach().cpu().numpy()
        mean_ablate = (sum_head_ablate / torch.clamp(cnt_head_ablate, min=1.0)).detach().cpu().numpy()

        head_rows = []
        for li_idx, li in enumerate(chosen_layers):
            for h in range(n_heads):
                head_rows.append({
                    "layer": int(li),
                    "head": int(h),
                    "mean_patch_restoration": float(mean_patch[li_idx, h]),
                    "mean_ablation_delta_margin": float(mean_ablate[li_idx, h]),
                    "n": int(cnt_head_patch[li_idx, h].item()),
                })
        df_head = pd.DataFrame(head_rows).sort_values(["layer", "mean_patch_restoration"], ascending=[True, False])
        out_csv = os.path.join(args.outdir, "head_restoration.csv")
        df_head.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

        # Summarize top heads overall
        topk = int(args.top_heads)
        df_sorted = df_head.sort_values(["mean_patch_restoration"], ascending=False).head(topk)
        summary = {
            "model": args.model,
            "task": args.task,
            "k": args.k,
            "prompt_style": args.prompt_style,
            "clean_filler": args.clean_filler,
            "corrupt_filler": args.corrupt_filler,
            "decoy_reps": args.decoy_reps,
            "n_pairs_total_requested": args.n_pairs,
            "layers_swept": layer_ids,
            "chosen_layers_for_head_sweep": chosen_layers,
            "top_heads_by_patch_restoration": df_sorted.to_dict(orient="records"),
            "interpretation_notes": [
                "mean_patch_restoration > 0: patching that head's output slice (clean->corrupt) increases target-vs-competitor margin",
                "mean_ablation_delta_margin > 0: ablation on corrupt increases target-vs-competitor margin (head contributes to misbinding or interference)",
                "mean_ablation_delta_margin < 0: ablation hurts (head contributes to correct binding even in corrupt condition)",
            ],
        }
        out_json = os.path.join(args.outdir, "top_heads.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_json}")

    # Optional attention inspection on 1 representative pair (rank0 only)
    if args.do_attn_inspect and rank == 0:
        # choose a single pair from gathered pairs list (across ranks) by gathering just 1 per rank
        local_one = pairs[0] if pairs else None
        gathered = all_gather_object(local_one)
        ex0 = None
        for g in gathered:
            if g is not None:
                ex0 = g
                break
        if ex0 is None:
            print("[WARN] No pair available for attention inspect.")
        else:
            # parse top heads from df_head
            try:
                df_head = pd.read_csv(os.path.join(args.outdir, "head_restoration.csv"))
                top = df_head.sort_values("mean_patch_restoration", ascending=False).head(min(8, len(df_head)))
                layers_check = sorted(list(set(int(x) for x in top["layer"].tolist())))
                heads_check = [int(x) for x in top["head"].tolist()]
            except Exception:
                layers_check = chosen_layers
                heads_check = list(range(min(4, n_heads)))
            attn_info = attention_inspect_one_pair(model, tok, ex0, layers_to_check=layers_check, heads_to_check=heads_check, device=device)
            out_json = os.path.join(args.outdir, "attn_inspect.json")
            with open(out_json, "w") as f:
                json.dump(attn_info, f, indent=2)
            print(f"Wrote {out_json}")

    # Cleanup
    del model
    gc.collect()
    try_empty_cache()
    ddp_barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

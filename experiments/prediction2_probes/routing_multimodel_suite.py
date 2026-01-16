#!/usr/bin/env python3
# Research-quality experiment files used to generate the results for procedural_chlon2026.
"""
routing_multimodel_suite_v2.py

V2: Fixes identified issues
- Token-ID filler insertion (Phi-3 no longer fails due to "no single-token filler string")
- Prefix-aware accuracy computed from generate() scores (no slicing bug; monotone vs strict)
- Iterative competitor patching (diagnoses multiple competitors above target)
- Separates candidate-only vs global-vocab accuracy and classifies failures as format-token interference

Run (example):
python routing_multimodel_suite_v2.py \
  --models Qwen/Qwen2.5-3B-Instruct google/gemma-2-2b-it microsoft/Phi-3-mini-4k-instruct \
  --k_values 128 256 512 \
  --trials_per_k 2000 \
  --batch_size 8 \
  --prompt_style eq \
  --bf16 \
  --do_probe --probe_k 256 \
  --do_prefix --prefix_k 256 --prefix_subset 512 --max_new_tokens 3 \
  --do_iter_patch --patch_k 256 --patch_subset_wrong 256 --patch_max_iters 8 \
  --out_json results_v2.json --out_csv results_v2.csv

Prompt styles:
- eq: assignment uses "KEY =" (no trailing space) + value with leading space, answer prefix "KEY ="
- eq_trailing_space: answer prefix ends with space "KEY = ", values have no leading space
- bracket: answer prefix "KEY=[", assignment writes "KEY=[{value} ]" (space before closing bracket)
"""

import argparse
import gc
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Repro & utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return "<decode_error>"


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def left_pad(tokenizer) -> None:
    tokenizer.padding_side = "left"


def add_bos_if_needed(tokenizer, ids: List[int]) -> List[int]:
    # Add BOS once at the beginning if the tokenizer/model uses it and it's not already present.
    bos = tokenizer.bos_token_id
    if bos is None:
        return ids
    if len(ids) > 0 and ids[0] == bos:
        return ids
    return [bos] + ids


def pick_dtype_kwargs(bf16: bool) -> Dict:
    # Prefer `dtype=` if supported; fall back to torch_dtype
    import inspect
    kwargs = {}
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if bf16 and torch.cuda.is_available():
        if "dtype" in sig.parameters:
            kwargs["dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


# ----------------------------
# Prompt styles
# ----------------------------

@dataclass
class PromptStyle:
    name: str
    instruction: str
    assignment_prefix: str
    assignment_suffix: str
    answer_prefix: str
    value_prefix: str  # inserted before raw value at both assignment and answer


def get_prompt_style(name: str) -> PromptStyle:
    name = name.strip().lower()
    instr = (
        "Read the following log. A variable KEY is defined exactly once. "
        "Return ONLY the value of KEY.\n"
    )
    if name == "eq":
        # critical: NO extra space after "="; value carries the single leading space
        return PromptStyle(
            name="eq",
            instruction=instr,
            assignment_prefix="KEY =",
            assignment_suffix="\n",
            answer_prefix="KEY =",
            value_prefix=" ",
        )
    if name == "eq_trailing_space":
        return PromptStyle(
            name="eq_trailing_space",
            instruction=instr,
            assignment_prefix="KEY = ",
            assignment_suffix="\n",
            answer_prefix="KEY = ",
            value_prefix="",
        )
    if name == "bracket":
        # assignment includes a space before closing bracket to prevent token merges like "VAL]"
        return PromptStyle(
            name="bracket",
            instruction=instr,
            assignment_prefix="KEY=[",
            assignment_suffix=" ]\n",
            answer_prefix="KEY=[",
            value_prefix="",
        )
    raise ValueError("prompt_style must be one of: eq, eq_trailing_space, bracket")


def suffix_text(style: PromptStyle) -> str:
    return "\nNow answer the question. What is KEY?\n" + style.answer_prefix


# ----------------------------
# Token-ID filler (Phi-3 fix)
# ----------------------------

def pick_filler_token_id(tokenizer, avoid_ids: set, seed: int, tries: int = 50000) -> int:
    """
    Pick a filler TOKEN ID (not string) that is:
    - not in avoid_ids (candidate values)
    - decodes to something non-empty and not purely whitespace
    Strategy:
      1) Try a short list of common strings if they are 1 token.
      2) Otherwise randomly sample token IDs from vocab until a usable one is found.
    """
    set_seed(seed)
    common = [" the", " a", " and", " of", " to", " in", " for", " ,", " .", ":", ";"]
    for s in common:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in avoid_ids:
            dec = safe_decode(tokenizer, ids[0])
            if dec.strip() != "":
                return int(ids[0])

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for _ in range(tries):
        tid = random.randrange(vocab_size)
        if tid in avoid_ids:
            continue
        dec = safe_decode(tokenizer, tid)
        # avoid special/meta tokens and pure whitespace
        if dec.strip() == "":
            continue
        if "<|" in dec or "|>" in dec or dec.startswith("<"):
            continue
        return int(tid)

    raise RuntimeError("Failed to pick a filler token id")


# ----------------------------
# Candidate value selection (single-token at answer boundary, consistent token id)
# ----------------------------

def find_values(
    tokenizer,
    style: PromptStyle,
    M: int,
    seed: int,
    max_tries: int = 200000,
) -> Tuple[List[str], List[int]]:
    """
    Choose M raw alphanumeric strings such that:
      - in answer context: encode(suffix + value_prefix+raw) = encode(suffix) + [one token]
      - in assignment context: encode(instruction+assignment_prefix + value_prefix+raw) ends in same token id
    This ensures the same token id appears in assignment & answer (clean copy task).
    """
    set_seed(seed)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    suf0 = suffix_text(style)
    suf_ids0 = tokenizer.encode(suf0, add_special_tokens=False)

    pre0 = style.instruction + style.assignment_prefix
    pre_ids0 = tokenizer.encode(pre0, add_special_tokens=False)

    values: List[str] = []
    token_ids: List[int] = []
    seen = set()

    for _ in range(max_tries):
        L = random.choice([2, 3, 4, 5])
        raw = "".join(random.choice(chars) for _ in range(L))
        val = style.value_prefix + raw

        suf_ids1 = tokenizer.encode(suf0 + val, add_special_tokens=False)
        if len(suf_ids1) != len(suf_ids0) + 1:
            continue
        if suf_ids1[:-1] != suf_ids0:
            continue
        tid_ans = suf_ids1[-1]

        pre_ids1 = tokenizer.encode(pre0 + val, add_special_tokens=False)
        if len(pre_ids1) != len(pre_ids0) + 1:
            continue
        if pre_ids1[:-1] != pre_ids0:
            continue
        tid_pre = pre_ids1[-1]

        if tid_pre != tid_ans:
            continue
        if tid_ans in seen:
            continue

        dec = safe_decode(tokenizer, tid_ans)
        if dec.strip() == "":
            continue
        # avoid obviously format-like tokens
        if any(ch in dec for ch in ["\n", "\t"]):
            continue

        seen.add(tid_ans)
        values.append(raw)
        token_ids.append(int(tid_ans))
        if len(values) >= M:
            break

    if len(values) < M:
        raise RuntimeError(f"Could only find {len(values)} values; try smaller M or different prompt_style.")
    return values, token_ids


# ----------------------------
# Building tokenized prompts with token-id filler
# ----------------------------

def build_prompt_ids_for_value(
    tokenizer,
    style: PromptStyle,
    raw_value: str,
    filler_id: int,
    k: int,
    suffix_ids_cached: Optional[List[int]] = None,
) -> List[int]:
    """
    Prompt = instruction + assignment + (filler_id repeated k times) + suffix(answer prefix)
    All encoded with add_special_tokens=False and then optional BOS added once.
    """
    val = style.value_prefix + raw_value
    prefix_text = style.instruction + style.assignment_prefix + val + style.assignment_suffix
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    if suffix_ids_cached is None:
        suffix_ids_cached = tokenizer.encode(suffix_text(style), add_special_tokens=False)

    ids = prefix_ids + [int(filler_id)] * int(k) + suffix_ids_cached
    ids = add_bos_if_needed(tokenizer, ids)
    return ids


def pad_left(input_ids: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-pad a list of equal/variable length sequences.
    Returns:
      input_ids_tensor [B,S], attention_mask [B,S]
    """
    max_len = max(len(x) for x in input_ids)
    B = len(input_ids)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, seq in enumerate(input_ids):
        seq = torch.tensor(seq, dtype=torch.long)
        ids[i, -len(seq):] = seq
        mask[i, -len(seq):] = 1
    return ids, mask


# ----------------------------
# Forward + metrics
# ----------------------------

@torch.inference_mode()
def forward_last_logits_and_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    need_hidden: bool,
    probe_layer: int,
    bf16: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns logits at last non-pad position: [B,V]
    and hidden at last non-pad from probe_layer: [B,d] if need_hidden
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (bf16 and device.type == "cuda")
        else nullcontext()
    )
    with autocast_ctx:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
            use_cache=False,
        )

    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    B = input_ids.shape[0]
    logits_last = out.logits[torch.arange(B, device=device), last_idx, :].detach()  # [B,V]

    hidden_last = None
    if need_hidden:
        hs = out.hidden_states[probe_layer]  # [B,S,d]
        hidden_last = hs[torch.arange(B, device=device), last_idx, :].detach()  # [B,d]

    return logits_last, hidden_last


def topk_contains(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    topk_ids = torch.topk(logits, k=k, dim=-1).indices
    return (topk_ids == target.unsqueeze(1)).any(dim=1)


# ----------------------------
# Probe
# ----------------------------

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.lin(x)


def train_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int, seed: int, epochs: int, device: str):
    set_seed(seed)
    N, d = X.shape
    perm = torch.randperm(N)
    split = int(0.75 * N)
    tr, te = perm[:split], perm[split:]

    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    Xte, yte = X[te].to(device), y[te].to(device)

    probe = LinearProbe(d, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2)

    for _ in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(probe(Xtr), ytr)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.inference_mode():
        acc = (probe(Xte).argmax(dim=-1) == yte).float().mean().item()
    return probe, float(acc)


@torch.inference_mode()
def probe_metrics(probe: LinearProbe, X: torch.Tensor, y: torch.Tensor, base_correct: torch.Tensor, device: str):
    probe.eval()
    pred = probe(X.to(device)).argmax(dim=-1).cpu()
    y = y.cpu()
    base_correct = base_correct.cpu().bool()
    overall = (pred == y).float().mean().item()

    wrong = ~base_correct
    right = base_correct
    on_wrong = float("nan") if not wrong.any() else (pred[wrong] == y[wrong]).float().mean().item()
    on_right = float("nan") if not right.any() else (pred[right] == y[right]).float().mean().item()
    return {"probe_acc_overall": float(overall), "probe_acc_on_wrong": float(on_wrong), "probe_acc_on_right": float(on_right)}


@torch.inference_mode()
def alignment_probe_vs_lm_head(probe: LinearProbe, model, value_token_ids: List[int], n_pairs: int, seed: int):
    set_seed(seed)
    device = next(model.parameters()).device
    lm_head = model.get_output_embeddings()
    W_lm = lm_head.weight[value_token_ids].detach().float().to(device)  # [M,d]
    Wp = probe.lin.weight.detach().float().to(device)                   # [M,d]
    M = Wp.shape[0]

    cos = []
    for _ in range(n_pairs):
        i, j = random.sample(range(M), 2)
        u1 = Wp[i] - Wp[j]
        u2 = W_lm[i] - W_lm[j]
        u1 = u1 / (u1.norm() + 1e-9)
        u2 = u2 / (u2.norm() + 1e-9)
        cos.append(torch.dot(u1, u2).item())

    cos = np.array(cos, dtype=np.float32)
    return {
        "align_mean_cos": float(cos.mean()),
        "align_std_cos": float(cos.std()),
        "align_q10": float(np.quantile(cos, 0.10)),
        "align_q90": float(np.quantile(cos, 0.90)),
    }


# ----------------------------
# Prefix-aware accuracy (fixed): uses generate(output_scores=True)
# ----------------------------

@torch.inference_mode()
def prefix_aware_from_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_ids: set,
    max_new_tokens: int,
    bf16: bool,
) -> Dict[str, float]:
    """
    Greedy generation with output_scores=True.
    For each example, find the first generated token that is NOT in ignore_ids,
    and compare to target_ids.
    Returns:
      acc_prefix_aware: accuracy over all examples
      acc_prefix_aware_on_wrong0: accuracy restricted to those wrong at step0
      strict_acc_from_scores: strict step0 accuracy computed from scores[0]
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_ids = target_ids.to(device)

    # Ensure we never ignore actual target tokens (monotonicity guarantee)
    ignore_ids = set(int(x) for x in ignore_ids)
    for tid in target_ids.unique().tolist():
        if tid in ignore_ids:
            ignore_ids.remove(tid)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (bf16 and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(model.config, "pad_token_id", None) or 0,
            eos_token_id=getattr(model.config, "eos_token_id", None),
        )

    scores = gen.scores  # list length T, each [B,V]
    if len(scores) == 0:
        return {"acc_prefix_aware": 0.0, "acc_prefix_aware_on_wrong0": float("nan"), "strict_acc_from_scores": 0.0}

    step0_pred = scores[0].float().argmax(dim=-1)  # [B]
    strict = (step0_pred == target_ids).float()
    strict_acc = strict.mean().item()

    B = target_ids.shape[0]
    first_non_ignored = torch.full((B,), -1, dtype=torch.long, device=device)

    for t, sc in enumerate(scores):
        pred_t = sc.float().argmax(dim=-1)
        mask = (first_non_ignored == -1) & (~torch.isin(pred_t, torch.tensor(list(ignore_ids), device=device)))
        first_non_ignored[mask] = pred_t[mask]

    ok_all = (first_non_ignored == target_ids).float().mean().item()

    wrong0 = (step0_pred != target_ids)
    if wrong0.any():
        ok_wrong0 = (first_non_ignored[wrong0] == target_ids[wrong0]).float().mean().item()
    else:
        ok_wrong0 = float("nan")

    # Safety: prefix-aware should not be < strict (numerically it can be equal or higher)
    # If it is lower, something is inconsistent; report it plainly.
    if ok_all + 1e-9 < strict_acc:
        print("[warn] prefix-aware < strict; check ignore_ids overlap or generation settings.")

    return {
        "acc_prefix_aware": float(ok_all),
        "acc_prefix_aware_on_wrong0": float(ok_wrong0),
        "strict_acc_from_scores": float(strict_acc),
    }


# ----------------------------
# Iterative competitor patching (fixes misinterpretation)
# ----------------------------

@torch.inference_mode()
def iterative_competitor_patch(
    model,
    h0: torch.Tensor,          # [N,d] float32 on CPU or GPU
    target_ids: torch.Tensor,  # [N]
    max_iters: int,
    eps: float,
) -> Dict[str, float]:
    """
    Iteratively patch at the LM head:
      while argmax != target:
        competitor = argmax
        minimally shift along W[target]-W[competitor] so target beats competitor by eps
    Tracks whether target becomes argmax within max_iters and how many steps/norm needed.

    This directly diagnoses "multiple competitors above target".
    """
    lm_head = model.get_output_embeddings()
    W = lm_head.weight.detach()  # [V,d]
    b = lm_head.bias.detach() if getattr(lm_head, "bias", None) is not None else None

    device = W.device
    w_dtype = W.dtype

    h = h0.detach().float().to(device)
    t = target_ids.detach().to(device)

    N = h.shape[0]
    done = torch.zeros((N,), dtype=torch.bool, device=device)
    iters_used = torch.zeros((N,), dtype=torch.long, device=device)
    delta_total = torch.zeros_like(h)

    # helper to compute logits
    def logits_from(h_in: torch.Tensor) -> torch.Tensor:
        return lm_head(h_in.to(dtype=w_dtype)).float()

    for it in range(1, max_iters + 1):
        active = ~done
        if not active.any():
            break

        logits = logits_from(h[active] + delta_total[active])  # [A,V]
        pred = logits.argmax(dim=-1)                           # [A]
        targ = t[active]

        now_done = (pred == targ)
        # mark done
        idx_active = active.nonzero(as_tuple=False).squeeze(-1)
        done[idx_active[now_done]] = True
        iters_used[idx_active[now_done]] = it

        still = ~now_done
        if not still.any():
            continue

        # competitors for remaining
        comp = pred[still]                    # [R]
        idx_r = idx_active[still]             # indices in original N
        h_r = h[idx_r] + delta_total[idx_r]   # [R,d]
        t_r = t[idx_r]

        Wt = W.index_select(0, t_r).float()
        Wc = W.index_select(0, comp).float()
        bt = b.index_select(0, t_r).float() if b is not None else 0.0
        bc = b.index_select(0, comp).float() if b is not None else 0.0

        logit_t = (Wt * h_r).sum(dim=-1) + bt
        logit_c = (Wc * h_r).sum(dim=-1) + bc
        margin = logit_t - logit_c  # <= 0 typically

        diff = (Wt - Wc)  # [R,d]
        diff_norm2 = (diff * diff).sum(dim=-1) + 1e-9
        alpha = (eps - margin) / diff_norm2
        delta = diff * alpha.unsqueeze(1)

        delta_total[idx_r] += delta

    # Final check
    logits_final = logits_from(h + delta_total)
    pred_final = logits_final.argmax(dim=-1)
    success = (pred_final == t).float().mean().item()
    mean_iters = iters_used[done].float().mean().item() if done.any() else float("nan")
    mean_delta_norm = delta_total.norm(dim=-1).float().mean().item()

    return {
        "iter_patch_success_rate": float(success),
        "iter_patch_mean_iters": float(mean_iters),
        "iter_patch_mean_delta_norm": float(mean_delta_norm),
        "iter_patch_fraction_done_within_budget": float(done.float().mean().item()),
    }


# ----------------------------
# Format token set
# ----------------------------

def format_token_ids(tokenizer, extra_ids: List[int], strings: List[str]) -> List[int]:
    ids = set(int(x) for x in extra_ids)
    for s in strings:
        tok = tokenizer.encode(s, add_special_tokens=False)
        if len(tok) == 1:
            ids.add(int(tok[0]))
    return sorted(ids)


# ----------------------------
# Reporting structs
# ----------------------------

@dataclass
class KReport:
    k: int
    acc_global_next: float
    acc_candidate_only: float
    frac_wrong_in_candidates: float
    frac_wrong_is_format: float
    margin_mean: float
    margin_std: float
    top1: float
    top2: float
    top5: float
    mean_target_rank_topk_on_wrong: float
    top_wrong: List[Tuple[str, int]]


@dataclass
class ModelReport:
    model_id: str
    prompt_style: str
    M: int
    filler_token_id: int
    filler_decoded: str
    k_reports: List[KReport]
    probe_test_acc_split: Optional[float] = None
    probe_acc_overall: Optional[float] = None
    probe_acc_on_wrong: Optional[float] = None
    probe_acc_on_right: Optional[float] = None
    align_mean_cos: Optional[float] = None
    align_std_cos: Optional[float] = None
    align_q10: Optional[float] = None
    align_q90: Optional[float] = None
    acc_prefix_aware: Optional[float] = None
    acc_prefix_aware_on_wrong0: Optional[float] = None
    strict_acc_from_scores: Optional[float] = None
    iter_patch_success_rate: Optional[float] = None
    iter_patch_mean_iters: Optional[float] = None
    iter_patch_mean_delta_norm: Optional[float] = None
    notes: Optional[str] = None


# ----------------------------
# Nullcontext helper
# ----------------------------

from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield


# ----------------------------
# Main runner per model
# ----------------------------

def run_model(args, model_id: str) -> ModelReport:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style = get_prompt_style(args.prompt_style)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer)
    left_pad(tokenizer)

    model_kwargs = dict(
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs.update(pick_dtype_kwargs(args.bf16))
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Choose candidate values
    values, value_token_ids = find_values(tokenizer, style, M=args.M, seed=args.seed, max_tries=args.max_value_tries)
    avoid = set(value_token_ids)

    filler_id = pick_filler_token_id(tokenizer, avoid, seed=args.seed)
    filler_dec = safe_decode(tokenizer, filler_id)

    # Cache suffix ids
    suf_ids = tokenizer.encode(suffix_text(style), add_special_tokens=False)
    suf_ids = add_bos_if_needed(tokenizer, suf_ids) if args.bos_on_suffix else suf_ids  # usually false

    # Precompute prompt prefix ids for each candidate value (so trials are fast)
    prefix_ids_per_value = []
    for raw in values:
        val = style.value_prefix + raw
        prefix_text = style.instruction + style.assignment_prefix + val + style.assignment_suffix
        ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        ids = add_bos_if_needed(tokenizer, ids)
        prefix_ids_per_value.append(ids)

    # sanity: all prefix lengths equal
    Ls = {len(x) for x in prefix_ids_per_value}
    if len(Ls) != 1:
        print(f"[warn] prefix token lengths vary across values: {sorted(Ls)} (still works, will pad)")

    candidate_ids_t = torch.tensor(value_token_ids, dtype=torch.long)
    id2cls = {tid: i for i, tid in enumerate(value_token_ids)}

    # format tokens set (for classification & prefix-ignore)
    fmt_ids = format_token_ids(
        tokenizer,
        extra_ids=[filler_id],
        strings=args.format_strings,
    )
    fmt_set = set(fmt_ids)

    rep = ModelReport(
        model_id=model_id,
        prompt_style=style.name,
        M=args.M,
        filler_token_id=int(filler_id),
        filler_decoded=filler_dec,
        k_reports=[],
    )

    # For probe collection
    X_probe = []
    y_probe = []
    base_correct_probe = []
    # For patch collection (wrong trials at patch_k)
    hidden_wrong = []
    target_wrong = []

    for k in args.k_values:
        # Build trials
        idxs = [random.randrange(args.M) for _ in range(args.trials_per_k)]
        target_ids = [value_token_ids[j] for j in idxs]
        target_cls = [id2cls[value_token_ids[j]] for j in idxs]

        input_ids_list = []
        for j in idxs:
            ids = prefix_ids_per_value[j] + [int(filler_id)] * int(k) + tokenizer.encode(suffix_text(style), add_special_tokens=False)
            ids = add_bos_if_needed(tokenizer, ids)
            input_ids_list.append(ids)

        input_ids, attn = pad_left(input_ids_list, pad_id=tokenizer.pad_token_id)

        # Batched forward
        B = args.trials_per_k
        preds = []
        corrects = []
        cand_corrects = []
        margins = []
        wrong_counter = Counter()
        ranks_on_wrong = []
        top1_list = []
        top2_list = []
        top5_list = []

        need_hidden = (args.do_probe and k == args.probe_k) or (args.do_iter_patch and k == args.patch_k)

        for start in tqdm(range(0, B, args.batch_size), desc=f"[{model_id}] forward@k={k}"):
            end = min(B, start + args.batch_size)
            b_ids = input_ids[start:end]
            b_attn = attn[start:end]
            b_tgt = torch.tensor(target_ids[start:end], dtype=torch.long)

            logits_last, hidden_last = forward_last_logits_and_hidden(
                model=model,
                input_ids=b_ids,
                attention_mask=b_attn,
                need_hidden=need_hidden,
                probe_layer=args.probe_layer,
                bf16=args.bf16,
            )

            device2 = logits_last.device
            cand_ids = candidate_ids_t.to(device2)

            pred = logits_last.argmax(dim=-1).detach().cpu()
            corr = pred.eq(b_tgt).bool()
            preds.append(pred)
            corrects.append(corr.cpu())

            # candidate-only
            cand_logits = logits_last.index_select(dim=1, index=cand_ids)  # [b,M]
            cand_pred = cand_ids[cand_logits.argmax(dim=-1)].detach().cpu()
            cand_corr = cand_pred.eq(b_tgt).bool()
            cand_corrects.append(cand_corr.cpu())

            # wrong token histogram & format classification
            wrong_mask = ~corr
            if wrong_mask.any():
                wrong_ids = pred[wrong_mask].tolist()
                wrong_counter.update(wrong_ids)

                # approximate target rank among topK for wrong trials
                topK = args.rank_topk
                topk_ids = torch.topk(logits_last[wrong_mask.to(device2)], k=topK, dim=-1).indices.detach().cpu()
                tgt_wrong = b_tgt[wrong_mask.cpu()]
                for r_i in range(topk_ids.shape[0]):
                    pos = (topk_ids[r_i] == tgt_wrong[r_i]).nonzero(as_tuple=False)
                    if pos.numel() == 0:
                        ranks_on_wrong.append(float(topK + 1))
                    else:
                        ranks_on_wrong.append(float(int(pos.item()) + 1))

            # margin: correct logit - best wrong (full vocab)
            top2_vals, top2_ids = torch.topk(logits_last, k=2, dim=-1)
            b_tgt_dev = b_tgt.to(device2)
            correct_logits = logits_last.gather(1, b_tgt_dev.view(-1, 1)).squeeze(1)
            best = top2_vals[:, 0]
            second = top2_vals[:, 1]
            best_id = top2_ids[:, 0]
            margin = torch.where(best_id.eq(b_tgt_dev), best - second, correct_logits - best).detach().cpu()
            margins.append(margin)

            top1_list.append(topk_contains(logits_last, b_tgt_dev, 1).detach().cpu())
            top2_list.append(topk_contains(logits_last, b_tgt_dev, 2).detach().cpu())
            top5_list.append(topk_contains(logits_last, b_tgt_dev, 5).detach().cpu())

            # collect probe/patch data if needed
            if need_hidden and hidden_last is not None:
                hidden_cpu = hidden_last.detach().cpu().float()
                if args.do_probe and k == args.probe_k:
                    X_probe.append(hidden_cpu)
                    y_probe.append(torch.tensor([target_cls[i] for i in range(start, end)], dtype=torch.long))
                    base_correct_probe.append(corr.cpu())
                if args.do_iter_patch and k == args.patch_k:
                    wrong_idx = (~corr).nonzero(as_tuple=False).squeeze(-1)
                    if wrong_idx.numel() > 0:
                        hidden_wrong.append(hidden_cpu[wrong_idx.cpu()])
                        target_wrong.append(b_tgt[wrong_idx.cpu()])

        preds = torch.cat(preds)
        corrects = torch.cat(corrects).bool()
        cand_corrects = torch.cat(cand_corrects).bool()
        margins = torch.cat(margins).float()
        top1 = torch.cat(top1_list).float().mean().item()
        top2 = torch.cat(top2_list).float().mean().item()
        top5 = torch.cat(top5_list).float().mean().item()

        acc_global = corrects.float().mean().item()
        acc_cand = cand_corrects.float().mean().item()

        wrong = ~corrects
        if wrong.any():
            wrong_preds = preds[wrong]
            frac_wrong_in_cand = torch.isin(wrong_preds, torch.tensor(value_token_ids)).float().mean().item()
            frac_wrong_is_fmt = torch.isin(wrong_preds, torch.tensor(fmt_ids)).float().mean().item()
            mean_rank_wrong = float(np.mean(ranks_on_wrong)) if len(ranks_on_wrong) > 0 else float("nan")
        else:
            frac_wrong_in_cand = float("nan")
            frac_wrong_is_fmt = float("nan")
            mean_rank_wrong = float("nan")

        top_wrong = [(repr(safe_decode(tokenizer, tid)), int(cnt)) for tid, cnt in wrong_counter.most_common(args.top_wrong)]

        rep.k_reports.append(KReport(
            k=int(k),
            acc_global_next=float(acc_global),
            acc_candidate_only=float(acc_cand),
            frac_wrong_in_candidates=float(frac_wrong_in_cand),
            frac_wrong_is_format=float(frac_wrong_is_fmt),
            margin_mean=float(margins.mean().item()),
            margin_std=float(margins.std(unbiased=False).item()),
            top1=float(top1),
            top2=float(top2),
            top5=float(top5),
            mean_target_rank_topk_on_wrong=float(mean_rank_wrong),
            top_wrong=top_wrong,
        ))

    # Probe + alignment
    if args.do_probe and X_probe:
        X = torch.cat(X_probe)
        y = torch.cat(y_probe)
        base_correct = torch.cat(base_correct_probe).bool()

        probe, probe_split_acc = train_probe(X, y, n_classes=args.M, seed=args.seed, epochs=args.probe_epochs, device=device)
        pm = probe_metrics(probe, X, y, base_correct, device=device)

        rep.probe_test_acc_split = probe_split_acc
        rep.probe_acc_overall = pm["probe_acc_overall"]
        rep.probe_acc_on_wrong = pm["probe_acc_on_wrong"]
        rep.probe_acc_on_right = pm["probe_acc_on_right"]

        am = alignment_probe_vs_lm_head(probe, model, value_token_ids, n_pairs=args.align_pairs, seed=args.seed)
        rep.align_mean_cos = am["align_mean_cos"]
        rep.align_std_cos = am["align_std_cos"]
        rep.align_q10 = am["align_q10"]
        rep.align_q90 = am["align_q90"]

    # Prefix-aware metrics (fixed)
    if args.do_prefix:
        # Evaluate on a subset at prefix_k
        k0 = args.prefix_k
        # Build a fresh subset batch at that k
        n = min(args.prefix_subset, args.trials_per_k)
        idxs = [random.randrange(args.M) for _ in range(n)]
        tgt = [value_token_ids[j] for j in idxs]

        ids_list = []
        for j in idxs:
            ids = prefix_ids_per_value[j] + [int(filler_id)] * int(k0) + tokenizer.encode(suffix_text(style), add_special_tokens=False)
            ids = add_bos_if_needed(tokenizer, ids)
            ids_list.append(ids)

        ids_t, attn_t = pad_left(ids_list, pad_id=tokenizer.pad_token_id)
        tgt_t = torch.tensor(tgt, dtype=torch.long)

        prefix_stats = prefix_aware_from_scores(
            model=model,
            input_ids=ids_t,
            attention_mask=attn_t,
            target_ids=tgt_t,
            ignore_ids=set(fmt_ids),  # ignore "format tokens" as allowable prefixes
            max_new_tokens=args.max_new_tokens,
            bf16=args.bf16,
        )
        rep.acc_prefix_aware = prefix_stats["acc_prefix_aware"]
        rep.acc_prefix_aware_on_wrong0 = prefix_stats["acc_prefix_aware_on_wrong0"]
        rep.strict_acc_from_scores = prefix_stats["strict_acc_from_scores"]

    # Iterative competitor patching (fixed)
    if args.do_iter_patch and hidden_wrong:
        hw = torch.cat(hidden_wrong)
        tw = torch.cat(target_wrong)

        # optionally subsample wrong trials to keep patching cheap
        if hw.shape[0] > args.patch_subset_wrong:
            sel = torch.randperm(hw.shape[0])[:args.patch_subset_wrong]
            hw = hw[sel]
            tw = tw[sel]

        patch_stats = iterative_competitor_patch(
            model=model,
            h0=hw,
            target_ids=tw,
            max_iters=args.patch_max_iters,
            eps=args.patch_eps,
        )
        rep.iter_patch_success_rate = patch_stats["iter_patch_success_rate"]
        rep.iter_patch_mean_iters = patch_stats["iter_patch_mean_iters"]
        rep.iter_patch_mean_delta_norm = patch_stats["iter_patch_mean_delta_norm"]

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rep


def flatten_to_rows(reports: List[ModelReport]) -> List[Dict]:
    rows = []
    for r in reports:
        if r.notes and r.notes.startswith("ERROR"):
            rows.append({"model_id": r.model_id, "notes": r.notes})
            continue
        for kr in r.k_reports:
            row = {
                "model_id": r.model_id,
                "prompt_style": r.prompt_style,
                "M": r.M,
                "filler_token_id": r.filler_token_id,
                "filler_decoded": r.filler_decoded,
                "k": kr.k,
                "acc_global_next": kr.acc_global_next,
                "acc_candidate_only": kr.acc_candidate_only,
                "frac_wrong_in_candidates": kr.frac_wrong_in_candidates,
                "frac_wrong_is_format": kr.frac_wrong_is_format,
                "margin_mean": kr.margin_mean,
                "margin_std": kr.margin_std,
                "top1": kr.top1,
                "top2": kr.top2,
                "top5": kr.top5,
                "mean_target_rank_topk_on_wrong": kr.mean_target_rank_topk_on_wrong,
                "probe_test_acc_split": r.probe_test_acc_split,
                "probe_acc_overall": r.probe_acc_overall,
                "probe_acc_on_wrong": r.probe_acc_on_wrong,
                "probe_acc_on_right": r.probe_acc_on_right,
                "align_mean_cos": r.align_mean_cos,
                "align_std_cos": r.align_std_cos,
                "align_q10": r.align_q10,
                "align_q90": r.align_q90,
                "acc_prefix_aware": r.acc_prefix_aware,
                "acc_prefix_aware_on_wrong0": r.acc_prefix_aware_on_wrong0,
                "strict_acc_from_scores": r.strict_acc_from_scores,
                "iter_patch_success_rate": r.iter_patch_success_rate,
                "iter_patch_mean_iters": r.iter_patch_mean_iters,
                "iter_patch_mean_delta_norm": r.iter_patch_mean_delta_norm,
                "notes": r.notes,
            }
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", required=True)
    ap.add_argument("--prompt_style", type=str, default="eq", help="eq | eq_trailing_space | bracket")
    ap.add_argument("--k_values", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--trials_per_k", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--probe_layer", type=int, default=-1)

    ap.add_argument("--max_value_tries", type=int, default=200000)
    ap.add_argument("--top_wrong", type=int, default=10)
    ap.add_argument("--rank_topk", type=int, default=10)

    # Probe
    ap.add_argument("--do_probe", action="store_true")
    ap.add_argument("--probe_k", type=int, default=256)
    ap.add_argument("--probe_epochs", type=int, default=30)
    ap.add_argument("--align_pairs", type=int, default=2000)

    # Prefix-aware (fixed)
    ap.add_argument("--do_prefix", action="store_true")
    ap.add_argument("--prefix_k", type=int, default=256)
    ap.add_argument("--prefix_subset", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=3)

    # Iterative competitor patching (fixed)
    ap.add_argument("--do_iter_patch", action="store_true")
    ap.add_argument("--patch_k", type=int, default=256)
    ap.add_argument("--patch_subset_wrong", type=int, default=256)
    ap.add_argument("--patch_max_iters", type=int, default=8)
    ap.add_argument("--patch_eps", type=float, default=1e-3)

    # Format tokens to classify/ignore
    ap.add_argument(
        "--format_strings",
        type=str,
        nargs="+",
        default=[" ", ' "', '"', "\n", "\t"],
        help="Any string that encodes to ONE token may be treated as a format token.",
    )

    # Loading options
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")

    # Outputs
    ap.add_argument("--out_json", type=str, default="results_v2.json")
    ap.add_argument("--out_csv", type=str, default="results_v2.csv")

    # BOS handling
    ap.add_argument("--bos_on_suffix", action="store_true", help="Normally false; keep suffix encoding BOS-free.")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    reports: List[ModelReport] = []
    for mid in args.models:
        try:
            reports.append(run_model(args, mid))
        except Exception as e:
            reports.append(ModelReport(
                model_id=mid,
                prompt_style=args.prompt_style,
                M=args.M,
                filler_token_id=-1,
                filler_decoded="",
                k_reports=[],
                notes=f"ERROR: {type(e).__name__}: {e}",
            ))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write JSON
    with open(args.out_json, "w") as f:
        json.dump([asdict(r) for r in reports], f, indent=2)

    # Write CSV
    rows = flatten_to_rows(reports)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    except Exception:
        # minimal fallback
        if rows:
            cols = list(rows[0].keys())
            with open(args.out_csv, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Print concise summary
    print("\n=== Concise per-model summary ===")
    for r in reports:
        if r.notes and r.notes.startswith("ERROR"):
            print(f"[{r.model_id}] {r.notes}")
            continue
        print(f"\n[{r.model_id}] style={r.prompt_style} filler_id={r.filler_token_id} filler_dec={repr(r.filler_decoded)}")
        for kr in r.k_reports:
            print(
                f"  k={kr.k} acc_global={kr.acc_global_next:.3f} acc_cand={kr.acc_candidate_only:.3f} "
                f"wrong_in_cand={kr.frac_wrong_in_candidates:.3f} wrong_is_fmt={kr.frac_wrong_is_format:.3f} "
                f"margin_mean={kr.margin_mean:.3f} top2={kr.top2:.3f} top5={kr.top5:.3f} "
                f"rank_top{args.rank_topk}_wrong={kr.mean_target_rank_topk_on_wrong:.2f}"
            )
            if kr.top_wrong:
                print(f"    top_wrong: {kr.top_wrong}")

        if args.do_probe:
            print(
                f"  probe_on_wrong={r.probe_acc_on_wrong} probe_overall={r.probe_acc_overall} "
                f"align_mean_cos={r.align_mean_cos}"
            )
        if args.do_prefix:
            print(
                f"  prefix(strict_from_scores)={r.strict_acc_from_scores:.3f} "
                f"prefix_aware={r.acc_prefix_aware:.3f} "
                f"prefix_aware_on_wrong0={r.acc_prefix_aware_on_wrong0}"
            )
        if args.do_iter_patch:
            print(
                f"  iter_patch_success={r.iter_patch_success_rate} "
                f"mean_iters={r.iter_patch_mean_iters} mean_delta_norm={r.iter_patch_mean_delta_norm}"
            )

    print(f"\nWrote JSON: {args.out_json}")
    print(f"Wrote CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()

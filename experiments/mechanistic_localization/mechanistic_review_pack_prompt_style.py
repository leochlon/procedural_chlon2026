# Research-quality experiment files used to generate the results for procedural_chlon2026.

#!/usr/bin/env python3
"""
mechanistic_review_pack.py

Implements the extra experiments reviewers asked for:

1) Head-level attention analysis (layer x head):
   - attention from the final query token to the source position
   - mean per head, plus correct-vs-wrong separation
   - heads whose attention-to-source changes most with distance or correlates with accuracy/margin

2) Causal interventions tied to probes:
   - train a linear probe on hidden states at the query position to predict the correct value token
   - "directional activation patching" along the probe's discriminative direction (W[y]-W[y_corrupt])
   - compare to full-vector patch and a random-direction control

3) Mechanistic analysis for the structured counting case study:
   - same head-level attention + probe-on-wrong + (optional) probe-direction interventions

Designed for Colab A100. Outputs CSVs + PNGs in --outdir.

Example:
  python mechanistic_review_pack.py \
    --model Qwen/Qwen2.5-3B \
    --distances 32,64,128,256 \
    --n_behavior 200 \
    --n_head 200 \
    --n_probe 2000 \
    --batch_size 8 \
    --patch_examples 12 \
    --patch_k_short 64 \
    --patch_k_long 256 \
    --outdir results_review_pack_qwen

Notes:
- Use --attn_impl eager for reliable attention tensors.
- If you only want the extra analyses, set --skip_behavior.
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Avoid CUDA allocator fragmentation on long runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def first_param_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

def set_tf32_safely() -> None:
    if not torch.cuda.is_available():
        return
    # Prefer new PyTorch API (still produces warnings in some builds; harmless).
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def choose_single_token(tokenizer, candidates: Sequence[str]) -> Optional[int]:
    for txt in candidates:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in set(tokenizer.all_special_ids):
            return ids[0]
    return None

def logit_margin(logits: torch.Tensor, correct_id: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, V]
    correct_id: [B]
    returns: [B] correct - best_wrong
    """
    correct_logits = logits.gather(1, correct_id.unsqueeze(1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, correct_id.unsqueeze(1), float("-inf"))
    best_wrong = tmp.max(dim=-1).values
    return correct_logits - best_wrong


# ----------------------------
# Model blocks detection (for patching)
# ----------------------------

def detect_blocks(model) -> Tuple[List[torch.nn.Module], str]:
    """
    Return (blocks, family_name) for common decoder-only architectures.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), "llama_like"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h), "gpt2_like"
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks), "mpt_like"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers), "gpt_neox_like"
    raise RuntimeError("Unsupported model architecture for activation patching.")


# ----------------------------
# Token pools and tasks
# ----------------------------

@dataclass
class TokenPool:
    value_token_ids: List[int]
    id_to_str: Dict[int, str]
    filler_token_id: int
    filler_ids: List[int]


def build_value_token_pool(tokenizer, num_values: int, max_value_len: int, prompt_style: str = "eq") -> TokenPool:
    """
    Single-token alphanumeric value pool.
    
    For 'eq' style: Prefer tokens with leading whitespace so that "KEY =" + " apple" => "KEY = apple".
    For 'bracket' style: Prefer tokens WITHOUT leading whitespace so that "KEY=[" + "apple" + "]" works.
    """
    special = set(tokenizer.all_special_ids)
    pat = re.compile(rf"^\s*[A-Za-z0-9]{{1,{max_value_len}}}$")

    value_ids: List[int] = []
    id_to_str: Dict[int, str] = {}

    if prompt_style == "bracket":
        # For bracket style: prefer NO leading whitespace
        for tid in range(tokenizer.vocab_size):
            if tid in special:
                continue
            s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if not s or not pat.match(s):
                continue
            # Prefer no leading whitespace
            if s[0].isspace():
                continue
            value_ids.append(tid)
            id_to_str[tid] = s.strip()
            if len(value_ids) >= num_values:
                break
        
        # Fallback: allow leading space if needed
        if len(value_ids) < max(16, num_values // 2):
            for tid in range(tokenizer.vocab_size):
                if tid in special or tid in set(value_ids):
                    continue
                s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                if not s or not pat.match(s):
                    continue
                value_ids.append(tid)
                id_to_str[tid] = s.strip()
                if len(value_ids) >= num_values:
                    break
    else:
        # Original 'eq' style: Prefer leading-space tokens.
        for tid in range(tokenizer.vocab_size):
            if tid in special:
                continue
            s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            if not s or not pat.match(s):
                continue
            if not (len(s) >= 2 and s[0].isspace()):
                continue
            value_ids.append(tid)
            id_to_str[tid] = s.strip()
            if len(value_ids) >= num_values:
                break

        # Fallback: allow no leading space.
        if len(value_ids) < max(16, num_values // 2):
            for tid in range(tokenizer.vocab_size):
                if tid in special or tid in set(value_ids):
                    continue
                s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                if not s or not pat.match(s):
                    continue
                value_ids.append(tid)
                id_to_str[tid] = s.strip()
                if len(value_ids) >= num_values:
                    break

    if len(value_ids) < max(16, num_values // 2):
        raise RuntimeError(f"Not enough single-token values found: {len(value_ids)}")

    filler_id = choose_single_token(tokenizer, [" the", " and", " to", " of", ".", ","])
    if filler_id is None:
        filler_id = value_ids[0]

    # Filler pool for random filler mode (single tokens only).
    filler_ids: List[int] = []
    for cand in [" the", " and", " to", " of", ".", ",", " in", " for", " with", " on"]:
        tid = choose_single_token(tokenizer, [cand])
        if tid is not None:
            filler_ids.append(tid)
    if not filler_ids:
        filler_ids = [filler_id]

    return TokenPool(
        value_token_ids=value_ids,
        id_to_str=id_to_str,
        filler_token_id=filler_id,
        filler_ids=filler_ids,
    )


@dataclass
class KeyBindingSpec:
    bos_ids: List[int]
    prefix_ids: List[int]
    suffix_ids: List[int]


def make_keybinding_spec(tokenizer) -> KeyBindingSpec:
    prefix_text = (
        "Read the following log. A variable KEY is defined exactly once.\n"
        "Return ONLY the exact value of KEY.\n"
        "KEY ="
    )
    suffix_text = (
        "\n\nNow answer the question.\n"
        "What is KEY?\n"
        "KEY ="
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return KeyBindingSpec(bos_ids=bos_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids)


def make_keybinding_spec_bracket(tokenizer) -> KeyBindingSpec:
    """
    Bracket-style prompt: KEY=[value] instead of KEY = value.
    This removes format ambiguity at the answer position.
    """
    prefix_text = (
        "Read the following log. A variable KEY is defined exactly once.\n"
        "Return ONLY the exact value of KEY.\n"
        "KEY=["
    )
    suffix_text = (
        "]\n\nNow answer the question.\n"
        "What is KEY?\n"
        "KEY=["
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return KeyBindingSpec(bos_ids=bos_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids)


def build_keybinding_ids(
    spec: KeyBindingSpec,
    pool: TokenPool,
    key_token_id: int,
    pad_len: int,
    rng: random.Random,
    filler_mode: str = "repeat",
    forced_filler_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Returns (ids, src_pos). Query token position is always last index.
    """
    ids: List[int] = []
    ids.extend(spec.bos_ids)
    ids.extend(spec.prefix_ids)
    src_pos = len(ids)
    ids.append(key_token_id)

    if forced_filler_ids is not None:
        if len(forced_filler_ids) != pad_len:
            raise ValueError("forced_filler_ids length must equal pad_len")
        ids.extend(forced_filler_ids)
    else:
        if filler_mode == "repeat":
            ids.extend([pool.filler_token_id] * pad_len)
        elif filler_mode == "random":
            ids.extend([rng.choice(pool.filler_ids) for _ in range(pad_len)])
        else:
            raise ValueError(f"Unknown filler_mode: {filler_mode}")

    ids.extend(spec.suffix_ids)
    return torch.tensor(ids, dtype=torch.long), src_pos


@dataclass
class CountingSpec:
    bos_ids: List[int]
    prefix_ids: List[int]
    step_prefix_ids: List[int]  # up to the point we inject char and count
    between_ids: List[int]      # between char and count injection
    suffix_ids: List[int]


def build_numeric_token_map(tokenizer, max_n: int) -> Dict[int, int]:
    """
    Map integer n -> token_id for single-token representation with leading space, i.e. " 17".
    """
    mapping: Dict[int, int] = {}
    for n in range(max_n + 1):
        s = f" {n}"
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in set(tokenizer.all_special_ids):
            mapping[n] = ids[0]
    return mapping


def make_counting_spec(tokenizer, target_char: str = "a") -> CountingSpec:
    # No trailing spaces where we inject single-token numeric values (" 7").
    prefix_text = (
        f"Below is a step log. The target character is '{target_char}'.\n"
        "Each step records the character and the running count of targets so far.\n"
        "Read the log and answer the final question.\n"
    )
    # We'll build steps as:
    # "\nStep {i}: ch=" + <char> + " count=" + <count_token>
    step_prefix_text = "\nStep "  # then i, then ": ch="
    between_text = ": ch="        # then <char> then " count="
    # We'll encode dynamic parts (i and char) as text to keep it readable; count injected as token.
    suffix_text = "\n\nQuestion: what is the final count?\nfinal_count ="

    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return CountingSpec(
        bos_ids=bos_ids,
        prefix_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
        step_prefix_ids=tokenizer.encode(step_prefix_text, add_special_tokens=False),
        between_ids=tokenizer.encode(between_text, add_special_tokens=False),
        suffix_ids=tokenizer.encode(suffix_text, add_special_tokens=False),
    )


def build_counting_ids(
    tokenizer,
    spec: CountingSpec,
    num_map: Dict[int, int],
    steps: int,
    rng: random.Random,
    target_char: str = "a",
    alphabet: str = "abc",
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns (ids, src_pos_final_count, label_token_id).
    Query token position is last index.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")

    # Build a random character string.
    chars = [rng.choice(alphabet) for _ in range(steps)]
    running = 0

    ids: List[int] = []
    ids.extend(spec.bos_ids)
    ids.extend(spec.prefix_ids)

    src_pos_final = -1
    final_count_token_id = -1

    for i, ch in enumerate(chars):
        # "\nStep " + str(i) + ": ch=" + ch + " count=" + <count_token>
        ids.extend(spec.step_prefix_ids)
        ids.extend(spec_prefix_i := tokenizer.encode(str(i), add_special_tokens=False))

        ids.extend(spec.between_ids)
        ids.extend(tokenizer.encode(ch, add_special_tokens=False))

        # " count=" (note leading space)
        ids.extend(tokenizer.encode(" count=", add_special_tokens=False))

        if ch == target_char:
            running += 1
        if running not in num_map:
            # If tokenizer doesn't have a single-token for this number, bail.
            raise RuntimeError(f"No single-token numeric id for {running}. Reduce steps or change model.")
        cnt_tok = num_map[running]
        pos = len(ids)
        ids.append(cnt_tok)

        if i == steps - 1:
            src_pos_final = pos
            final_count_token_id = cnt_tok

    ids.extend(spec.suffix_ids)
    if src_pos_final < 0:
        raise RuntimeError("Failed to set src_pos_final")

    return torch.tensor(ids, dtype=torch.long), src_pos_final, final_count_token_id


# ----------------------------
# Forward helpers (cache attention with full mask)
# ----------------------------

def _call_model(model, **kwargs):
    """
    Some models don't accept position_ids; try with, then retry without.
    """
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("position_ids", None)
        return model(**kwargs)


@torch.inference_mode()
def cached_step_attn_and_logits(
    model,
    input_ids: torch.Tensor,   # [B, S]
    src_pos: int,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Compute logits for the next token and attentions for the cached last-step.
    Returns:
      logits: [B, V]
      attns: tuple of length n_layers, each [B, H, 1, ctx_len]

    OOM-safe: if a batch is too large for long sequences, it automatically splits along batch dim.
    """
    device = first_param_device(model)
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [B, S]")

    def _run(ids2d: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, S = ids2d.shape
        if S < 2:
            raise ValueError("Need at least 2 tokens to do cached last-step forward")

        ctx = ids2d[:, :-1].to(device)       # [B, ctx_len]
        last_tok = ids2d[:, -1:].to(device)  # [B, 1]
        ctx_len = ctx.shape[1]

        # Masks
        ctx_mask = torch.ones((B, ctx_len), device=device, dtype=torch.long)
        full_mask = torch.ones((B, ctx_len + 1), device=device, dtype=torch.long)

        # Position ids help for RoPE-style models.
        pos_ctx = torch.arange(ctx_len, device=device).unsqueeze(0).expand(B, -1)
        pos_last = torch.full((B, 1), ctx_len, device=device, dtype=torch.long)

        out_ctx = _call_model(
            model,
            input_ids=ctx,
            attention_mask=ctx_mask,
            position_ids=pos_ctx,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        past = out_ctx.past_key_values

        out_last = _call_model(
            model,
            input_ids=last_tok,
            attention_mask=full_mask,
            position_ids=pos_last,
            past_key_values=past,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
        )

        if out_last.attentions is None:
            raise RuntimeError(
                "Model did not return attentions. Try --attn_impl eager (and avoid flash/sdpa)."
            )

        logits = out_last.logits[:, -1, :]  # [B, V]
        attns = out_last.attentions

        if src_pos >= ctx_len:
            raise RuntimeError(f"src_pos {src_pos} not in context length {ctx_len}")

        return logits, attns

    try:
        return _run(input_ids)
    except torch.cuda.OutOfMemoryError:
        if not torch.cuda.is_available():
            raise
        torch.cuda.empty_cache()
        if input_ids.shape[0] <= 1:
            raise
        mid = input_ids.shape[0] // 2
        logits1, attns1 = cached_step_attn_and_logits(model, input_ids[:mid], src_pos)
        logits2, attns2 = cached_step_attn_and_logits(model, input_ids[mid:], src_pos)
        logits = torch.cat([logits1, logits2], dim=0)
        attns = tuple(torch.cat([a1, a2], dim=0) for (a1, a2) in zip(attns1, attns2))
        return logits, attns


# ----------------------------
# Memory-safe forward helpers
# ----------------------------

def _hs_from_block_output(output):
    """Decoder block output may be Tensor or tuple where first element is hidden states."""
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def _forward_last_logits_impl(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return [B,V] logits at the final position, with attentions/hidden-states disabled."""
    device = first_param_device(model)
    input_ids = input_ids.to(device)
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    return out.logits[:, -1, :]


def forward_last_logits_safe(model, input_ids: torch.Tensor) -> torch.Tensor:
    """OOM-safe wrapper for _forward_last_logits_impl (splits along batch dim if needed)."""
    try:
        return _forward_last_logits_impl(model, input_ids)
    except torch.cuda.OutOfMemoryError:
        if not torch.cuda.is_available():
            raise
        torch.cuda.empty_cache()
        if input_ids.shape[0] <= 1:
            raise
        mid = input_ids.shape[0] // 2
        a = forward_last_logits_safe(model, input_ids[:mid])
        b = forward_last_logits_safe(model, input_ids[mid:])
        return torch.cat([a, b], dim=0)


@torch.inference_mode()
def _forward_logits_and_capture_layers_last_impl(
    model,
    input_ids: torch.Tensor,                 # [B,S]
    layers_to_capture: Sequence[int],        # which decoder blocks to capture
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Returns:
      logits_last: [B,V]
      captured: dict layer_idx -> [B,d] hidden at final token AFTER that layer

    Uses forward hooks (does NOT request output_hidden_states=True).
    """
    device = first_param_device(model)
    blocks, _ = detect_blocks(model)
    input_ids = input_ids.to(device)
    B, S = input_ids.shape
    pos_q = S - 1
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    # Register hooks
    for li in layers_to_capture:
        if li < 0 or li >= len(blocks):
            raise ValueError(f"layer index {li} out of range for {len(blocks)} layers")

        def make_hook(layer_i: int):
            def hook_fn(module, inputs, output):
                hs = _hs_from_block_output(output)
                captured[layer_i] = hs[:, pos_q, :].detach()
            return hook_fn

        handles.append(blocks[li].register_forward_hook(make_hook(li)))

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )

    for h in handles:
        h.remove()

    logits_last = out.logits[:, -1, :]
    return logits_last, captured


def forward_logits_and_capture_layers_last_safe(
    model,
    input_ids: torch.Tensor,
    layers_to_capture: Sequence[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """OOM-safe wrapper for _forward_logits_and_capture_layers_last_impl."""
    layers_list = list(layers_to_capture)
    try:
        return _forward_logits_and_capture_layers_last_impl(model, input_ids, layers_list)
    except torch.cuda.OutOfMemoryError:
        if not torch.cuda.is_available():
            raise
        torch.cuda.empty_cache()
        if input_ids.shape[0] <= 1:
            raise
        mid = input_ids.shape[0] // 2
        logits_a, cap_a = forward_logits_and_capture_layers_last_safe(model, input_ids[:mid], layers_list)
        logits_b, cap_b = forward_logits_and_capture_layers_last_safe(model, input_ids[mid:], layers_list)
        logits = torch.cat([logits_a, logits_b], dim=0)
        cap: Dict[int, torch.Tensor] = {}
        for li in layers_list:
            cap[li] = torch.cat([cap_a[li], cap_b[li]], dim=0)
        return logits, cap


# ----------------------------
# Generic evaluation loops
# ----------------------------

def run_behavior_curve(
    model,
    build_batch_fn,
    distances: List[int],
    n_samples: int,
    batch_size: int,
    seed: int,
    candidate_token_ids: Optional[List[int]] = None,
    tokenizer = None,
) -> pd.DataFrame:
    """
    Run behavioral evaluation with extended diagnostics.
    
    If candidate_token_ids is provided, also computes:
    - acc_candidate_only: accuracy when argmax restricted to candidates
    - frac_wrong_in_candidates: fraction of wrong preds that are candidates
    - top_wrong_tokens: most common wrong predictions (decoded if tokenizer provided)
    """
    device = first_param_device(model)
    rng = random.Random(seed)
    rows = []
    
    cand_set = set(candidate_token_ids) if candidate_token_ids else None
    cand_tensor = torch.tensor(candidate_token_ids, device=device) if candidate_token_ids else None
    
    for k in tqdm(distances, desc="Behavior"):
        n_done = 0
        n_correct = 0
        n_correct_cand = 0
        margins = []
        all_preds = []
        all_labels = []
        wrong_preds = []
        
        while n_done < n_samples:
            b = min(batch_size, n_samples - n_done)
            input_ids, labels, _src_pos = build_batch_fn(k, b, rng)
            labels = labels.to(device)

            logits = forward_last_logits_safe(model, input_ids)  # [B,V]
            preds = logits.argmax(dim=-1)
            n_correct += int((preds == labels).sum().item())
            margins.append(logit_margin(logits, labels).detach().float().cpu())
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # Candidate-only accuracy
            if cand_tensor is not None:
                cand_logits = logits.index_select(dim=1, index=cand_tensor)
                cand_preds = cand_tensor[cand_logits.argmax(dim=-1)]
                n_correct_cand += int((cand_preds == labels).sum().item())
            
            # Collect wrong predictions
            wrong_mask = (preds != labels)
            if wrong_mask.any():
                wrong_preds.extend(preds[wrong_mask].cpu().tolist())
            
            n_done += b
        
        acc = n_correct / max(1, n_samples)
        mean_margin = torch.cat(margins, dim=0).mean().item() if margins else float("nan")
        
        row = {
            "k": k, 
            "n": n_samples, 
            "accuracy": acc, 
            "mean_logit_margin": mean_margin
        }
        
        # Extended diagnostics
        if cand_set is not None:
            row["acc_candidate_only"] = n_correct_cand / max(1, n_samples)
            
            if wrong_preds:
                wrong_in_cand = sum(1 for p in wrong_preds if p in cand_set)
                row["frac_wrong_in_candidates"] = wrong_in_cand / len(wrong_preds)
                row["n_wrong"] = len(wrong_preds)
                
                # Top wrong tokens
                from collections import Counter
                wrong_counts = Counter(wrong_preds).most_common(10)
                if tokenizer is not None:
                    row["top_wrong_tokens"] = [
                        (tokenizer.decode([tid]), tid, cnt) 
                        for tid, cnt in wrong_counts
                    ]
                else:
                    row["top_wrong_tokens"] = wrong_counts
            else:
                row["frac_wrong_in_candidates"] = float("nan")
                row["n_wrong"] = 0
                row["top_wrong_tokens"] = []
        
        rows.append(row)
    return pd.DataFrame(rows)


def run_head_level_attention(
    model,
    build_batch_fn,
    distances: List[int],
    n_samples: int,
    batch_size: int,
    seed: int,
    outdir: str,
    tag: str,
) -> pd.DataFrame:
    """
    Produces a long DataFrame:
      k, layer, head, mean_attn, mean_attn_correct, mean_attn_wrong, diff, corr_margin
    plus top-head summaries and heatmaps.
    """
    device = first_param_device(model)
    blocks, _ = detect_blocks(model)
    n_layers = len(blocks)

    rng = random.Random(seed)
    all_rows = []

    top_rows = []

    for k in tqdm(distances, desc=f"Head attention ({tag})"):
        # Streaming stats over all samples
        sum_all = None         # [L,H]
        sum_correct = None     # [L,H]
        sum_wrong = None       # [L,H]
        sum_w = None           # [L,H]
        sum_w2 = None          # [L,H]
        sum_wm = None          # [L,H]
        sum_m = 0.0
        sum_m2 = 0.0
        n_total = 0
        n_correct = 0
        n_wrong = 0

        # Process in batches
        n_done = 0
        while n_done < n_samples:
            b = min(batch_size, n_samples - n_done)
            input_ids, labels, src_pos = build_batch_fn(k, b, rng)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, attns = cached_step_attn_and_logits(model, input_ids, src_pos)
            preds = logits.argmax(dim=-1)
            correct_mask = (preds == labels)
            wrong_mask = ~correct_mask

            margin = logit_margin(logits, labels).detach().float()  # [B]
            sum_m += float(margin.sum().item())
            sum_m2 += float((margin * margin).sum().item())

            # Initialize arrays after we see head dim
            if sum_all is None:
                n_heads = attns[0].shape[1]
                sum_all = torch.zeros((n_layers, n_heads), device=device, dtype=torch.float32)
                sum_correct = torch.zeros_like(sum_all)
                sum_wrong = torch.zeros_like(sum_all)
                sum_w = torch.zeros_like(sum_all)
                sum_w2 = torch.zeros_like(sum_all)
                sum_wm = torch.zeros_like(sum_all)

            for layer_idx in range(n_layers):
                a = attns[layer_idx].detach().float()      # [B,H,1,ctx_len]
                w = a[:, :, 0, src_pos]                    # [B,H]
                sum_all[layer_idx] += w.sum(dim=0)
                if correct_mask.any():
                    sum_correct[layer_idx] += w[correct_mask].sum(dim=0)
                if wrong_mask.any():
                    sum_wrong[layer_idx] += w[wrong_mask].sum(dim=0)
                sum_w[layer_idx] += w.sum(dim=0)
                sum_w2[layer_idx] += (w * w).sum(dim=0)
                sum_wm[layer_idx] += (w * margin.unsqueeze(1)).sum(dim=0)

            n_total += b
            n_correct += int(correct_mask.sum().item())
            n_wrong += int(wrong_mask.sum().item())
            n_done += b

        assert sum_all is not None
        n_heads = sum_all.shape[1]

        # Compute per-head stats
        mean_m = sum_m / max(1, n_total)
        var_m = (sum_m2 / max(1, n_total)) - mean_m * mean_m
        var_m = max(var_m, 1e-12)

        mean_all = (sum_all / max(1, n_total)).detach().cpu().numpy()
        mean_c = (sum_correct / max(1, n_correct)).detach().cpu().numpy() if n_correct > 0 else np.full_like(mean_all, np.nan)
        mean_wro = (sum_wrong / max(1, n_wrong)).detach().cpu().numpy() if n_wrong > 0 else np.full_like(mean_all, np.nan)

        Ew = (sum_w / max(1, n_total)).detach().cpu().numpy()
        Ew2 = (sum_w2 / max(1, n_total)).detach().cpu().numpy()
        Ewm = (sum_wm / max(1, n_total)).detach().cpu().numpy()

        # corr(w, margin) per head
        corr = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
        for li in range(n_layers):
            for hi in range(n_heads):
                vw = float(Ew2[li, hi] - Ew[li, hi] * Ew[li, hi])
                if vw <= 1e-12:
                    continue
                cov = float(Ewm[li, hi] - Ew[li, hi] * mean_m)
                corr[li, hi] = cov / math.sqrt(vw * var_m)

        # Assemble rows
        for li in range(n_layers):
            for hi in range(n_heads):
                all_rows.append({
                    "k": int(k),
                    "layer": int(li),
                    "head": int(hi),
                    "mean_attn_to_source": float(mean_all[li, hi]),
                    "mean_attn_correct": float(mean_c[li, hi]) if np.isfinite(mean_c[li, hi]) else np.nan,
                    "mean_attn_wrong": float(mean_wro[li, hi]) if np.isfinite(mean_wro[li, hi]) else np.nan,
                    "diff_correct_minus_wrong": float(mean_c[li, hi] - mean_wro[li, hi]) if (np.isfinite(mean_c[li, hi]) and np.isfinite(mean_wro[li, hi])) else np.nan,
                    "corr_with_margin": float(corr[li, hi]) if np.isfinite(corr[li, hi]) else np.nan,
                    "n_total": int(n_total),
                    "n_correct": int(n_correct),
                    "n_wrong": int(n_wrong),
                })

        # Heatmap: mean attention to source over heads/layers
        heat = mean_all  # [L,H]
        plt.figure(figsize=(10, 6))
        plt.imshow(heat, aspect="auto", origin="lower")
        plt.colorbar(label="mean attn to source")
        plt.xlabel("head")
        plt.ylabel("layer")
        plt.title(f"{tag}: mean attention-to-source (k={k})")
        plt.savefig(os.path.join(outdir, f"{tag}_head_attn_heat_k{k}.png"), bbox_inches="tight")
        plt.close()

        # Top heads by correct-wrong gap at this k
        if n_wrong >= max(10, n_samples // 10):
            flat = []
            for li in range(n_layers):
                for hi in range(n_heads):
                    d = mean_c[li, hi] - mean_wro[li, hi]
                    if np.isfinite(d):
                        flat.append((d, li, hi))
            flat.sort(reverse=True, key=lambda x: x[0])
            for rank, (d, li, hi) in enumerate(flat[:20]):
                top_rows.append({
                    "k": int(k),
                    "rank": int(rank + 1),
                    "layer": int(li),
                    "head": int(hi),
                    "diff_correct_minus_wrong": float(d),
                    "mean_correct": float(mean_c[li, hi]),
                    "mean_wrong": float(mean_wro[li, hi]),
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(outdir, f"{tag}_head_level_attention.csv"), index=False)

    if top_rows:
        df_top = pd.DataFrame(top_rows)
        df_top.to_csv(os.path.join(outdir, f"{tag}_top_heads_by_correct_wrong_gap.csv"), index=False)

    # Cross-distance correlation: does head mean track accuracy over k?
    # Compute per (layer, head): corr(mean_attn(k), acc(k))
    # We'll need accuracy curve from behavior file, so we compute it here from df.
    try:
        # mean attention per k per head
        mean_by = df.groupby(["k", "layer", "head"])["mean_attn_to_source"].mean().reset_index()
        # overall accuracy at each k from the same data (since we have n_correct/n_total in rows)
        acc_by = df.groupby("k").apply(lambda g: g["n_correct"].iloc[0] / max(1, g["n_total"].iloc[0])).reset_index(name="acc_from_head_run")
        merged = mean_by.merge(acc_by, on="k", how="left")

        rows_corr = []
        for (li, hi), g in merged.groupby(["layer", "head"]):
            if g.shape[0] < 3:
                continue
            x = g["mean_attn_to_source"].to_numpy()
            y = g["acc_from_head_run"].to_numpy()
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            rows_corr.append({"layer": int(li), "head": int(hi), "corr_mean_attn_vs_acc_over_k": r})
        if rows_corr:
            pd.DataFrame(rows_corr).to_csv(os.path.join(outdir, f"{tag}_head_corr_over_k.csv"), index=False)
    except Exception:
        pass

    return df


# ----------------------------
# Probes + probe-on-wrong + weights
# ----------------------------

@dataclass
class ProbeResult:
    layer: int
    test_acc: float
    test_acc_on_model_wrong: float
    n_test: int
    n_wrong_test: int
    class_token_ids: List[int]
    W: torch.Tensor  # [C, d] on CPU float32
    b: torch.Tensor  # [C] on CPU float32


def train_linear_probe(
    X_train: torch.Tensor,  # [N,d] float32
    y_train: torch.Tensor,  # [N] int64 in [0,C)
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> Tuple[torch.nn.Linear, float, torch.Tensor]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = int(y_train.max().item()) + 1
    d = X_train.shape[1]
    clf = torch.nn.Linear(d, C, bias=True).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = X_train.to(device)
    ytr = y_train.to(device)
    Xte = X_test.to(device)
    yte = y_test.to(device)

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = clf(Xtr)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = clf(Xte)
        pred = logits.argmax(dim=-1)
        acc = float((pred == yte).float().mean().item())
    return clf, acc, pred.detach().cpu()


@torch.inference_mode()
def collect_probe_dataset_batched(
    model,
    build_batch_fn,
    distances: List[int],
    n_samples_total: int,
    batch_size: int,
    layer_idx: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collects:
      X: [N,d] float32 hidden at query position after layer_idx
      y_token: [N] token ids (labels)
      y_model: [N] model argmax token ids

    Memory-safe: uses forward hooks to capture ONLY the needed layer at the final token,
    rather than requesting output_hidden_states=True.
    """
    device = first_param_device(model)
    rng = random.Random(seed)
    blocks, _ = detect_blocks(model)
    n_layers = len(blocks)
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"layer_idx out of range: {layer_idx}")

    # split roughly evenly across distances
    per_k = max(1, int(math.ceil(n_samples_total / max(1, len(distances)))))

    X_list = []
    y_list = []
    y_model_list = []

    n_collected = 0
    for k in distances:
        if n_collected >= n_samples_total:
            break
        need = min(per_k, n_samples_total - n_collected)
        done = 0
        while done < need:
            b = min(batch_size, need - done)
            input_ids, labels, _src_pos = build_batch_fn(k, b, rng)
            labels = labels.to(device)

            logits, cap = forward_logits_and_capture_layers_last_safe(model, input_ids, [layer_idx])
            preds = logits.argmax(dim=-1)  # [B]
            vec = cap[layer_idx].detach().float().cpu()  # [B,d] float32 on CPU

            X_list.append(vec)
            y_list.append(labels.detach().cpu())
            y_model_list.append(preds.detach().cpu())

            done += b
            n_collected += b

    X = torch.cat(X_list, dim=0) if X_list else torch.empty((0, 1), dtype=torch.float32)
    y = torch.cat(y_list, dim=0) if y_list else torch.empty((0,), dtype=torch.long)
    y_model = torch.cat(y_model_list, dim=0) if y_model_list else torch.empty((0,), dtype=torch.long)
    return X, y, y_model


def run_probe_and_probe_on_wrong(
    model,
    build_batch_fn,
    distances: List[int],
    n_probe: int,
    batch_size: int,
    layer_idx: int,
    seed: int,
    outdir: str,
    tag: str,
) -> ProbeResult:
    X, y_token, y_model = collect_probe_dataset_batched(
        model=model,
        build_batch_fn=build_batch_fn,
        distances=distances,
        n_samples_total=n_probe,
        batch_size=batch_size,
        layer_idx=layer_idx,
        seed=seed,
    )

    # Restrict to classes that appear
    class_token_ids = sorted(set(int(t) for t in y_token.tolist()))
    tok_to_class = {t: i for i, t in enumerate(class_token_ids)}
    y = torch.tensor([tok_to_class[int(t)] for t in y_token.tolist()], dtype=torch.long)

    # Train/test split
    N = X.shape[0]
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    split = int(0.75 * N)
    tr = idx[:split]
    te = idx[split:]

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    clf, acc, pred_class = train_linear_probe(Xtr, ytr, Xte, yte, seed=seed)

    # Probe-on-wrong subset (model wrong vs ground truth token)
    y_token_te = y_token[te]
    y_model_te = y_model[te]
    wrong_mask = (y_model_te != y_token_te)
    n_wrong = int(wrong_mask.sum().item())
    if n_wrong > 0:
        acc_wrong = float((pred_class[wrong_mask] == yte[wrong_mask]).float().mean().item())
    else:
        acc_wrong = float("nan")

    # Save CSV summary
    df = pd.DataFrame([{
        "tag": tag,
        "layer": layer_idx,
        "probe_test_acc": acc,
        "probe_test_acc_on_model_wrong": acc_wrong,
        "n_test": int(len(te)),
        "n_wrong_test": n_wrong,
        "num_classes": int(len(class_token_ids)),
    }])
    df.to_csv(os.path.join(outdir, f"{tag}_probe_summary_layer{layer_idx}.csv"), index=False)

    # Save weights
    W = clf.weight.detach().float().cpu().clone()
    b = clf.bias.detach().float().cpu().clone()
    torch.save({"class_token_ids": class_token_ids, "W": W, "b": b}, os.path.join(outdir, f"{tag}_probe_weights_layer{layer_idx}.pt"))

    return ProbeResult(
        layer=layer_idx,
        test_acc=float(acc),
        test_acc_on_model_wrong=float(acc_wrong),
        n_test=int(len(te)),
        n_wrong_test=int(n_wrong),
        class_token_ids=class_token_ids,
        W=W,
        b=b,
    )


# ----------------------------
# Causal interventions: patch along probe direction
# ----------------------------

def _hook_patch_hidden(
    output,
    pos_idx: int,
    new_vec: torch.Tensor,
):
    """
    output may be Tensor [B,S,d] or tuple where first element is that tensor.
    Returns modified output in same structure.
    """
    if isinstance(output, tuple):
        hs = output[0].clone()
        rest = output[1:]
    else:
        hs = output.clone()
        rest = None

    hs[:, pos_idx, :] = new_vec.to(dtype=hs.dtype)

    if rest is None:
        return hs
    return (hs,) + rest


@torch.inference_mode()
def run_probe_direction_patch_sweep(
    model,
    build_pair_fn,
    probe: ProbeResult,
    pad_len: int,
    n_examples: int,
    seed: int,
    outdir: str,
    tag: str,
    patch_every: int = 1,
) -> pd.DataFrame:
    """
    Standard causal tracing setup:
      clean input has KEY=A, corrupt input has KEY=B
      define ld = logit(A) - logit(B)
      patch corrupt activations at query position after layer L
        (a) full-vector patch: set h_L(query) := h_clean_L(query)
        (b) probe-direction patch: match only projection along d = normalize(W[A]-W[B])
        (c) random-direction control: match same projection delta along random orthogonal vector

    Outputs per layer:
      mean_restoration_full, mean_restoration_dir, mean_restoration_rand

    Memory-safe: captures only the needed per-layer vectors via hooks (no output_hidden_states=True).
    """
    device = first_param_device(model)
    blocks, _ = detect_blocks(model)
    n_layers = len(blocks)

    # Build mapping token_id -> class index
    tok_to_class = {t: i for i, t in enumerate(probe.class_token_ids)}
    W = probe.W.to(device)  # [C,d] float32

    rng = random.Random(seed)

    layers = list(range(0, n_layers, max(1, patch_every)))

    acc_full = torch.zeros(len(layers), device=device)
    acc_dir = torch.zeros(len(layers), device=device)
    acc_rand = torch.zeros(len(layers), device=device)
    counts = torch.zeros(len(layers), device=device)

    for _ in tqdm(range(n_examples), desc=f"Probe-dir patch sweep ({tag}, k={pad_len})"):
        clean_ids, corrupt_ids, tok_A, tok_B = build_pair_fn(pad_len, rng)
        clean_ids = clean_ids.to(device).unsqueeze(0)
        corrupt_ids = corrupt_ids.to(device).unsqueeze(0)

        # Need class ids for A and B
        if tok_A not in tok_to_class or tok_B not in tok_to_class:
            continue
        cA = tok_to_class[tok_A]
        cB = tok_to_class[tok_B]

        # Baselines + capture per-layer vectors at query position
        logits_clean, cap_clean = forward_logits_and_capture_layers_last_safe(model, clean_ids, layers)
        logits_corrupt, cap_corrupt = forward_logits_and_capture_layers_last_safe(model, corrupt_ids, layers)

        ld_clean = float((logits_clean[0, tok_A] - logits_clean[0, tok_B]).item())
        ld_corrupt = float((logits_corrupt[0, tok_A] - logits_corrupt[0, tok_B]).item())
        denom = ld_clean - ld_corrupt
        if abs(denom) < 1e-6:
            continue

        pos_q = clean_ids.shape[1] - 1

        # Direction in hidden space
        d = (W[cA] - W[cB]).detach()
        d = d / (d.norm() + 1e-8)
        d = d.to(device)

        attn_mask = torch.ones_like(corrupt_ids, dtype=torch.long, device=device)

        for li, layer_idx in enumerate(layers):
            h_clean = cap_clean[layer_idx][0].detach().float()
            h_corr = cap_corrupt[layer_idx][0].detach().float()

            # Full patch target
            target_full = h_clean

            # Directional patch: match projection along d
            proj_clean = float(torch.dot(h_clean, d).item())
            proj_corr = float(torch.dot(h_corr, d).item())
            delta_proj = proj_clean - proj_corr
            target_dir = (h_corr + delta_proj * d)

            # Random orthogonal control with same delta magnitude
            r = torch.randn_like(d)
            r = r - torch.dot(r, d) * d
            r = r / (r.norm() + 1e-8)
            target_rand = (h_corr + delta_proj * r)

            block = blocks[layer_idx]

            def eval_with_target(target_vec: torch.Tensor) -> float:
                def hook_fn(module, inputs, output):
                    return _hook_patch_hidden(output, pos_q, target_vec)

                handle = block.register_forward_hook(hook_fn)
                try:
                    out = model(
                        input_ids=corrupt_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                finally:
                    handle.remove()

                logits = out.logits[:, -1, :]
                ld = float((logits[0, tok_A] - logits[0, tok_B]).item())
                restoration = (ld - ld_corrupt) / denom
                return restoration

            try:
                r_full = eval_with_target(target_full)
                r_dir = eval_with_target(target_dir)
                r_rand = eval_with_target(target_rand)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception:
                continue

            acc_full[li] += r_full
            acc_dir[li] += r_dir
            acc_rand[li] += r_rand
            counts[li] += 1.0

    rows = []
    for i, layer_idx in enumerate(layers):
        c = float(counts[i].item())
        if c <= 0:
            continue
        rows.append({
            "k": int(pad_len),
            "layer": int(layer_idx),
            "n_examples_used": int(c),
            "mean_restoration_full_patch": float((acc_full[i] / c).item()),
            "mean_restoration_probe_dir_patch": float((acc_dir[i] / c).item()),
            "mean_restoration_random_dir_control": float((acc_rand[i] / c).item()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"{tag}_probe_direction_patch_sweep_k{pad_len}.csv"), index=False)

    if not df.empty:
        plt.figure()
        plt.plot(df["layer"], df["mean_restoration_full_patch"], marker="o", label="full vector patch")
        plt.plot(df["layer"], df["mean_restoration_probe_dir_patch"], marker="o", label="probe-direction patch")
        plt.plot(df["layer"], df["mean_restoration_random_dir_control"], marker="o", label="random-direction control")
        plt.xlabel("Layer (patched after)")
        plt.ylabel("Restoration (A vs B logit diff)")
        plt.title(f"{tag}: directional patch restoration (k={pad_len})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(outdir, f"{tag}_probe_direction_patch_sweep_k{pad_len}.png"), bbox_inches="tight")
        plt.close()

    return df


# ----------------------------
# Task builders for batching
# ----------------------------

def make_keybinding_builders(tokenizer, pool: TokenPool, filler_mode: str, prompt_style: str = "eq"):
    if prompt_style == "bracket":
        spec = make_keybinding_spec_bracket(tokenizer)
    else:
        spec = make_keybinding_spec(tokenizer)

    def build_batch(k: int, b: int, rng: random.Random):
        ids_list = []
        labels = []
        src_pos0 = None
        for _ in range(b):
            key = rng.choice(pool.value_token_ids)
            ids, src_pos = build_keybinding_ids(spec, pool, key, k, rng, filler_mode=filler_mode)
            ids_list.append(ids)
            labels.append(key)
            if src_pos0 is None:
                src_pos0 = src_pos
            else:
                if src_pos != src_pos0:
                    raise RuntimeError("src_pos changed across batch unexpectedly")
        input_ids = torch.stack(ids_list, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels_t, int(src_pos0)

    def build_pair(k: int, rng: random.Random):
        tok_A = rng.choice(pool.value_token_ids)
        tok_B = rng.choice([t for t in pool.value_token_ids if t != tok_A])
        # IMPORTANT: for causal tracing, keep everything identical except the KEY value.
        forced_fill = None
        if filler_mode == "random":
            forced_fill = [rng.choice(pool.filler_ids) for _ in range(k)]
        clean_ids, _ = build_keybinding_ids(spec, pool, tok_A, k, rng, filler_mode=filler_mode, forced_filler_ids=forced_fill)
        corrupt_ids, _ = build_keybinding_ids(spec, pool, tok_B, k, rng, filler_mode=filler_mode, forced_filler_ids=forced_fill)
        return clean_ids, corrupt_ids, int(tok_A), int(tok_B)

    return build_batch, build_pair


def make_counting_builders(tokenizer, steps_max: int, target_char: str = "a"):
    spec = make_counting_spec(tokenizer, target_char=target_char)
    num_map = build_numeric_token_map(tokenizer, max_n=steps_max + 5)
    if 0 not in num_map and 1 not in num_map:
        raise RuntimeError("Tokenizer has no single-token numeric forms for small integers; counting task won't work.")

    def build_batch(k: int, b: int, rng: random.Random):
        ids_list = []
        labels = []
        src_pos0 = None
        for _ in range(b):
            while True:
                try:
                    ids, src_pos, y_tok = build_counting_ids(tokenizer, spec, num_map, steps=k, rng=rng, target_char=target_char)
                    break
                except RuntimeError:
                    continue
            ids_list.append(ids)
            labels.append(y_tok)
            if src_pos0 is None:
                src_pos0 = src_pos
            else:
                if src_pos != src_pos0:
                    # should be fixed given deterministic tokenization lengths for fixed k
                    raise RuntimeError("src_pos changed across batch unexpectedly; tokenization length changed")
        input_ids = torch.stack(ids_list, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels_t, int(src_pos0)

    def build_pair(k: int, rng: random.Random):
        # Create two traces with different final count by regenerating a different random string.
        # Not minimal-corruption, but keeps the prompt internally consistent.
        while True:
            try:
                idsA, srcA, yA = build_counting_ids(tokenizer, spec, num_map, steps=k, rng=rng, target_char=target_char)
                idsB, srcB, yB = build_counting_ids(tokenizer, spec, num_map, steps=k, rng=rng, target_char=target_char)
                break
            except RuntimeError:
                continue
        if srcA != srcB:
            raise RuntimeError("src_pos mismatch between pair; unexpected")
        return idsA, idsB, int(yA), int(yB)

    return build_batch, build_pair


# ----------------------------
# Loading
# ----------------------------

def load_model_and_tokenizer(model_name: str, attn_impl: str, dtype_str: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    # Use memory-friendly defaults. We only request attentions/hidden-states explicitly where needed.
    kwargs = dict(device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    if attn_impl and attn_impl.lower() != "auto":
        kwargs["attn_implementation"] = attn_impl

    # transformers has been shifting toward `dtype`; keep compatibility
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    model.eval()

    # IMPORTANT: do NOT globally enable output_attentions (OOM on long sequences).
    try:
        model.config.output_attentions = False
    except Exception:
        pass
    try:
        model.config.output_hidden_states = False
    except Exception:
        pass

    return model, tokenizer


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results_review_pack")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--distances", type=str, default="32,64,128,256")
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--num_values", type=int, default=64)
    ap.add_argument("--max_value_len", type=int, default=4)
    ap.add_argument("--filler_mode", type=str, default="repeat", choices=["repeat", "random"])
    ap.add_argument("--prompt_style", type=str, default="eq", choices=["eq", "bracket"],
                    help="Prompt style: 'eq' uses 'KEY = value', 'bracket' uses 'KEY=[value]'")

    ap.add_argument("--n_behavior", type=int, default=200)
    ap.add_argument("--n_head", type=int, default=200)
    ap.add_argument("--n_probe", type=int, default=2000)

    ap.add_argument("--probe_layer", type=int, default=-1, help="Layer index for probe; -1 means final layer.")
    ap.add_argument("--probe_epochs", type=int, default=200)

    ap.add_argument("--patch_examples", type=int, default=12)
    ap.add_argument("--patch_k_short", type=int, default=64)
    ap.add_argument("--patch_k_long", type=int, default=256)
    ap.add_argument("--patch_every", type=int, default=1, help="Patch every N layers (speed/scale).")

    ap.add_argument("--run_tasks", type=str, default="keybinding,counting", help="Comma list: keybinding,counting")
    ap.add_argument("--skip_behavior", action="store_true")
    ap.add_argument("--skip_head", action="store_true")
    ap.add_argument("--skip_probe", action="store_true")
    ap.add_argument("--skip_causal", action="store_true")

    ap.add_argument("--attn_impl", type=str, default="eager", help="eager|sdpa|flash_attention_2|auto")
    ap.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")

    ap.add_argument("--counting_target", type=str, default="a")
    ap.add_argument("--counting_steps_max", type=int, default=512)

    args = ap.parse_args()
    ensure_dir(args.outdir)
    seed_everything(args.seed)
    set_tf32_safely()

    distances = parse_int_list(args.distances)
    tasks = set(parse_str_list(args.run_tasks))

    # Save run meta
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, args.attn_impl, args.dtype)

    # -------------------------
    # Key-binding task
    # -------------------------
    if "keybinding" in tasks:
        print("\n=== Task: keybinding ===")
        out_kb = os.path.join(args.outdir, "keybinding")
        ensure_dir(out_kb)

        pool = build_value_token_pool(tokenizer, num_values=args.num_values, max_value_len=args.max_value_len, prompt_style=args.prompt_style)
        build_batch, build_pair = make_keybinding_builders(tokenizer, pool, filler_mode=args.filler_mode, prompt_style=args.prompt_style)
        print(f"Using prompt_style: {args.prompt_style}")

        if not args.skip_behavior:
            df_beh = run_behavior_curve(
                model, build_batch, distances, args.n_behavior, args.batch_size, args.seed + 1,
                candidate_token_ids=pool.value_token_ids,
                tokenizer=tokenizer,
            )
            df_beh.to_csv(os.path.join(out_kb, "behavior.csv"), index=False)
            
            # Print diagnostic summary
            print("\n=== DIAGNOSTIC: Keybinding behavior ===")
            for _, row in df_beh.iterrows():
                print(f"k={row['k']:4d}  acc={row['accuracy']:.3f}  ", end="")
                if 'acc_candidate_only' in row:
                    print(f"acc_cand={row['acc_candidate_only']:.3f}  ", end="")
                if 'frac_wrong_in_candidates' in row and not pd.isna(row['frac_wrong_in_candidates']):
                    print(f"wrong_in_cand={row['frac_wrong_in_candidates']:.3f}  ", end="")
                if 'n_wrong' in row:
                    print(f"n_wrong={row['n_wrong']}  ", end="")
                print()
                if 'top_wrong_tokens' in row and row['top_wrong_tokens']:
                    print(f"       top_wrong: {row['top_wrong_tokens'][:5]}")
            
            plt.figure()
            plt.plot(df_beh["k"], df_beh["accuracy"], marker="o", label="global")
            if "acc_candidate_only" in df_beh.columns:
                plt.plot(df_beh["k"], df_beh["acc_candidate_only"], marker="s", label="candidate_only")
            plt.xlabel("k (filler tokens)")
            plt.ylabel("accuracy")
            plt.title(f"Keybinding accuracy vs distance ({args.model})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_kb, "behavior_accuracy.png"), bbox_inches="tight")
            plt.close()

        if not args.skip_head:
            _ = run_head_level_attention(
                model=model,
                build_batch_fn=build_batch,
                distances=distances,
                n_samples=args.n_head,
                batch_size=args.batch_size,
                seed=args.seed + 2,
                outdir=out_kb,
                tag="keybinding",
            )

        probe_layer = args.probe_layer
        if probe_layer < 0:
            blocks, _ = detect_blocks(model)
            probe_layer = len(blocks) - 1

        probe_res = None
        if not args.skip_probe:
            probe_res = run_probe_and_probe_on_wrong(
                model=model,
                build_batch_fn=build_batch,
                distances=distances,
                n_probe=args.n_probe,
                batch_size=args.batch_size,
                layer_idx=probe_layer,
                seed=args.seed + 3,
                outdir=out_kb,
                tag="keybinding",
            )
            print(f"Probe test acc: {probe_res.test_acc:.3f}, on model-wrong: {probe_res.test_acc_on_model_wrong:.3f} (n_wrong={probe_res.n_wrong_test})")

        if (not args.skip_causal) and (probe_res is not None):
            # probe-direction patch sweeps at short and long distances
            _ = run_probe_direction_patch_sweep(
                model=model,
                build_pair_fn=build_pair,
                probe=probe_res,
                pad_len=args.patch_k_short,
                n_examples=args.patch_examples,
                seed=args.seed + 4,
                outdir=out_kb,
                tag="keybinding",
                patch_every=args.patch_every,
            )
            _ = run_probe_direction_patch_sweep(
                model=model,
                build_pair_fn=build_pair,
                probe=probe_res,
                pad_len=args.patch_k_long,
                n_examples=args.patch_examples,
                seed=args.seed + 5,
                outdir=out_kb,
                tag="keybinding",
                patch_every=args.patch_every,
            )

    # -------------------------
    # Counting case study task
    # -------------------------
    if "counting" in tasks:
        print("\n=== Task: counting_case_study ===")
        out_ct = os.path.join(args.outdir, "counting")
        ensure_dir(out_ct)

        build_batch_c, build_pair_c = make_counting_builders(
            tokenizer=tokenizer,
            steps_max=args.counting_steps_max,
            target_char=args.counting_target
        )

        if not args.skip_behavior:
            df_beh = run_behavior_curve(model, build_batch_c, distances, args.n_behavior, args.batch_size, args.seed + 11)
            df_beh.to_csv(os.path.join(out_ct, "behavior.csv"), index=False)
            plt.figure()
            plt.plot(df_beh["k"], df_beh["accuracy"], marker="o")
            plt.xlabel("k (steps)")
            plt.ylabel("accuracy")
            plt.title(f"Counting-case-study accuracy vs steps ({args.model})")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_ct, "behavior_accuracy.png"), bbox_inches="tight")
            plt.close()

        if not args.skip_head:
            _ = run_head_level_attention(
                model=model,
                build_batch_fn=build_batch_c,
                distances=distances,
                n_samples=args.n_head,
                batch_size=args.batch_size,
                seed=args.seed + 12,
                outdir=out_ct,
                tag="counting",
            )

        probe_layer = args.probe_layer
        if probe_layer < 0:
            blocks, _ = detect_blocks(model)
            probe_layer = len(blocks) - 1

        probe_res_c = None
        if not args.skip_probe:
            probe_res_c = run_probe_and_probe_on_wrong(
                model=model,
                build_batch_fn=build_batch_c,
                distances=distances,
                n_probe=args.n_probe,
                batch_size=args.batch_size,
                layer_idx=probe_layer,
                seed=args.seed + 13,
                outdir=out_ct,
                tag="counting",
            )
            print(f"[counting] Probe test acc: {probe_res_c.test_acc:.3f}, on model-wrong: {probe_res_c.test_acc_on_model_wrong:.3f} (n_wrong={probe_res_c.n_wrong_test})")

        if (not args.skip_causal) and (probe_res_c is not None):
            # Optional: direction patching on the counting case study.
            _ = run_probe_direction_patch_sweep(
                model=model,
                build_pair_fn=build_pair_c,
                probe=probe_res_c,
                pad_len=args.patch_k_short,
                n_examples=max(6, args.patch_examples // 2),
                seed=args.seed + 14,
                outdir=out_ct,
                tag="counting",
                patch_every=args.patch_every,
            )
            _ = run_probe_direction_patch_sweep(
                model=model,
                build_pair_fn=build_pair_c,
                probe=probe_res_c,
                pad_len=args.patch_k_long,
                n_examples=max(6, args.patch_examples // 2),
                seed=args.seed + 15,
                outdir=out_ct,
                tag="counting",
                patch_every=args.patch_every,
            )

    print(f"\nDone. Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()

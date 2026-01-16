#!/usr/bin/env python3
# Research-quality experiment files used to generate the results for procedural_chlon2026.
"""
checkpoint_multitask_smoke.py

Generalized checkpoint smoke test across multiple task types.
Addresses critique: "Character-counting is a narrow testbed."

Tasks:
1. COUNTING: Original character-counting task
2. ARITHMETIC: Multi-step arithmetic, retrieve intermediate result
3. LIST_NTH: Find the Nth item in a generated list, report at end
4. CUMSUM: Cumulative sum, retrieve value at specific position

Each task tests the same hypothesis: emitting values at computation time
prevents retrieval failures that occur when values must be retrieved later.

Models: 12-14B instruction-tuned models that demonstrated task competence
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/Phi-3-medium-4k-instruct",
]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TASK 1: COUNTING (original task)
# =============================================================================

def counting_baseline(word: str, target: str, k: int) -> Tuple[str, int]:
    """Baseline: generate full trace, then ask for word_count."""
    n = len(word)
    true_wc = sum(1 for c in word if c == target)
    prompt = f"""Count occurrences of '{target}' in a cycling trace.

Word: {word}
Target character: {target}
Trace length: {k}

Generate a step for each position i=0 to i={k-1}.
At each step, show: i, the character (word[i mod {n}]), and the running count of '{target}'.

After ALL steps, output the final count for just the first {n} characters (positions 0 to {n-1}).

Format your answer as:
STEPS:
i=0 c=X count=N
...
ANSWER: [count at position {n-1}]

Begin:
STEPS:
"""
    return prompt, true_wc


def counting_checkpoint(word: str, target: str, k: int) -> Tuple[str, int]:
    """Checkpoint: emit answer immediately after step n-1."""
    n = len(word)
    true_wc = sum(1 for c in word if c == target)
    prompt = f"""Count occurrences of '{target}' in a cycling trace.

Word: {word}
Target character: {target}
Trace length: {k}

Generate steps for i=0 to i={k-1}.
IMPORTANT: After step i={n-1}, IMMEDIATELY output ANSWER before continuing.

Format:
STEPS:
i=0 c=X count=N
...
i={n-1} c=X count=N
ANSWER: [count at position {n-1}]
i={n} c=X count=N
...
DONE

Begin:
STEPS:
"""
    return prompt, true_wc


def parse_counting(text: str) -> Optional[int]:
    m = re.search(r"ANSWER:\s*\[?(\d+)\]?", text)
    return int(m.group(1)) if m else None


# =============================================================================
# TASK 2: ARITHMETIC (multi-step, retrieve intermediate)
# =============================================================================

def arithmetic_baseline(steps: List[Tuple[str, int]], target_step: int) -> Tuple[str, int]:
    """
    Multi-step arithmetic. Must retrieve result from step target_step at the end.
    steps: list of (operation, operand) like [('+', 5), ('*', 2), ('-', 3), ...]
    """
    # Compute running values
    values = [0]
    for op, operand in steps:
        prev = values[-1]
        if op == '+':
            values.append(prev + operand)
        elif op == '-':
            values.append(prev - operand)
        elif op == '*':
            values.append(prev * operand)
    
    true_answer = values[target_step + 1]  # +1 because values[0] is initial
    
    steps_text = "\n".join([f"Step {i}: {op} {operand}" for i, (op, operand) in enumerate(steps)])
    
    prompt = f"""Perform these arithmetic operations starting from 0.
Show the result after each step.

Operations:
{steps_text}

After ALL steps, report the result that was computed at Step {target_step}.

Format:
WORK:
After Step 0: [result]
After Step 1: [result]
...
ANSWER: [result from Step {target_step}]

Begin:
WORK:
"""
    return prompt, true_answer


def arithmetic_checkpoint(steps: List[Tuple[str, int]], target_step: int) -> Tuple[str, int]:
    """Checkpoint: emit answer immediately after target step."""
    values = [0]
    for op, operand in steps:
        prev = values[-1]
        if op == '+':
            values.append(prev + operand)
        elif op == '-':
            values.append(prev - operand)
        elif op == '*':
            values.append(prev * operand)
    
    true_answer = values[target_step + 1]
    
    steps_text = "\n".join([f"Step {i}: {op} {operand}" for i, (op, operand) in enumerate(steps)])
    
    prompt = f"""Perform these arithmetic operations starting from 0.
Show the result after each step.

Operations:
{steps_text}

IMPORTANT: Immediately after showing the result for Step {target_step}, output ANSWER with that result.
Then continue with remaining steps.

Format:
WORK:
After Step 0: [result]
...
After Step {target_step}: [result]
ANSWER: [that result]
After Step {target_step + 1}: [result]
...
DONE

Begin:
WORK:
"""
    return prompt, true_answer


def parse_arithmetic(text: str) -> Optional[int]:
    m = re.search(r"ANSWER:\s*\[?(-?\d+)\]?", text)
    return int(m.group(1)) if m else None


# =============================================================================
# TASK 3: LIST_NTH (find Nth item, report at end)
# =============================================================================

def list_nth_baseline(items: List[str], target_idx: int, num_distractor_ops: int) -> Tuple[str, str]:
    """
    Generate a list, then do distractor operations, then report item at target_idx.
    """
    true_answer = items[target_idx]
    
    items_text = "\n".join([f"{i}: {item}" for i, item in enumerate(items)])
    
    # Distractor operations (counting vowels in random items, etc.)
    distractors = []
    for i in range(num_distractor_ops):
        idx = (target_idx + i + 1) % len(items)
        distractors.append(f"Operation {i}: Count letters in item {idx}")
    distractor_text = "\n".join(distractors)
    
    prompt = f"""Here is a list of items:

{items_text}

Now perform these operations:
{distractor_text}

After ALL operations, report the item at index {target_idx}.

Format:
OPERATIONS:
Operation 0: item X has Y letters
...
ANSWER: [item at index {target_idx}]

Begin:
OPERATIONS:
"""
    return prompt, true_answer


def list_nth_checkpoint(items: List[str], target_idx: int, num_distractor_ops: int) -> Tuple[str, str]:
    """Checkpoint: emit answer immediately after reading the list."""
    true_answer = items[target_idx]
    
    items_text = "\n".join([f"{i}: {item}" for i, item in enumerate(items)])
    
    distractors = []
    for i in range(num_distractor_ops):
        idx = (target_idx + i + 1) % len(items)
        distractors.append(f"Operation {i}: Count letters in item {idx}")
    distractor_text = "\n".join(distractors)
    
    prompt = f"""Here is a list of items:

{items_text}

IMPORTANT: Before doing any operations, first output ANSWER with the item at index {target_idx}.
Then perform these operations:
{distractor_text}

Format:
ANSWER: [item at index {target_idx}]
OPERATIONS:
Operation 0: item X has Y letters
...
DONE

Begin:
"""
    return prompt, true_answer


def parse_list_nth(text: str) -> Optional[str]:
    m = re.search(r"ANSWER:\s*\[?([a-zA-Z]+)\]?", text)
    return m.group(1).lower() if m else None


# =============================================================================
# TASK 4: CUMSUM (cumulative sum, retrieve at position)
# =============================================================================

def cumsum_baseline(numbers: List[int], target_idx: int, total_len: int) -> Tuple[str, int]:
    """
    Compute cumulative sums, continue past target, then report sum at target_idx.
    """
    cumsums = []
    total = 0
    for n in numbers:
        total += n
        cumsums.append(total)
    
    true_answer = cumsums[target_idx]
    
    numbers_text = ", ".join(map(str, numbers))
    
    prompt = f"""Compute the cumulative sum of these {len(numbers)} numbers:
{numbers_text}

Show the running total after adding each number.
After ALL numbers, report the cumulative sum after position {target_idx} (0-indexed).

Format:
CUMSUM:
After position 0: [sum]
After position 1: [sum]
...
ANSWER: [cumsum at position {target_idx}]

Begin:
CUMSUM:
"""
    return prompt, true_answer


def cumsum_checkpoint(numbers: List[int], target_idx: int, total_len: int) -> Tuple[str, int]:
    """Checkpoint: emit answer immediately after reaching target position."""
    cumsums = []
    total = 0
    for n in numbers:
        total += n
        cumsums.append(total)
    
    true_answer = cumsums[target_idx]
    
    numbers_text = ", ".join(map(str, numbers))
    
    prompt = f"""Compute the cumulative sum of these {len(numbers)} numbers:
{numbers_text}

Show the running total after adding each number.
IMPORTANT: After position {target_idx}, immediately output ANSWER, then continue.

Format:
CUMSUM:
After position 0: [sum]
...
After position {target_idx}: [sum]
ANSWER: [that sum]
After position {target_idx + 1}: [sum]
...
DONE

Begin:
CUMSUM:
"""
    return prompt, true_answer


def parse_cumsum(text: str) -> Optional[int]:
    m = re.search(r"ANSWER:\s*\[?(-?\d+)\]?", text)
    return int(m.group(1)) if m else None


# =============================================================================
# Trial generation
# =============================================================================

def generate_trials(task: str, num_items: int, difficulty: str, seed: int) -> List[Dict]:
    """Generate trials for a given task and difficulty level."""
    rng = random.Random(seed)
    trials = []
    
    # Difficulty controls distance between computation and retrieval
    if difficulty == "easy":
        k_or_steps = 10
    elif difficulty == "medium":
        k_or_steps = 20
    elif difficulty == "hard":
        k_or_steps = 40
    else:
        k_or_steps = 20
    
    for item_id in range(num_items):
        item_seed = seed * 1000 + item_id
        item_rng = random.Random(item_seed)
        
        if task == "counting":
            word = "".join(item_rng.choices("abcdefghij", k=10))
            target = item_rng.choice(word)
            
            baseline_prompt, true_answer = counting_baseline(word, target, k_or_steps)
            checkpoint_prompt, _ = counting_checkpoint(word, target, k_or_steps)
            parse_fn = "counting"
            
        elif task == "arithmetic":
            ops = [(item_rng.choice(['+', '-', '*']), item_rng.randint(1, 9)) 
                   for _ in range(k_or_steps)]
            # Avoid multiplication blowup
            ops = [('+' if abs(eval_ops(ops[:i+1])) > 1000 else op, n) for i, (op, n) in enumerate(ops)]
            target_step = min(4, k_or_steps - 5)  # Early step
            
            baseline_prompt, true_answer = arithmetic_baseline(ops, target_step)
            checkpoint_prompt, _ = arithmetic_checkpoint(ops, target_step)
            parse_fn = "arithmetic"
            
        elif task == "list_nth":
            words = ["apple", "banana", "cherry", "date", "elder", "fig", "grape", 
                     "honey", "iris", "jade", "kiwi", "lemon", "mango", "nectar",
                     "olive", "peach", "quince", "rose", "sage", "thyme"]
            items = item_rng.sample(words, min(10, len(words)))
            target_idx = 2  # Early position
            num_ops = k_or_steps - len(items)
            
            baseline_prompt, true_answer = list_nth_baseline(items, target_idx, max(5, num_ops))
            checkpoint_prompt, _ = list_nth_checkpoint(items, target_idx, max(5, num_ops))
            parse_fn = "list_nth"
            
        elif task == "cumsum":
            numbers = [item_rng.randint(1, 20) for _ in range(k_or_steps)]
            target_idx = 4  # Early position
            
            baseline_prompt, true_answer = cumsum_baseline(numbers, target_idx, k_or_steps)
            checkpoint_prompt, _ = cumsum_checkpoint(numbers, target_idx, k_or_steps)
            parse_fn = "cumsum"
        
        trials.append({
            "task": task,
            "difficulty": difficulty,
            "item_id": item_id,
            "baseline_prompt": baseline_prompt,
            "checkpoint_prompt": checkpoint_prompt,
            "true_answer": true_answer,
            "parse_fn": parse_fn,
        })
    
    return trials


def eval_ops(ops):
    """Helper to evaluate arithmetic operations."""
    val = 0
    for op, n in ops:
        if op == '+':
            val += n
        elif op == '-':
            val -= n
        elif op == '*':
            val *= n
    return val


# =============================================================================
# Main runner
# =============================================================================

def run_model(
    model_id: str,
    tasks: List[str],
    difficulties: List[str],
    num_items: int,
    seed: int,
    max_new_tokens: int,
    batch_size: int,
    out_dir: Path,
) -> Path:
    """Run all tasks for one model."""
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # Generate all trials
    all_trials = []
    for task in tasks:
        for diff in difficulties:
            trials = generate_trials(task, num_items, diff, seed)
            all_trials.extend(trials)
    
    # Flatten into baseline and checkpoint runs
    runs = []
    for trial in all_trials:
        for protocol in ["baseline", "checkpoint"]:
            prompt = trial["baseline_prompt"] if protocol == "baseline" else trial["checkpoint_prompt"]
            
            # Apply chat template
            if hasattr(tok, "apply_chat_template") and tok.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt
            
            runs.append({
                **trial,
                "protocol": protocol,
                "formatted_prompt": formatted,
            })
    
    results = []
    total = len(runs)
    t0 = time.time()
    
    parse_fns = {
        "counting": parse_counting,
        "arithmetic": parse_arithmetic,
        "list_nth": parse_list_nth,
        "cumsum": parse_cumsum,
    }
    
    for batch_start in range(0, total, batch_size):
        batch = runs[batch_start:batch_start + batch_size]
        prompts = [r["formatted_prompt"] for r in batch]
        
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        
        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        
        input_len = input_ids.shape[1]
        for i, run in enumerate(batch):
            new_tokens = gen[i, input_len:]
            completion = tok.decode(new_tokens, skip_special_tokens=True)
            
            parse_fn = parse_fns[run["parse_fn"]]
            pred = parse_fn(completion)
            
            # Handle string vs int comparison
            true_ans = run["true_answer"]
            if isinstance(true_ans, str):
                correct = pred is not None and pred.lower() == true_ans.lower()
            else:
                correct = pred == true_ans
            
            results.append({
                "model_id": model_id,
                "task": run["task"],
                "difficulty": run["difficulty"],
                "protocol": run["protocol"],
                "item_id": run["item_id"],
                "true_answer": true_ans,
                "pred_answer": pred,
                "correct": correct,
                "completion": completion[:500],
            })
        
        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  Progress: {done}/{total} ({100*done/total:.0f}%) - {rate:.1f}/s - ETA: {eta/60:.1f}m")
    
    # Cleanup
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_id.replace("/", "__")
    out_path = out_dir / f"{slug}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_id}")
    print(f"{'='*70}")
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    summary = df.groupby(["task", "difficulty", "protocol"]).agg(
        n=("correct", "count"),
        correct=("correct", "sum"),
    ).reset_index()
    summary["accuracy"] = (100 * summary["correct"] / summary["n"]).round(1)
    
    # Pivot for readability
    for task in df["task"].unique():
        print(f"\n{task.upper()}:")
        task_df = summary[summary["task"] == task]
        pivot = task_df.pivot(index="difficulty", columns="protocol", values="accuracy")
        pivot["delta"] = pivot["checkpoint"] - pivot["baseline"]
        print(pivot.to_string())
    
    # Save summary
    summary.to_csv(out_dir / f"{slug}.summary.csv", index=False)
    
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tasks", type=str, default="counting,arithmetic,cumsum,list_nth")
    ap.add_argument("--difficulties", type=str, default="easy,medium,hard")
    ap.add_argument("--num_items", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="results_multitask_checkpoint")
    args = ap.parse_args()
    
    set_seed(args.seed)
    
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Difficulties: {difficulties}")
    print(f"Items per condition: {args.num_items}")
    print(f"Total runs per model: {args.num_items * len(tasks) * len(difficulties) * 2}")
    
    for model_id in models:
        try:
            run_model(
                model_id=model_id,
                tasks=tasks,
                difficulties=difficulties,
                num_items=args.num_items,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"FAILED: {model_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

# 08 · Evaluation (`evaluate.py`)

After training produces a checkpoint, `training/evaluate.py` measures
how well the fine-tuned model evolves held-out seed factors.

## Metrics

Matching the paper's Table 2:

| Metric | Definition |
|---|---|
| **VR** (Valid Ratio) | Across all tool calls in all trajectories, the fraction that returned `success=True` (i.e. the expression parsed and backtested without error). Averaged per-seed, then averaged across seeds. |
| **Pass@T** | Fraction of seeds for which, within the first `T` turns, the model proposes **at least one** factor with `IR > max(0, seed_ir)`. Computed cumulatively by turn. Default `T=3`. |
| **Beat rate** | Overall pass rate (any turn) — equivalent to `Pass@max_turns`. |
| **Mean IR improvement** | Mean of `(best_ir - max(0, seed_ir))` over the seeds that were beaten. |

The paper reports `VR ≈ 0.96`, `Pass@3 ≈ 0.90` on the 99-seed test set.

## Inputs

- **Base model**: `--base-model` (default
  `/home/dc_analyst/models/Qwen3-4B-Thinking-2507`).
- **Checkpoint**: `--checkpoint` — path to the LoRA adapter directory
  (output of training). Omit to evaluate the **base** model as a
  baseline.
- **Data**: `--data` — parquet file with the same schema as training
  data (`prompt`, `seed_expr`, `seed_ir`, `seed_name`).

Because the evaluator uses multi-turn tool calling against the real
backtest API, the backtest server must be running (`bash start_api.sh`
for local, or `api_server_verl.py` for the v2 track).

## Running

```bash path=null start=null
# Evaluate the base model (no fine-tuning) on the test set
python training/evaluate.py \
    --data data/test.parquet \
    --label base

# Evaluate a trained LoRA checkpoint
python training/evaluate.py \
    --data data/test.parquet \
    --checkpoint outputs/grpo_1710000000/checkpoint-80 \
    --label step80

# Custom turn budget and generation length
python training/evaluate.py \
    --data data/val.parquet \
    --checkpoint outputs/grpo_.../checkpoint-80 \
    --max-turns 5 \
    --max-new-tokens 8000
```

## What `evaluate.py` does

1. **Loads** the base model in `bf16` and optionally merges a LoRA
   adapter with `PeftModel.from_pretrained(...).merge_and_unload()`.
2. Handles the same Thinking-template swap as `train.py` so the chat
   template is consistent.
3. For each seed row in the parquet:
   a. Applies the chat template with the `evaluate_factor` tool
      schema (built by `build_tool_schema()`).
   b. Generates up to `max_new_tokens` tokens with temperature=0.7,
      top_p=0.9.
   c. Parses `<tool_call>` blocks out of the response with
      `parse_tool_calls`.
   d. Executes each tool call via `FactorTool.evaluate` (real
      backtest), appends `tool_call` + `tool_response` to the
      conversation, and regenerates until the model stops calling
      tools or hits `max_turns`.
4. Collects all `(success, ir, factor_expr, turn)` tuples.
5. Computes per-seed metrics (best IR, beaten?, Pass@t).
6. Aggregates to `{vr, pass_at_3, beat_rate, mean_ir_improvement,
   n_beaten}` and saves
   `eval_results/eval_<label>_<dataset>.json`.

## Output

Stdout prints a summary table:

```
RESULTS: step80
============================================================
  Seeds:              99
  Valid Ratio (VR):   0.9634
  Pass@3:             0.8788
  Beat Rate (overall):0.9192
  Seeds beaten:       91/99
  Mean IR improvement:0.0842
  Total time:         1823.1s (18.4s/seed)

Per-seed results:
  Seed                 Seed IR  Best IR  Valid  Beat
  -------------------- -------- -------- ------ -----
  LIQ_NET_FLOW_PER_R...   1.2300   1.3120    3/4   YES
  ...
```

A JSON file with the full per-seed details lands in `eval_results/`.

## Interpreting results

- **VR low** (<0.8): the model is generating invalid expressions.
  Check `system_prompt.md` — the operator list might be out of sync
  with `function_lib.py`, or the chat template might be stripping
  content.
- **Pass@3 low** but VR high: the model produces valid expressions
  that don't actually beat the seed. Either the seed IR is too high
  (check `data/seed_backtest_results.json`) or training has
  under-fit. Try a later checkpoint.
- **Best IR at step 150 < best IR at step 80**: matches the paper's
  over-fitting observation; use step 80.
- **All tool calls return `success=False`**: the backtest API is
  probably down or using wrong `daily_pv.h5`. Check the API logs.

## What is NOT evaluated

- Turnover, capacity, out-of-sample performance beyond the test
  period, regime-specific robustness, transaction-cost sensitivity
  — none of these are part of the "evolution" metric. They would
  require a separate offline analysis on the chosen alpha factors.
- The Verl track ships its own evaluation harness inside the paper's
  Verl fork (`search_multiturn_grpo.yaml` + `test.parquet`); this
  `evaluate.py` is for the local `trl` track's LoRA checkpoints.

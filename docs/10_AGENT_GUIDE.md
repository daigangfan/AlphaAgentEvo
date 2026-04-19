# 10 · AI Agent Guide — How to Modify and Run AlphaAgentEvo

This document is the **operational playbook** for an AI agent (or any
engineer) who has just been dropped into this repository and needs to
make changes safely. It assumes you have skimmed the other docs in
`docs/` — it references them heavily.

> Golden rule: everything in this project is a **pipeline**. Market data
> → DSL expression → backtest → IR → reward → GRPO update. Every change
> must preserve the contracts between adjacent stages.

## 0 · Project identity & absolute paths

- Repo root: `F:\projects\AlphaAgentEvo` (Windows) or
  `/workspace/AlphaAgentEvo` (Runpod).
- Key files you'll most often edit:
  - `expression_manager/function_lib.py` — add / change operators.
  - `training/system_prompt.md` — what the agent "knows".
  - `training/factor_tool.py` / `deploy/v2/factor_tool.py` — reward
    logic.
  - `configs/grpo_config.yaml` — local training hyperparameters.
  - `deploy/v2/train.sh` — Verl training hyperparameters (Hydra
    overrides).
- Do NOT edit (unless you really mean it):
  - `backtest/data/daily_pv.h5` — source of truth.
  - `data/*.parquet` — regenerate via `training/generate_dataset.py`.
  - `.venv/`, `.python-version`, `uv.lock` — managed by `uv`.

## 1 · The mental model in 60 seconds

1. `Qwen3-4B` is asked to evolve a **seed factor** (a DSL string +
   baseline IR).
2. It emits `<tool_call>{"name":"evaluate_factor", ...}</tool_call>`.
3. The rollout engine parses that, calls the **backtest API**, which
   runs `execute_expression` → returns an IR.
4. The IR is injected back into the conversation as
   `<tool_response>{...}</tool_response>`.
5. The LLM sees the response and either proposes a better factor or
   stops.
6. At the end of the trajectory, a **5-dim hierarchical reward** is
   computed (paper Eq.5) and used to update the LLM via **GRPO**.

Full details of each stage live in the numbered docs:
[data](./02_data.md), [DSL](./03_expression_language.md),
[operators](./04_operators.md), [backtest](./05_backtest.md),
[reward](./06_reward.md), [training](./07_training.md).

## 2 · Minimum viable loop to run locally

End-to-end on Windows + RTX 5090 (approximate timings):

```bash path=null start=null
# (a) Deps (one-time, ~3 min)
uv sync

# (b) Start the backtest API (Terminal 1)
bash start_api.sh                         # port 8001, blocks

# (c) Health-check (Terminal 2)
curl -s http://localhost:8001/health
# → {"status":"ok","service":"alphaagentevo-backtest"}

# (d) Quick factor sanity test
curl -s -X POST http://localhost:8001/evaluate_factor \
    -H "Content-Type: application/json" \
    -d '{"factor_expr":"RANK(TS_PCTCHANGE($close, 20))","period":"train"}'

# (e) Smoke-test training (5 steps, ~5 min)
bash start_training.sh --smoke-test

# (f) Full training (150 steps, ~5 hours on RTX 5090)
bash start_training.sh

# (g) Evaluate the resulting checkpoint
python training/evaluate.py \
    --data data/test.parquet \
    --checkpoint outputs/grpo_<ts>/checkpoint-80 \
    --label step80
```

If step (e) completes without errors and prints non-zero rewards, the
full pipeline is healthy. Move on to (f) only then.

## 3 · Where to make common changes

### 3.1 "I want the model to know about a new operator"

1. Implement the operator in `expression_manager/function_lib.py`.
   See [`04_operators.md`](./04_operators.md) for conventions.
2. Add its description to the "Available Operators" section of
   `training/system_prompt.md`. **The LLM will not use operators it
   doesn't see.**
3. (Optional) Regenerate the training parquet if you want augmented
   seeds using your new operator:
   `python training/generate_dataset.py --augment`.
4. Sanity check:
   ```bash path=null start=null
   python -c "
   from backtest.factor_executor import execute_expression, load_data
   load_data()
   print(execute_expression('MY_OP($close, 10)', period='train'))
   "
   ```

### 3.2 "I want to change the reward formula"

1. Edit `training/factor_tool.py::calc_reward` (+ the `_calc_*`
   helpers). Keep the paper Eq.5 shape unless you really mean to
   diverge — see [`06_reward.md`](./06_reward.md).
2. Mirror the change in `deploy/v2/factor_tool.py::calc_reward` for
   the Verl track.
3. Run the built-in test: `python training/factor_tool.py`. Ensure
   the printed reward looks sensible (typically 0.02 – 0.6).
4. Consider whether the reward scale change requires a different
   `beta` (KL coefficient) in `configs/grpo_config.yaml`.

### 3.3 "I want to train on different data / market"

The minimum set of changes:

1. Produce a new `backtest/data/daily_pv.h5` with the same MultiIndex
   `(datetime, instrument)` and the same 12 columns (see
   [`02_data.md`](./02_data.md)). If you drop / add columns, update
   `factor_executor.py` and the system prompt's "Available Variables"
   list.
2. Produce new seed factors: either a
   `data/seed_factors.json` (legacy format) or
   `data/vn_seeds_300.jsonl` (paper format).
3. Update `configs/grpo_config.yaml::backtest.periods` to the new
   date ranges (train/val/test splits).
4. Regenerate parquet: `python training/generate_dataset.py --augment`.
5. Retrain with `bash start_training.sh`.

### 3.4 "I want to change the base LLM"

1. Point `configs/grpo_config.yaml::model.name` at the new model
   (local path or HF id).
2. If it's a thinking/reasoning variant, check the chat-template
   swap logic in `training/train.py::main` (around line 343); you
   may need a different base-template source.
3. Watch the VRAM budget: Qwen3-4B + LoRA + KV cache + grads ≈ 18-20
   GB on 32 GB RTX 5090. A 7B model won't fit in LoRA; consider the
   Verl track.
4. For Verl, edit `MODEL` in `deploy/v2/train.sh`.

### 3.5 "I want to add / rebuild the training dataset"

See [`02_data.md`](./02_data.md) §"Regenerating the dataset". The
important lever is `--augment` (adds programmatic factors + mutations
= ~300 rows for the trl track).

### 3.6 "I want to change backtest semantics"

`backtest/qlib_backtester.py::compute_portfolio_ir` is the only place
where portfolio construction lives. Be careful — the defaults
(`top_k=10`, `n_drop=2`, `hold_thresh=2`, `cost_buy=0.0013`,
`rebalance_freq=5`) are **paper-matched**. Change them only if you
accept that your IR numbers will no longer be directly comparable to
the paper.

## 4 · Common failure modes (and where to look)

| Symptom | Likely cause | Where to look |
|---|---|---|
| `pyarrow.lib.ArrowInvalid: Repetition level histogram size mismatch` on parquet read | Parquet was written by an older pyarrow | Regenerate `data/*.parquet` (see [`02_data.md`](./02_data.md)) |
| `NameError: name 'ZIGZAG_TOP' is not defined` during backtest | Seed uses an unimplemented operator | Either implement it in `function_lib.py` or drop that seed |
| Rewards always 0 | No successful tool calls | Check backtest API log, check `system_prompt.md` has correct tool schema, check `factor_tool.py` URL is correct |
| API responds but `"success": false` always | `daily_pv.h5` missing / corrupted or operator error | `curl -v ...` and inspect the `error` field |
| Chat template crash during training | Using Thinking variant without swapping template | See `train.py:343` — the repo already does this swap |
| OOM on RTX 5090 | Too large `max_new_tokens` or `gradient_accumulation_steps` | Lower `max_new_tokens` to 3072 (already the default); make sure `bf16: true` |
| OOM on H100 (Verl) | `gpu_memory_utilization` too high | Drop from 0.6 to 0.5 in `deploy/v2/train.sh` |
| Verl import error (`verl` not on path) | env not activated | `conda activate verl041` before every shell |
| `ContinueGenerationReqInput` missing | sglang version drift | Pin `sglang==0.4.6.post5` |
| `libnuma.so.1: cannot open shared object` | Missing system package (pod restarted) | `apt-get install -y libnuma-dev` (already in `train.sh`) |

## 5 · Testing checklist before a long run

Before kicking off an expensive 150-step run, verify each layer:

1. **DSL parser**
   ```bash path=null start=null
   python -c "from expression_manager.expr_parser import parse_expression; print(parse_expression('RANK(DELTA(\$close,1))'))"
   ```
2. **AST parser / similarity**
   ```bash path=null start=null
   python expression_manager/factor_ast.py   # runs the __main__ smoke
   ```
3. **Backtest executor**
   ```bash path=null start=null
   python backtest/factor_executor.py        # runs 3 built-in test exprs
   ```
4. **API layer**
   ```bash path=null start=null
   bash start_api.sh &
   sleep 5
   curl -s http://localhost:8001/health
   ```
5. **Reward function**
   ```bash path=null start=null
   python training/factor_tool.py            # runs reward unit test
   ```
6. **Dataset**
   ```bash path=null start=null
   python -c "import pandas as pd; df = pd.read_parquet('data/train.parquet'); print(df.shape, df.columns.tolist())"
   ```
7. **Model load + 1 generation** (smoke test)
   ```bash path=null start=null
   bash start_training.sh --smoke-test
   ```
8. **Evaluator**
   ```bash path=null start=null
   python training/evaluate.py --data data/val.parquet --label base
   ```

All six of these should pass before you commit to a 150-step run.

## 6 · File-level contract summary

When modifying one file, these are the files that also care about
your change:

| If you change… | Also touch / check… |
|---|---|
| `function_lib.py` | `training/system_prompt.md`, any test expressions in `factor_executor.py`'s `__main__` |
| `expr_parser.py` | `factor_executor.py` (interacts via `parse_expression` + `parse_symbol`) |
| `factor_ast.py` | `training/factor_tool.py::_ast_similarity` |
| `qlib_backtester.py` | `factor_executor.py` (calls it); note IR numbers break paper-parity |
| `factor_executor.py` | Both API servers (`api_server.py`, `api_server_verl.py`) |
| `api_server.py` | `training/factor_tool.py::FactorTool.evaluate` (URL / payload) |
| `api_server_verl.py` | `deploy/v2/factor_tool.py::_call_backtest_api` |
| `factor_tool.py` (local) | `training/train.py::reward_func`, `training/evaluate.py` |
| `factor_tool.py` (Verl) | `deploy/v2/factor_tool_config.yaml` (tool registration) |
| `system_prompt.md` | `training/generate_dataset.py::load_system_prompt()` (rebuild parquet) |
| `train.py` | `configs/grpo_config.yaml`, `start_training.sh` |
| `train.sh` (Verl) | `factor_tool_config.yaml`, `factor_reward.py` |
| `configs/grpo_config.yaml` | Both API servers read `backtest.periods` from here |
| `data/*.parquet` | Re-run `evaluate.py` afterwards to confirm compatibility |

## 7 · Communication contracts (the invariants)

These are the small, rigid contracts that glue the system together.
**Breaking them silently breaks training.**

### 7.1 Backtest API responses

`training/factor_tool.py::FactorTool.evaluate` expects the local
(`/evaluate_factor`) API to return a JSON object with at least:

```json path=null start=null
{"success": bool, "ir": float, "ic_mean": float, "error": str | null}
```

The Verl `/backtest` endpoint returns:

```json path=null start=null
{"data": {"success": bool, "metrics": {"Information_Ratio_with_cost": float, ...}}}
```

Changing these wrappers requires updating the consumer in the same commit.

### 7.2 Parquet columns

`trl.GRPOTrainer` passes every parquet column as `**kwargs` to the
reward function. We use this for `seed_expr` and `seed_ir`. Keep
these names spelled exactly — changing them silently zeroes out the
reward.

### 7.3 Tool schema

The `evaluate_factor` tool **must** have parameters named exactly
`factor_name` and `factor_expr` across:

- `training/train.py::evaluate_factor` (docstring + signature)
- `training/evaluate.py::build_tool_schema`
- `training/tool_config.yaml`
- `deploy/v2/factor_tool_config.yaml`

If you rename one, the model will emit tool calls with one set of
names while the rollout engine validates against another — resulting
in zero-reward trajectories with no obvious error.

### 7.4 System prompt ↔ function_lib ↔ seed operators

`system_prompt.md` enumerates every operator the LLM may use. Keep
it a **superset** of what seeds actually use, and a **subset** of
what `function_lib.py` implements. If a seed uses `BARSLAST` but the
prompt omits it, the model will avoid it (low VR). If the prompt
mentions `MY_OP` but it's not in `function_lib.py`, the model will
generate it and every call will `success=False`.

## 8 · Glossary

| Term | Meaning |
|---|---|
| **Alpha factor** | Cross-sectional ranker whose rank correlates with next-period return |
| **IR** | Information Ratio — mean excess return / std excess return × √252 |
| **Seed factor** | Expert-curated starting expression with known baseline IR |
| **Evolution** | LLM generating incremental variants of the seed and testing them |
| **GRPO** | Group Relative Policy Optimization — PPO variant using in-group advantage normalization |
| **LoRA** | Low-rank adapter fine-tuning (used only in the local track) |
| **FSDP** | Fully Sharded Data Parallel — used for full fine-tuning in Verl |
| **sglang** | LLM inference engine with native multi-turn tool calling |
| **Verl** | Volcengine's distributed RL training framework |
| **VR** | Valid Ratio — % of tool calls that returned `success=True` |
| **Pass@T** | Fraction of seeds where ≥1 evolved factor beats seed IR within T turns |

## 9 · When in doubt

1. **Run the smoke tests in §5.** If any layer fails, fix that layer
   first; do not proceed upstream.
2. **Consult the matching topical doc** (`docs/02_*` through
   `docs/09_*`) before making non-trivial changes.
3. **Check `PLAN.md`** for the currently-known list of gaps between
   this implementation and the paper.
4. **Check `STATUS.md`** for the history of problems that have
   already been solved — chances are someone hit your bug first.
5. **Pin dependencies aggressively**, especially in the Verl track.
   Version drift is the #1 source of lost engineering time here.

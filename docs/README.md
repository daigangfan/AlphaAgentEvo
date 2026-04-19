# AlphaAgentEvo — Project Documentation

This folder contains a complete, self-contained reference for the
**AlphaAgentEvo** project. It describes every layer of the system, from raw
market data to the trained LLM agent that discovers alpha factors for the
Vietnam stock market.

The documentation is organized so that a new engineer (human or AI) can,
with only these docs plus the source tree, understand the project, modify
any component, and reproduce training/evaluation.

## Reading order

If you want the fastest path to running or modifying the project, go
straight to [`10_AGENT_GUIDE.md`](./10_AGENT_GUIDE.md) — it is the
"operational playbook" and points back into the other docs where details
are needed.

For a top-down understanding, read the docs in order:

1. [`01_overview.md`](./01_overview.md) — Problem statement, goals, high-level architecture, directory map.
2. [`02_data.md`](./02_data.md) — Market data (`daily_pv.h5`), seed factors, and training parquet files.
3. [`03_expression_language.md`](./03_expression_language.md) — Factor expression DSL, parser, and AST.
4. [`04_operators.md`](./04_operators.md) — Catalog of every operator available in `function_lib.py`.
5. [`05_backtest.md`](./05_backtest.md) — Factor executor, Qlib-consistent portfolio backtester, and API server.
6. [`06_reward.md`](./06_reward.md) — 5-component hierarchical reward (paper equation 5).
7. [`07_training.md`](./07_training.md) — GRPO training — both the local `trl` path and the distributed `Verl` path.
8. [`08_evaluation.md`](./08_evaluation.md) — Metrics (VR, Pass@3, beat rate) and the `evaluate.py` script.
9. [`09_deployment.md`](./09_deployment.md) — Environments, infrastructure, shell scripts, and known issues.
10. [`10_AGENT_GUIDE.md`](./10_AGENT_GUIDE.md) — **Final guide: how an AI agent should modify and run the project.**

## One-paragraph summary

AlphaAgentEvo fine-tunes `Qwen3-4B-Thinking-2507` with **GRPO**
(Group Relative Policy Optimization) and **multi-turn tool calling**.
Given a **seed alpha factor** (expression + baseline Information Ratio),
the LLM proposes evolved expressions and calls an `evaluate_factor` tool,
which runs a **Qlib-consistent backtest** on 219 Vietnam stocks
(2016–2026). The backtest returns the IR, which is fed back into the
conversation. A **5-component hierarchical reward** (paper equation 5)
rewards valid tool usage, structural consistency with the seed, novelty,
absolute performance, and consecutive breakthroughs. After ~150 training
steps the model learns to produce stronger alpha factors than the seeds.

## Project layout at a glance

```
AlphaAgentEvo/
├── backtest/              # Factor executor + Qlib backtester + FastAPI server
│   ├── api_server.py           # Port 8001 — /evaluate_factor endpoint
│   ├── factor_executor.py      # Parse + execute expression, compute IR
│   ├── qlib_backtester.py      # TopkDropout, T+2, 0.13% costs portfolio sim
│   └── data/daily_pv.h5        # 219 VN stocks, 12 columns, 2016-2026
├── configs/
│   └── grpo_config.yaml        # Local training config (RTX 5090)
├── data/                       # Seed factors (JSONL) + train/val/test parquet
├── deploy/                     # Cloud deployment (Runpod + Verl)
│   ├── api_server_verl.py      # Port 8002 — /backtest endpoint (paper contract)
│   ├── v2/                     # Verl v0.4.1 (stable) setup + train scripts
│   └── ... (v1 paper-fork files, factor_tool_vn.py, setup_runpod.sh)
├── expression_manager/         # DSL parser + AST tools
│   ├── expr_parser.py          # Expression string → executable Python code
│   ├── factor_ast.py           # Typed AST + structural similarity
│   └── function_lib.py         # All operators (RANK, TS_MEAN, MACD, ...)
├── training/                   # Local training (trl GRPOTrainer)
│   ├── train.py                # Main trainer
│   ├── factor_tool.py          # Tool interface + 5-dim reward
│   ├── generate_dataset.py     # Build train/val parquet from seed factors
│   ├── evaluate.py             # Evaluate checkpoints (VR, Pass@3)
│   ├── system_prompt.md        # Agent system prompt
│   └── tool_config.yaml        # evaluate_factor tool schema
├── configs/grpo_config.yaml    # Training hyperparameters + backtest periods
├── start_api.sh                # Launch FastAPI backtest server (port 8001)
├── start_training.sh           # Launch local GRPO training
├── PLAN.md                     # v2 gap-closure plan (paper parity)
├── STATUS.md                   # Project status, infra, resolved issues
└── docs/                       # You are here
```

## Convention used in these docs

- Code paths are given relative to the repo root
  (`F:\projects\AlphaAgentEvo` on the current machine).
- Line references use `path:line` or `path (start-end)`.
- Environment commands assume **bash** (WSL / Runpod / Linux). Windows
  users can run Python entry-points directly (see `09_deployment.md`).

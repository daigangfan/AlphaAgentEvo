# 01 · Overview and Architecture

## Problem

Discovering profitable **alpha factors** (functions of market data whose
cross-sectional ranking predicts future returns) is traditionally done by
human quant analysts. The original AlphaAgentEvo paper shows that an LLM
can learn this task via reinforcement learning: given a *seed* factor,
the LLM proposes improved variations, evaluates them through a backtest
tool, and learns from the resulting reward signal.

This repository is a **Vietnam-market re-implementation** of the paper.
It contains everything needed to:

1. Parse and execute an alpha factor expression written in a custom DSL.
2. Run a Qlib-consistent backtest on Vietnam equities and compute the
   Information Ratio (IR).
3. Expose that backtest as a FastAPI tool the LLM can call.
4. Train `Qwen3-4B-Thinking-2507` via **GRPO + multi-turn tool calling**
   using either a local `trl` pipeline (single GPU) or a distributed
   `Verl` pipeline (multi-GPU cluster).
5. Evaluate the trained model on held-out seed factors.

## High-level architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       GRPO Training Loop                           │
│                                                                    │
│   ┌─────────┐      ┌──────────┐       ┌────────────────────┐      │
│   │ Qwen3-4B│─────>│ Rollout   │──────>│ Tool Execution     │      │
│   │  LLM    │      │ engine    │       │ (multi-turn)       │      │
│   │ (trained│<─────│ (trl or   │<──────│                    │      │
│   │  w/ LoRA│      │  sglang)  │       │  1. Parse tool_call│      │
│   │  or FSDP│      └──────────┘       │  2. POST backtest   │      │
│   └─────────┘                          │  3. Inject result   │      │
│        │                               │     as tool_response│      │
│        │ grad                           │  4. Resume gen      │      │
│        v                                └────────────────────┘      │
│   ┌────────┐                                      │                  │
│   │ GRPO   │   ┌──────────────┐                   │                  │
│   │ update │<──│ Hierarchical │<──────────────────┘                  │
│   │        │   │ Reward Eq.5  │                                      │
│   └────────┘   └──────────────┘                                      │
└───────────────────────────────────────────────────────────────────┘
                               │
                               v
               ┌──────────────────────────┐
               │  Backtest API            │
               │  (FastAPI, port 8001/8002)│
               │                          │
               │  ┌──────────────────────┐│
               │  │ factor_executor      ││
               │  │  └─ expr_parser      ││
               │  │  └─ function_lib     ││
               │  │  └─ qlib_backtester  ││
               │  └──────────────────────┘│
               │  daily_pv.h5             │
               │  219 VN stocks × 12 cols │
               └──────────────────────────┘
```

### Request flow during training (one rollout)

1. The trainer samples a prompt from the parquet dataset. Each prompt
   contains the **system prompt** (operator reference, strategy hints) and
   a **user message** naming a seed factor and its baseline IR.
2. The LLM is asked to generate a response. The response is expected to
   contain one or more `<tool_call>{...}</tool_call>` blocks calling
   `evaluate_factor(factor_name, factor_expr)`.
3. The rollout engine parses each tool call, forwards it to the FastAPI
   backtest server, and re-injects the result as a `<tool_response>…`
   block. Generation resumes until the model stops calling tools or hits
   the turn limit (paper: 2 assistant turns, up to several tool calls per
   turn).
4. The reward function inspects the full trajectory
   (all successful/failed factor expressions and IRs) plus the seed
   expression and seed IR, and returns a single scalar reward.
5. GRPO normalizes rewards within a group of `num_generations`
   completions per prompt, computes advantages, and updates the model
   (with a KL penalty against the reference policy).

## Two deployment / training tracks

The codebase supports **two** completely separate training paths that
share all the lower-level machinery (data, backtester, DSL):

| Track | Location | Target hardware | Engine | Config |
|---|---|---|---|---|
| **Local (trl)** | `training/train.py` | 1× RTX 5090 (32 GB) | `trl.GRPOTrainer` + PEFT/LoRA | `configs/grpo_config.yaml` |
| **Cloud (Verl v0.4.1)** | `deploy/v2/train.sh` | 4× H100 80 GB (Runpod) | `verl.trainer.main_ppo` + sglang + FSDP | Hydra overrides in shell |

The backtest API is also split in two flavors:

| API | Port | Endpoint | Contract | Used by |
|---|---|---|---|---|
| `backtest/api_server.py` | 8001 | `POST /evaluate_factor` | `{factor_expr, factor_name, period}` | Local `trl` training |
| `deploy/api_server_verl.py` | 8002 | `POST /backtest` | Paper contract `{exprs:{name:expr}, ...}` | Cloud Verl training |

Both APIs wrap the same underlying `backtest.factor_executor.execute_expression`.

## Components in depth

Each component has its own doc:

- **Data layer** → [`02_data.md`](./02_data.md)
- **Expression DSL and parser** → [`03_expression_language.md`](./03_expression_language.md)
- **Operator library** → [`04_operators.md`](./04_operators.md)
- **Backtest (executor + Qlib portfolio sim + API)** → [`05_backtest.md`](./05_backtest.md)
- **Reward function (paper equation 5)** → [`06_reward.md`](./06_reward.md)
- **Training** → [`07_training.md`](./07_training.md)
- **Evaluation** → [`08_evaluation.md`](./08_evaluation.md)
- **Deployment scripts / environments** → [`09_deployment.md`](./09_deployment.md)
- **How to modify and run as an agent** → [`10_AGENT_GUIDE.md`](./10_AGENT_GUIDE.md)

## Key hyperparameters (paper-matching)

| Parameter | Value | Where set |
|---|---|---|
| Batch size (prompts/step) | 20 | `deploy/v2/train.sh` (`BATCH_SIZE=N_GPUS*7` → tune to 20) |
| Rollouts per prompt | 3 | `actor_rollout_ref.rollout.n=3` |
| Max assistant turns | 2 | `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2` |
| Total training steps | 150 | `trainer.total_training_steps=150` |
| Learning rate | 1e-6 | `actor_rollout_ref.actor.optim.lr=1e-6` |
| KL coefficient (β) | 0.001 | `actor_rollout_ref.actor.kl_loss_coef=0.001` |
| Save frequency | every 10 steps | `trainer.save_freq=10` |
| Eval checkpoint | step 80 | Paper recommendation |
| Temporal splits | train 2016–2023, val 2024, test 2025–2026 | `configs/grpo_config.yaml` |

## Why two trainers?

The local `trl` path was the first iteration — it proved the full
pipeline works on a single RTX 5090 using LoRA `r=64`, but capacity was
limited (rewards stalled around 0.22 vs. the paper's 0.38). The Verl
path performs **full fine-tuning** on multi-GPU H100s using FSDP and the
**sglang** engine that natively supports multi-turn tool calling — this
is the production path.

See [`09_deployment.md`](./09_deployment.md) for the painful dependency
story behind the Verl path (pinned to `v0.4.1`, sglang `0.4.6.post5`,
torch `2.6.0`) and [`PLAN.md`](../PLAN.md) for the list of gaps between
the local `trl` path and the paper's reference behavior.

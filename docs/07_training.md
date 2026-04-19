# 07 · Training (GRPO)

There are **two** training paths, described here in parallel: the
**local `trl` path** (single-GPU, LoRA) and the **distributed `Verl`
path** (multi-GPU, full fine-tuning, paper-parity).

At a high level, both paths implement the same training loop:

```
for step in range(total_steps):
    prompts  = sample_batch(train_dataset, batch_size)
    for each prompt:
        for g in range(num_generations):
            completion, trajectory = multi_turn_generate(model, prompt, tools=[evaluate_factor])
            reward = calc_reward(trajectory, seed_expr, seed_ir)
    advantages = group_normalize(rewards_per_prompt)
    loss = -E[min(ratio * adv, clip(ratio) * adv)] + β·KL(policy || ref)
    backward(); optimizer.step()
```

The differences are (a) which framework implements multi-turn
generation, and (b) how the policy model is parallelized.

## Path A — Local (`trl`) on a single RTX 5090

### Files

| File | Role |
|---|---|
| `training/train.py` | Entry point |
| `training/factor_tool.py` | `FactorTool.evaluate` + `calc_reward` |
| `training/tool_config.yaml` | `evaluate_factor` schema (for docs) |
| `training/system_prompt.md` | System message |
| `configs/grpo_config.yaml` | Hyperparameters + backtest periods |
| `start_api.sh`, `start_training.sh` | Shell launchers |

### Model

- Base: `Qwen3-4B-Thinking-2507` (configurable via
  `model.name` in `grpo_config.yaml`).
- The **Thinking variant** has a chat template that forces
  `<think>\n` on every assistant turn. `trl` cannot handle that, so
  `train.py` loads the **base** Qwen3-4B template (by removing the
  `-Thinking-2507` suffix) and overrides `tokenizer.chat_template`.
- Mixed precision: `bf16`.
- Attention: `flash_attention_2` if `model.flash_attn: true`, else
  `sdpa` (the default on RTX 5090 because flash-attn isn't built for
  sm_120 yet).

### LoRA adapter

Configured in `configs/grpo_config.yaml::lora`:

```yaml path=F:\projects\AlphaAgentEvo\configs\grpo_config.yaml start=8
lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj, k_proj, v_proj, o_proj
    - gate_proj, up_proj, down_proj
```

### GRPO loop

`train.py` uses `trl.GRPOTrainer` directly. Key choices:

```python path=F:\projects\AlphaAgentEvo\training\train.py start=405
grpo_config = GRPOConfig(
    num_generations=num_gens,                # default 4
    generation_batch_size=num_gens,
    max_completion_length=training_config.get("max_new_tokens", 3072),
    max_tool_calling_iterations=training_config.get("max_tool_calling_iterations", 3),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=150,
    learning_rate=1e-6,
    bf16=True,
    beta=0.001,                               # KL coefficient
    seed=42,
)
```

- `max_tool_calling_iterations=3` — trl's native multi-turn tool loop
  runs at most 3 model → tool → tool-response cycles per completion.
- `tools=[evaluate_factor]` — trl discovers the schema from the
  function's signature + docstring.

### Reward function bridging

`trl.GRPOTrainer` calls `reward_func(prompts=..., completions=...,
**kwargs)` with dataset columns passed through `**kwargs`. We use
this to pipe `seed_expr` and `seed_ir` into `calc_reward`:

```python path=F:\projects\AlphaAgentEvo\training\train.py start=259
def reward_func(prompts, completions, **kwargs):
    seed_exprs = kwargs.get("seed_expr", [])
    seed_irs   = kwargs.get("seed_ir", [])
    rewards = []
    for i, completion in enumerate(completions):
        trajectory = parse_trajectory_from_completion(completion)
        rewards.append(factor_tool.calc_reward(trajectory, seed_exprs[i], float(seed_irs[i])))
    return rewards
```

### How to run

```bash path=null start=null
# Terminal 1 — backtest API (port 8001)
bash start_api.sh

# Terminal 2 — GRPO training
bash start_training.sh                 # full 150-step run
# or
bash start_training.sh --smoke-test    # 5 steps, 5 examples
```

Outputs land in `outputs/grpo_<unix_timestamp>/`. The final merged
LoRA model is saved under `outputs/grpo_<ts>/final/`.

TensorBoard:

```bash path=null start=null
tensorboard --logdir outputs/grpo_*/
```

### Known gaps vs paper (see `PLAN.md`)

The local trl pipeline has 4 known differences from the paper:

1. **Tool responses are not real execution** in trl's multi-turn loop
   — trl just re-invokes the function. Our custom
   `parse_trajectory_from_completion` works around this by
   re-evaluating any expression the model proposes.
2. **Loss masking**: we don't manually mask out `<tool_response>`
   tokens from the policy loss (the paper does). `PLAN.md` describes
   a custom GRPO loop that would.
3. **Dataset size** is ~66 factors with `seed_factors.json` only; to
   reach paper's 300/30/100 splits, run `generate_dataset.py
   --augment` or use the Verl track's `vn_seeds_300.jsonl`.
4. **Reward formula** used here matches the paper; however the
   `deploy/v2/factor_tool.py` Verl variant uses a simpler
   consistency/exploration term.

## Path B — Distributed (`Verl v0.4.1`) on 4× H100

This is the paper-parity training path and the recommended one for
final runs.

### Files

| File | Role |
|---|---|
| `deploy/v2/setup.sh` | One-command environment bootstrap |
| `deploy/v2/train.sh` | Launch `verl.trainer.main_ppo` |
| `deploy/v2/factor_tool.py` | `BaseTool` implementation |
| `deploy/v2/factor_tool_config.yaml` | Verl Hydra tool registration |
| `deploy/v2/factor_reward.py` | Fallback `compute_score` for the `custom_reward_function` hook |
| `deploy/v2/{train,val,test}.parquet` | Data (same schema as `data/*.parquet`) |
| `deploy/api_server_verl.py` | Backtest API on port 8002 (paper contract) |

### Environment (pinned, critical)

```
Python      3.10  (conda env verl041)
Verl        v0.4.1 (pinned tag, NOT HEAD)
sglang      0.4.6.post5 (from Verl setup.py)
torch       2.6.0 (from Verl setup.py)
tensordict  <=0.6.2 (from Verl setup.py)
```

These versions are chosen because Verl v0.4.1's release notes
explicitly include:

- sglang multi-turn rollout,
- Verl interaction system,
- fix for "tool call parser not found" in sglang 0.4.6.post5.

See `deploy/v2/setup.sh` for the full install sequence. It also
patches `flash_attn.bert_padding` imports out of Verl because flash
attention has an ABI mismatch with torch 2.6.0 on H100.

### Training launch

```bash path=null start=null
# Once
bash deploy/v2/setup.sh

# Every run
nohup python deploy/api_server_verl.py > /workspace/v2/logs/api.log 2>&1 &
nohup bash deploy/v2/train.sh          > /workspace/v2/logs/train.log 2>&1 &
```

Key Hydra overrides in `deploy/v2/train.sh`:

| Override | Value |
|---|---|
| `algorithm.adv_estimator` | `grpo` |
| `data.train_batch_size` | `$N_GPUS * 7` (≈28 on 4 GPUs; tune to 20 for paper) |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |
| `actor_rollout_ref.rollout.name` | `sglang` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.6` |
| `actor_rollout_ref.rollout.n` | `3` (rollouts per seed) |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `2` |
| `trainer.save_freq` | `10` |
| `trainer.total_training_steps` | `150` |
| `trainer.logger` | `["console","tensorboard"]` |

The tool config is registered at
`actor_rollout_ref.rollout.multi_turn.tool_config_path=
$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml`.

### Verl tool interface

Verl's **sglang rollout engine** understands tool calls natively. Our
`FactorTool` (in `deploy/v2/factor_tool.py`) subclasses
`verl.tools.base_tool.BaseTool` and implements:

| Hook | Purpose |
|---|---|
| `create(instance_id)` | init per-trajectory state (seed expr, seed IR, streak, ...) |
| `execute(instance_id, parameters)` | call `/backtest`, update state, compute partial reward |
| `calc_reward(instance_id)` | compute paper Eq.5 reward over accumulated state |
| `release(instance_id)` | cleanup |

This is the key architectural difference from the local path: the
reward is computed **inside** the rollout engine at each tool call,
rather than after the fact by a reward function that inspects the
completion text.

### Checkpoints

Saved every 10 steps under
`$WORK/checkpoints/<project>/<experiment>/`. Paper recommends using
**step 80** for evaluation (not the last checkpoint — over-training
from step 80 to 150 degrades evaluation metrics).

### Monitoring

```bash path=null start=null
tail -n 40 /workspace/v2/logs/train.log | grep -E "step|reward|critic"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

Key training metrics printed by Verl:

- `critic/score/mean` — the mean trajectory reward (this is what we
  want to grow).
- `actor/entropy` — token-level entropy of the policy.
- `actor/loss` — PPO-style clipped loss.
- `actor/kl_loss` — KL against the reference policy.

If `critic/score/mean` stays at 0 for the first few steps, the most
likely cause is the backtest API not being reachable — check
`/workspace/v2/logs/api.log`.

## Summary table

| Dimension | `trl` local | Verl distributed |
|---|---|---|
| GPUs | 1× RTX 5090 | 4× H100 (min) |
| Fine-tune | LoRA r=64 | Full (FSDP) |
| Framework | `trl.GRPOTrainer` | `verl.trainer.main_ppo` |
| Rollout | `trl` native tool calling | `sglang` + BaseTool |
| API port | 8001 (`/evaluate_factor`) | 8002 (`/backtest`) |
| Config | `configs/grpo_config.yaml` | Hydra overrides in `deploy/v2/train.sh` |
| Reward impl | `training/factor_tool.py` (AST-based) | `deploy/v2/factor_tool.py` (simplified) |
| Steps to reach paper-level IR | Limited by LoRA capacity (~0.22) | ~150 to reach ~0.38 |
| Known issues | Parquet version skew; flash-attn unavailable on sm_120 | Pinned dependency hell — use v0.4.1 exactly |

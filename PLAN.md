# PLAN · 16 GB VRAM single-GPU (RTX 4080) training budget
Goal: shrink the `training/train.py` (trl GRPO + Qwen3-4B + LoRA) pipeline so
that a smoke-test run at `per_device_train_batch_size=1` completes on a
16 GB RTX 4080. Today's `configs/grpo_config.yaml` targets a 32 GB RTX 5090
and overshoots the 4080 budget by 6–10 GB (see §1).
## 1 · Current VRAM footprint (estimate, per `configs/grpo_config.yaml`)
Qwen3-4B (~4.0B params), bf16 weights, LoRA r=64 on 7 modules,
`num_generations=4`, `max_new_tokens=5000`, `max_tool_calling_iterations=3`:
- Base weights (bf16): ~8.0 GB
- LoRA params (r=64, 7 modules) + grads + AdamW state: ~1.0 GB
- Reference policy (PEFT base, shared w/ base, no KL ref model needed): ~0 GB
- Activation + KV cache during GRPO rollout (4 gens × up to 5000 new tokens × prompt): ~8–12 GB
- Generation workspace + misc. allocator fragmentation: ~1–2 GB
Total: ~18–23 GB. Fits on 32 GB RTX 5090, OOMs on 16 GB RTX 4080.
## 2 · Optimization strategy
We combine five memory-reduction techniques. QLoRA (4-bit base) is the
single largest win and is the foundation; the rest trim rollout/activation
memory so that the GRPO loop itself (not just the weights) fits.
### 2.1 · QLoRA (4-bit NF4 base) — primary lever
Quantize the base Qwen3-4B weights to NF4 via `bitsandbytes` (already in
`requirements.txt`). Base-weight memory drops from ~8.0 GB → ~2.3 GB
while LoRA adapters remain in bf16 and train normally.
- Rationale over raw fp8: native fp8 training on Ada (sm_89 / 4080) needs
  TransformerEngine + custom fused kernels and is not wired into
  `trl.GRPOTrainer` or `peft`. NF4 QLoRA is the standard, battle-tested
  path and gives ~4× weight compression vs. bf16 with negligible quality
  loss for LoRA fine-tuning.
- `bnb_4bit_compute_dtype=bfloat16`, `bnb_4bit_quant_type="nf4"`,
  `bnb_4bit_use_double_quant=True`.
- After load, call `peft.prepare_model_for_kbit_training(model)` so
  LayerNorm/embedding hooks and gradient checkpointing are wired
  correctly.
### 2.2 · Smaller LoRA adapter
Reduce capacity (and therefore gradient + optimizer memory) from the
paper-matched r=64 config to a smoke-test-friendly r=16 adapter.
- `lora.r`: 64 → 16
- `lora.alpha`: 128 → 32 (keep α/r = 2)
- `target_modules`: keep all 7 (attention + MLP) projections — dropping MLP
  modules hurts GRPO factor-evolution learning more than shrinking `r`.
  Net param reduction is ~4× (r dominates).
### 2.3 · Shorter rollouts
The 5 000-token ceiling is the largest activation driver. Reduce it, plus
cut rollout breadth.
- `max_new_tokens`: 5000 → 768 (the paper's factor expressions + short
  thinking blocks fit comfortably in < 700 tokens; system prompt already
  contains `/no_think`).
- `num_generations`: 4 → 2 (GRPO's group-relative advantage requires
  ≥2; 2 is the minimum valid group size).
- `max_tool_calling_iterations`: 3 → 2 (fewer multi-turn cycles ⇒ shorter
  final trajectory the trainer has to backprop through).
### 2.4 · Gradient checkpointing
Set `gradient_checkpointing=True` on `GRPOConfig` with
`gradient_checkpointing_kwargs={"use_reentrant": False}` (required with
PEFT + 4-bit). Recomputes activations in the backward pass; trades ~20 %
time for ~30–40 % activation memory. `use_cache` is automatically
disabled during training by `prepare_model_for_kbit_training`.
### 2.5 · Keep `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`
Unchanged: the user explicitly asked for `batch=1`. Effective batch
remains 4.
## 3 · Expected post-optimization footprint
- Base weights (NF4 + double-quant): ~2.3 GB
- LoRA (r=16, 7 modules) + grads + AdamW: ~0.15 GB
- Activations during backward (grad-ckpt, 768-token completions): ~3–4 GB
- Rollout KV cache (2 gens × 768 tokens, bf16): ~1.5 GB
- Allocator/fragmentation headroom: ~1–2 GB
Projected total: ~8–10 GB. Leaves >5 GB headroom under the 16 GB cap for
tokenizer/dataset buffers and occasional peaks during `sglang`-style
multi-turn rollouts.
## 4 · Files to change
### 4.1 · `configs/grpo_config.yaml`
- Under `model`: add `quantization: "nf4"` flag (consumed by `train.py`).
- Under `lora`: `r: 16`, `alpha: 32` (keep `target_modules` list).
- Under `training`: `num_generations: 2`, `max_new_tokens: 768`,
  `max_tool_calling_iterations: 2`, `gradient_checkpointing: true`.
- Update the top-of-file comment to reflect the new 16 GB target.
### 4.2 · `training/train.py`
- Import `BitsAndBytesConfig` from `transformers`.
- When `model_config["quantization"] == "nf4"`, build a
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`
  and pass it to `from_pretrained` (dropping the top-level `dtype` kwarg
  in that branch — bnb handles compute dtype).
- Call `prepare_model_for_kbit_training(model,
  use_gradient_checkpointing=training_config.get("gradient_checkpointing",
  True))` right after load.
- Allow an environment-variable override for `model.name` (`MODEL_PATH`)
  so the current hardcoded Linux path
  (`/home/dc_analyst/models/Qwen3-4B-Thinking-2507`) does not block the
  Windows smoke test; fall back to the config value if unset.
- Pass `gradient_checkpointing=True` and
  `gradient_checkpointing_kwargs={"use_reentrant": False}` to
  `GRPOConfig`.
### 4.3 · (No change)
- `training/factor_tool.py`, reward logic, dataset format — untouched.
- `docs/10_AGENT_GUIDE.md` §3.4 warns that a 7 B doesn't fit in LoRA on
  32 GB — this plan reinforces that warning, so no doc edit is needed.
## 5 · Smoke-test procedure
The user's `start_training.sh` drives the run. On Windows/pwsh the bash
script can be mirrored via:
```powershell path=null start=null
# (1) Terminal A: backtest API on port 8001
bash start_api.sh
# (2) Terminal B: point at a local Qwen3-4B checkout if needed
$env:MODEL_PATH = "F:\models\Qwen3-4B-Thinking-2507"   # or any local path / HF id
python training/train.py --config configs/grpo_config.yaml --smoke-test
```
Success criteria for the smoke test:
1. Peak `nvidia-smi` memory stays < 15.5 GB on the 4080.
2. Five GRPO steps complete without OOM.
3. At least one step shows a non-zero reward (validates tool-calling +
   reward plumbing end-to-end, independent of the memory changes).
## 6 · Rollback levers (if 4080 still OOMs)
Apply in order of least → most quality impact:
1. `max_new_tokens: 768 → 512`.
2. `num_generations: 2` (already minimum) — instead drop
   `gradient_accumulation_steps: 4 → 1` to shorten the train step.
3. Shrink `target_modules` to attention-only (drop `gate_proj`,
   `up_proj`, `down_proj`).
4. `lora.r: 16 → 8`.
5. Switch optimizer to `paged_adamw_8bit` (bitsandbytes) via
   `GRPOConfig(optim="paged_adamw_8bit")`.

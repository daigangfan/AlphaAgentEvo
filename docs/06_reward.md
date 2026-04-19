# 06 · Reward Function (Paper Equation 5)

The GRPO trainer optimizes a **hierarchical reward** computed over an
entire multi-turn trajectory. It is the single most important piece of
research logic in the project — it tells the model what "good" means.

The reward function lives in two nearly-identical locations:

- `training/factor_tool.py` — used by the local `trl` track.
- `deploy/v2/factor_tool.py` — used by the Verl track (minor
  simplifications so it can run inside a stateful `BaseTool` that
  accumulates trajectory info across tool calls).

## Reward formula

From the paper (equation 5):

```
R(τ) = [ min(R_cons, C_cons) + min(R_expl, C_expl) ] / min(R_tool, C_tool)
       + min(R_perf, C_perf) · min(R_streak, C_streak)
```

Caps (`C_*`) and weights (`α_*`) are taken from the paper's Appendix D:

| Term | Cap `C` | Weight `α` |
|---|---|---|
| `R_tool` | 1.0 | α_succ = 0.1, α_fail = 0.2 |
| `R_cons` | 0.2 | α_cons = 0.02 |
| `R_expl` | 0.3 | α_exp = 0.02 |
| `R_perf` | 0.5 | α_perf = 0.1 |
| `R_streak` | 0.6 | α_streak = 0.15 |

A **floor** `R_TOOL_FLOOR = 0.01` is applied to `R_tool` as the
denominator (paper code has a divide-by-zero bug noted in the
`PLAN.md` BigPicture doc; we safeguard with `max(r_tool, 0.01)`).

## Component definitions

All implemented in `training/factor_tool.py::calc_reward` (and mirrored
in the helpers `_calc_consistency`, `_calc_exploration`,
`_calc_performance`, `_calc_streak`):

### R_tool — "did the tool actually work?"

```
R_tool = α_succ · N_success − α_fail · N_failure
       = 0.1 · N_success − 0.2 · N_failure
```

Used as the **denominator** of the directional term. Rewards the model
for emitting syntactically-valid tool calls that actually backtest.

### R_cons — consistency with the seed

```
R_cons = Σ_{f_i ∈ F_success} α_cons · 1[ H_LOW < sim(f_i, f_seed) < H_HIGH ]
       = 0.02 · (# of successful factors structurally similar to seed)
```

With `H_LOW = 0.1`, `H_HIGH = 0.9`. Rewards proposals that build on the
seed's idea without being identical.

### R_expl — exploration / novelty

```
R_expl = Σ_{f_i ∈ F_success} α_exp · (1 − max_{f_j ∈ F_<i ∪ {seed}} sim(f_i, f_j))
       = 0.02 · Σ (1 − max_sim_to_predecessors)
```

Rewards each new proposal for being less similar to ALL earlier
proposals (including the seed).

### R_perf — absolute performance

```
R_perf = α_perf · log(1 + exp(s(f*) − max(0, s(f_seed))))
       = 0.1 · softplus(best_ir − max(0, seed_ir))
```

A softplus over the improvement of the best successful factor's IR
relative to the seed IR. Always positive (`softplus(0) ≈ 0.693`).
Numerically safeguarded by clamping the exponent to `[-20, 20]`.

### R_streak — consecutive breakthroughs

```
R_streak = α_streak · N_streak
         = 0.15 · (# of times a new best_ir was set during the trajectory)
```

Initial `best_so_far = seed_ir`. Each time a successful factor beats
the running best, `best_so_far` is updated and the streak counter
increments.

## Interpretation

Split the formula into:

- **Directional term** `(R_cons + R_expl) / R_tool` — rewards
  *diverse yet seed-aware* proposals while penalising wasted tool
  calls. The division is what encourages the model to get the first
  few calls right.
- **Quality term** `R_perf · R_streak` — rewards progress in IR,
  scaled by how many consecutive breakthroughs occurred.

The two terms are added, giving a reward that's roughly in
`[0, ~1.0]` in practice. The paper reports models reaching ~0.38 by
step 150 (local `trl` LoRA plateaued around 0.22).

## What inputs does it take?

```python path=F:\projects\AlphaAgentEvo\training\factor_tool.py start=115
def calc_reward(
    self,
    trajectory_results: list[dict],   # [{success, ir, factor_expr}, ...]
    seed_expr: str,                    # seed's DSL expression (string)
    seed_ir: float,                    # seed's baseline IR
) -> float:
```

The local trainer assembles `trajectory_results` from the multi-turn
completion via `parse_trajectory_from_completion` in
`training/train.py`. That parser handles:

- **Structured** message lists (with `role: assistant|tool` and
  `tool_calls` arrays) — preferred when available.
- **Text** completions with interleaved `<tool_call>` / `<tool_response>`
  blocks (fallback for engines that serialize multi-turn to a single
  string).

If a response can't be parsed, the parser re-evaluates the expression
directly via the backtest API, so the reward is always consistent with
the real backtest result rather than the model's own claim.

## Verl variant

`deploy/v2/factor_tool.py::FactorTool.calc_reward` computes the **same
hierarchical reward** but with simplified consistency/exploration
terms that do not use AST similarity (they count unique expressions
instead). This keeps the Verl tool self-contained (no dependency on
`expression_manager`). If you want the AST-based reward during Verl
training, port `_ast_similarity` into `deploy/v2/factor_tool.py` or
import it from the repo.

## Unit test

Running `python training/factor_tool.py` (its `__main__` block)
exercises `calc_reward` on a mocked trajectory:

```python path=F:\projects\AlphaAgentEvo\training\factor_tool.py start=256
if __name__ == "__main__":
    # Quick test of reward calculation (without API)
    tool = FactorTool()
    seed_expr = "RANK(TS_MEAN($return, 5) / (TS_STD($return, 20) + 1e-8)) * SIGN(DELTA($close, 5) / ($close + 1e-8))"
    seed_ir = 1.15
    ...
    reward = tool.calc_reward(trajectory, seed_expr, seed_ir)
```

Expected output includes `R_tool raw/capped/denom` and a final reward
value. Use this as a quick sanity check after modifying any of the
reward components.

## Modifying the reward

Good hygiene when tweaking any of the caps, weights, or component
formulas:

1. Change **both** `training/factor_tool.py` and
   `deploy/v2/factor_tool.py` (unless the change is Verl-specific).
2. Re-run the `__main__` smoke test to ensure values are in a
   reasonable range.
3. Consider updating `system_prompt.md` if the new behavior requires
   different agent strategy (e.g. tolerating more failures).
4. If you change `R_perf`, ensure `seed_ir` values in
   `data/*.parquet` still make sense — negative seed IRs are OK
   (clamped to 0 in the baseline).

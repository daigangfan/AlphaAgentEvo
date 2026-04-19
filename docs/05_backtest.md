# 05 · Backtest (Executor + Qlib Portfolio Sim + API)

The backtest layer turns a DSL expression into an **Information Ratio**
scalar. It has three stacked components.

```
┌─────────────────────────────────────────────┐
│         FastAPI server (API)                 │
│  backtest/api_server.py (port 8001)          │
│  deploy/api_server_verl.py (port 8002)       │
└──────────────────┬──────────────────────────┘
                   │ execute_expression(expr, period)
                   v
┌─────────────────────────────────────────────┐
│       Factor Executor                        │
│  backtest/factor_executor.py                 │
│   1. parse DSL → Python code                 │
│   2. eval() against cached dataframe         │
│   3. call compute_portfolio_ir()             │
└──────────────────┬──────────────────────────┘
                   │ factor_values + price_df
                   v
┌─────────────────────────────────────────────┐
│       Qlib-Consistent Portfolio Simulator    │
│  backtest/qlib_backtester.py                 │
│   - top_k=10 dropout-style rebalance          │
│   - T+2 settlement                            │
│   - 0.13% transaction costs                   │
│   - IR, Sharpe, MDD, turnover                 │
└─────────────────────────────────────────────┘
```

## 1. Factor Executor — `backtest/factor_executor.py`

### Entry point

```python path=F:\projects\AlphaAgentEvo\backtest\factor_executor.py start=108
def execute_expression(factor_expr: str,
                       data_path: str | Path | None = None,
                       period: str | None = None) -> dict:
```

Returns a dict with:

| Key | Meaning |
|---|---|
| `success` | `True` if everything ran end-to-end |
| `score` / `ir` | Information Ratio (the main metric) |
| `annualized_return` | Mean excess return × 252 |
| `annualized_volatility` | Std excess return × √252 |
| `sharpe` | Same as IR in this implementation |
| `mdd` | Max drawdown of cumulative excess return |
| `total_return` | Cumulative daily return over the period |
| `n_days` | Number of effective trading days in the simulated portfolio |
| `error` | Error string or `None` |
| `exec_time` | Wall time in seconds |

### Internal steps

1. `parse_expression(factor_expr)` → Python code string.
2. `parse_symbol(code, columns)` strips `$` from variable names; an
   additional regex replace maps Python keywords to safe `col_*`
   names.
3. `exec_namespace` is built by auto-importing every non-private
   callable from `function_lib` plus `np`, `pd`, and each column of
   `daily_pv.h5`.
4. `eval(executable_code, exec_namespace)` runs the factor.
5. The resulting Series (or DataFrame's first column) is passed to
   `qlib_backtester.compute_portfolio_ir` with the `$open` / `$close`
   prices and `$bench_return`.
6. Exceptions are caught and returned as `error` with
   `success=False`.

### Period masking

```python path=F:\projects\AlphaAgentEvo\backtest\factor_executor.py start=47
def configure_periods(periods: dict[str, dict[str, str]]) -> None:
```

Call once at startup with a `{name: {start, end}}` map, typically
loaded from `configs/grpo_config.yaml::backtest.periods`. Period masks
are then passed to `compute_portfolio_ir` as `start_date`/`end_date`.

## 2. Qlib-Consistent Portfolio Simulator — `backtest/qlib_backtester.py`

### Strategy

Matches the AlphaAgent paper's `conf_vn_combined_kdd_ver.yaml` Qlib
config **exactly**. The simulator answers the question "if we rebuild
a top-10 portfolio every 5 days based on the factor's cross-sectional
ranking, what Information Ratio would we have earned?"

Key rules:

- **Signal day**: use yesterday's factor values (`T-1` signal, prevents
  look-ahead bias).
- **Rebalance**: every `max(rebalance_freq, hold_thresh) = 5` trading
  days. On each rebalance day:
  1. Force-sell holdings past `hold_thresh` days (T+2 settlement).
  2. Drop the `n_drop=2` lowest-scored current positions (plus any
     positions not in the new top-k).
  3. Buy the highest-scored remaining names, filling up to `top_k=10`
     slots.
- **Execution**: buy at `$open` today, sell at `$close` today (same day
  it's decided), reflecting the paper's "force-exit top-k dropout"
  strategy.
- **Transaction cost**: 0.13% per side (0.0013 buy + 0.0013 sell).
- **Equal weight** per position at entry; P&L is marked-to-market at
  close every day.

### Metrics (Qlib "sum" mode)

```
mean_r = mean(daily_excess_return)
std_r  = std (daily_excess_return, ddof=1)

IR                = mean_r / std_r · √252
annualized_return = mean_r · 252
annualized_vol    = std_r  · √252
MDD               = min(cumsum - cummax(cumsum))
total_return      = sum(daily_returns)  # NOT excess
```

These match Qlib's formulas for `AnnualizedReturn`, `InformationRatio`,
`MaxDrawdown` in "sum" accumulation mode. The paper specifically uses
Qlib's implementation, so our outputs are comparable to the paper's
IR numbers.

### Benchmark

If `$bench_return` is in the dataframe, the portfolio's daily return is
subtracted from it to yield the **excess** return. Otherwise the raw
portfolio return is used. VNINDEX is the benchmark throughout.

### Common returns of `compute_portfolio_ir`

`success=False` is returned if the requested period has fewer than
`rebalance_freq + hold_thresh + 1` days, or if `$open`/`$close`
columns are missing. An empty result struct is returned with IR=0.

## 3. API servers — `api_server.py` & `api_server_verl.py`

### `backtest/api_server.py` (port 8001) — local track

- On startup: load `grpo_config.yaml`, call `configure_periods`, load
  data.
- `POST /evaluate_factor` body:
  ```json path=null start=null
  {"factor_name": "momentum_20d",
   "factor_expr": "RANK(TS_PCTCHANGE($close, 20))",
   "period": "train"}
  ```
- Response matches `EvaluateResponse` pydantic model (flat dict:
  `success, score, ic_mean, ic_std, ir, error, exec_time`).
- `POST /batch_evaluate` accepts a **list** of the above bodies.
- `GET /health` → `{"status": "ok", "service": "alphaagentevo-backtest"}`.

`training/factor_tool.py::FactorTool.evaluate` calls this endpoint
during training.

### `deploy/api_server_verl.py` (port 8002) — Verl track

Speaks the **paper's contract** so it can be consumed by the paper's
`factor_tool.py`:

- `POST /backtest` body mirrors the paper's request:
  ```json path=null start=null
  {"exprs": {"momentum_20d": "RANK(TS_PCTCHANGE($close, 20))"},
   "backtest_start_time": "2016-01-01",
   "backtest_end_time": "2023-12-31",
   "stock_pool": "VN100",
   ...}
  ```
- Response wraps the metric dict the paper expects:
  ```json path=null start=null
  {"data": {
     "success": true,
     "metrics": {
       "Information_Ratio_with_cost": 0.123,
       "Information_Ratio_without_cost": 0.123,
       "Annualized_Return_with_cost": 0.08,
       "Max_Drawdown_with_cost": -0.09,
       "IC": 0.006,         // approximate, ir * 0.05
       "ICIR": 0.123,
       "RankIC": 0.005,
       "RankICIR": 0.098
     },
     "exec_time": 0.83
  }}
  ```

  Internally both endpoints ultimately call
  `backtest.factor_executor.execute_expression(...)`.

### Starting the API

| Track | Command |
|---|---|
| Local (port 8001) | `bash start_api.sh` (uses `uvicorn`) |
| Cloud (port 8002) | `nohup python deploy/api_server_verl.py > logs/api.log 2>&1 &` |

Health check before starting training:

```bash path=null start=null
curl -s http://localhost:8001/health   # or 8002
```

Both endpoints emit a single startup log line showing the loaded
period masks:

```
[FactorExecutor] Period 'train': 2016-01-01 to 2023-12-31 — 1986 days, 436029 rows
[FactorExecutor] Period 'val':   2024-01-01 to 2024-12-31 — 249 days, ...
```

## Quick smoke test

```bash path=null start=null
# Start API
bash start_api.sh

# In another terminal
curl -s -X POST http://localhost:8001/evaluate_factor \
    -H "Content-Type: application/json" \
    -d '{"factor_name":"mom20","factor_expr":"RANK(TS_PCTCHANGE($close, 20))","period":"train"}'
```

Expected: a JSON with `success: true` and a non-zero `ir`. If you see
`success: false`, check:

- `daily_pv.h5` is present in `backtest/data/`.
- The expression uses only operators defined in `function_lib.py`
  (see [`04_operators.md`](./04_operators.md)).
- The period name is one of `train`, `val`, `test`, or unset.

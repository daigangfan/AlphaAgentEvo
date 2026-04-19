# 04 · Operator Library (`function_lib.py`)

Every operator in the DSL resolves to a Python function in
`expression_manager/function_lib.py`. This file is the **single source
of truth** for what operators exist. If an expression uses a name that
is not defined here, `execute_expression` will raise `NameError` at
`eval` time and the backtest API will return `success=False`.

All operators assume the input data is indexed by
`(datetime, instrument)` — rolling (`TS_*`) functions group by
`instrument`; cross-sectional functions group by `datetime`.

## Decorator: `datatype_adapter`

Wraps most functions so they tolerate `np.ndarray`, `pd.DataFrame`,
`pd.Series`, and scalars. Converts to DataFrame on entry, passes through
the real function, and does not convert back on exit (pandas objects
are returned). This is why most operators appear to take `DataFrame`
even though the executor passes `Series`.

## Operator catalog

Operators grouped by category. The **signature column** uses the DSL
names as written in expressions (not the Python function names, which
are the same except sometimes typed differently internally).

### Time-series (per instrument)

| Operator | Signature | Purpose |
|---|---|---|
| `DELTA` | `DELTA(X, d)` | `X_t - X_{t-d}` |
| `DELAY` | `DELAY(X, d)` | `X_{t-d}` (must have `d ≥ 0`) |
| `TS_MEAN` | `TS_MEAN(X, d)` | Rolling mean |
| `TS_STD` | `TS_STD(X, d)` | Rolling standard deviation (default `d=20`) |
| `TS_VAR` | `TS_VAR(X, d, ddof=1)` | Rolling variance |
| `TS_SUM` | `TS_SUM(X, d)` | Rolling sum |
| `TS_MAX` | `TS_MAX(X, d)` | Rolling max |
| `TS_MIN` | `TS_MIN(X, d)` | Rolling min |
| `TS_RANK` | `TS_RANK(X, d)` | Rolling percentile rank |
| `TS_MEDIAN` | `TS_MEDIAN(X, d)` | Rolling median |
| `TS_ARGMAX` | `TS_ARGMAX(X, d)` | Days since rolling max |
| `TS_ARGMIN` | `TS_ARGMAX(X, d)` | Days since rolling min |
| `TS_CORR` | `TS_CORR(X, Y, d)` | Rolling correlation; supports array Y |
| `TS_COVARIANCE` | `TS_COVARIANCE(X, Y, d)` | Rolling covariance |
| `TS_ZSCORE` | `TS_ZSCORE(X, d)` | Rolling z-score |
| `TS_MAD` | `TS_MAD(X, d)` | Rolling median absolute deviation |
| `TS_QUANTILE` | `TS_QUANTILE(X, d, q)` | Rolling quantile |
| `TS_PCTCHANGE` | `TS_PCTCHANGE(X, d)` | Rolling pct change (fills NaN with 0) |
| `TS_SKEW` | `TS_SKEW(X, d)` | Rolling skewness |
| `TS_KURT` | `TS_KURT(X, d)` | Rolling excess kurtosis |
| `SUMAC` | `SUMAC(X, d)` | Alias of rolling sum, p required int |
| `PROD` | `PROD(X, d)` | Rolling product |
| `DECAYLINEAR` | `DECAYLINEAR(X, d)` | Linear-decay weighted rolling average |
| `HIGHDAY` | `HIGHDAY(X, d)` | Days since rolling max within `d` |
| `LOWDAY` | `LOWDAY(X, d)` | Days since rolling min within `d` |
| `SLOPE` | `SLOPE(X, d)` | Rolling OLS slope against `1..d`; uses `min_periods=d` |
| `ATR` | `ATR(H, L, C, d)` | Average True Range (needs high, low, close) |

### Moving averages / smoothers

| Operator | Signature | Purpose |
|---|---|---|
| `SMA` | `SMA(X, m)` or `SMA(X, m, n)` | If `n` is None → rolling mean of window `m`; else EWM with `alpha = n/m` |
| `EMA` | `EMA(X, p)` | Exponential MA with `span=p` |
| `WMA` | `WMA(X, p)` | Geometric-decay weighted MA (`w_i = 0.9^i`) |
| `BB_MIDDLE` | `BB_MIDDLE(X, w)` | Bollinger middle band; supports dynamic `w` |
| `BB_UPPER` | `BB_UPPER(X, w)` | `BB_MIDDLE + std` |
| `BB_LOWER` | `BB_LOWER(X, w)` | `BB_MIDDLE - std` |
| `MACD` | `MACD(X, short=12, long=26)` | `EMA(X, short) - EMA(X, long)` |
| `RSI` | `RSI(X, w=14)` | Relative strength index |

### Cross-sectional (per datetime)

| Operator | Signature | Purpose |
|---|---|---|
| `RANK` | `RANK(X)` | Cross-sectional pct rank |
| `MEAN` | `MEAN(X)` | Cross-sectional mean |
| `STD` | `STD(X)` | Cross-sectional std |
| `SKEW` | `SKEW(X)` | Cross-sectional skewness |
| `KURT` | `KURT(X)` | Cross-sectional kurtosis |
| `MAX` | `MAX(X)` (one arg) vs `MAX(X, Y[, Z])` (pairwise max) | See **Note** below |
| `MIN` | same as MAX | Minimum version |
| `MEDIAN` | `MEDIAN(X)` | Cross-sectional median |
| `PERCENTILE` | `PERCENTILE(X, q[, p])` | Rolling quantile if `p`, else cross-sectional |
| `ZSCORE` | `ZSCORE(X)` | Cross-sectional z-score |
| `SCALE` | `SCALE(X, target_sum=1.0)` | Scale absolute values to sum `target_sum` |
| `INDUSTRY_NEUTRALIZE` | `INDUSTRY_NEUTRALIZE(X, $industry)` | Remove per-(datetime, industry) mean |

> **Note on `MAX`/`MIN`**: `function_lib.py` defines `MAX` and `MIN`
> **twice** — the **second** (non-decorated) definition shadows the
> first and takes 2-3 args, returning `np.maximum` / `np.minimum`. In
> practice this means you **can't** do cross-sectional `MAX($close)`
> the same way you do `RANK($close)`. If you need the cross-sectional
> maximum, call `TS_MAX` or use `MAX($x, 0)` for clipping.

### Math / element-wise

| Operator | Signature | Purpose |
|---|---|---|
| `ABS` | `ABS(X)` | Absolute value |
| `SIGN` | `SIGN(X)` | `-1/0/1` |
| `LOG` | `LOG(X)` | `log(X+1)` (offset by 1 for numerical stability) |
| `EXP` | `EXP(X)` | Exponential |
| `SQRT` | `SQRT(X)` | Square root |
| `POW` | `POW(X, n)` | `X^n` |
| `INV` | `INV(X)` | `1 / X` |
| `FLOOR` | `FLOOR(X)` | `floor(X)` |

### Logic / conditional helpers

| Operator | Signature | Purpose |
|---|---|---|
| `AND` | `AND(X, Y)` | Bitwise AND after casting to bool |
| `OR` | `OR(X, Y)` | Bitwise OR after casting to bool |
| `COUNT` | `COUNT(cond, d)` | Rolling count of true values |
| `SUMIF` | `SUMIF(X, d, cond)` | Rolling sum of `X * cond` |
| `FILTER` | `FILTER(X, cond)` | `X * cond` (zeroes out where `cond` false) |
| `SEQUENCE` | `SEQUENCE(n)` | Returns np.array `[1..n]` — use as the "X" for `TS_CORR`, `REGBETA` to do regressions against time |

### Regression

| Operator | Signature | Purpose |
|---|---|---|
| `REGBETA` | `REGBETA(Y, X, d)` | Rolling OLS slope (joblib-parallel per instrument) |
| `REGRESI` | `REGRESI(Y, X, d)` | Rolling OLS residual (latest value in the window) |

Both accept the second operand as a `pd.DataFrame/Series` (aligned by
index) **or** a `np.ndarray` (broadcast to the length of each window).

### Internal / non-DSL

`parse_arith_op` in the execution parser also inserts these helpers:

| Function | Purpose |
|---|---|
| `ADD(a, b)` | `np.add(a, b)` |
| `SUBTRACT(a, b)` | `np.subtract(a, b)` |
| `MULTIPLY(a, b)` | `np.multiply(a, b)` |
| `DIVIDE(a, b)` | `np.divide(a, b)` |

These are never written by users — they appear only in generated Python
code.

## Operators **referenced by paper seeds but not implemented**

These appear in `data/paper_seeds_converted.jsonl` and
`data/vn_seeds_300.jsonl`. Expressions that use them will fail during
`eval`, and the corresponding training examples will produce
`success=False` trajectories. If you need to unblock them, add them to
`function_lib.py` (see **Extending the library** below).

- `ZIGZAG_TOP`, `ZIGZAG_BOTTOM`, `ZIGZAG_TOP_DAYS`, `ZIGZAG_BOTTOM_DAYS`
- `BARSLAST`
- `WR` (Williams %R)
- `CCI` (Commodity Channel Index)
- `BBI` (Bull-Bear Index)
- `R_SQUARE`
- `RESI` (synonym for `REGRESI`)

## Extending the library

To add a new operator `MY_OP`:

1. Implement it in `function_lib.py` with the `@datatype_adapter`
   decorator if its primary dispatch is per-instrument or
   per-datetime. Respect the `(datetime, instrument)` MultiIndex
   convention and use `groupby('instrument')` / `groupby('datetime')`
   accordingly.
2. Because `factor_executor.py` automatically adds every public
   callable of `function_lib` into the `exec_namespace`, **no registry
   edit is needed** — your operator is immediately callable from DSL
   expressions.
3. Update `training/system_prompt.md` so the LLM knows the operator
   exists.
4. Add a unit test: evaluate an expression that uses `MY_OP` via
   `backtest/factor_executor.py::execute_expression` and check for
   `success=True`.
5. (Optional) if the operator is recognized as a **function** by the
   AST parser already (since `FunctionNode` is generic), no change is
   needed in `factor_ast.py`.

## Operator usage in seeds

A quick map of which operators dominate each seed group helps when
debugging or curating data:

- **Legacy `seed_factors.json`**: `RANK, TS_MEAN, TS_STD, TS_CORR,
  DELTA, SIGN` (simple momentum/vol factors).
- **Paper seeds (`vn_seeds_300.jsonl`)**: heavy use of conditional
  `cond ? value : nan`, `INDUSTRY_NEUTRALIZE`, `ATR`, `REGBETA`,
  `REGRESI`, `SLOPE`, `TS_SKEW/TS_KURT`, `BB_UPPER/LOWER/MIDDLE`,
  `MACD`, `RSI`, `DECAYLINEAR`, `COUNT`, `SUMIF`, `PERCENTILE`.
- **Generated programmatic factors** (from
  `training/generate_dataset.py::generate_programmatic_factors`): 10
  categories (momentum, mean-reversion, volatility, volume-price,
  foreign-flow, technical indicators, recency, regression, nonlinear
  transforms, composites) that together cover ~60% of the operator
  surface.

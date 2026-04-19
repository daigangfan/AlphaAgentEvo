# 02 · Data

The project has **three** conceptually distinct data assets:

1. The **market data panel** `daily_pv.h5` — the source of truth for
   backtests.
2. The **seed factor library** — JSON / JSONL files containing seed
   expressions (name, expression, baseline IR).
3. **Training-ready parquet files** — conversation prompts built from
   seeds that are fed into the GRPO trainer.

## 1. Market data — `backtest/data/daily_pv.h5`

A single HDF5 file holding a **MultiIndex pandas DataFrame**:

- **Shape**: 289 083 rows × 12 columns
- **Index**: `(datetime, instrument)` where `datetime` is a `Timestamp`
  (daily) and `instrument` is a ticker string (e.g. `ACB`, `VIC`,
  `VNM`).
- **Date range**: `2016-01-04` to `2026-02-13` (partially synthetic /
  forward-looking; see `STATUS.md`).
- **Instruments**: 219 unique tickers on HOSE.

### Columns

| Column | Meaning | Notes |
|---|---|---|
| `$open` | Daily opening price | |
| `$close` | Daily closing price | |
| `$high` | Intraday high | |
| `$low` | Intraday low | |
| `$volume` | Daily traded share volume | |
| `$return` | Daily return | Typically `DELTA(close)/DELAY(close,1)` |
| `$amount` | Daily traded value (≈ `volume × close`) | |
| `$net_foreign_val` | Net foreign trading value (VND) | Positive = foreign buying |
| `$net_foreign_vol` | Net foreign trading volume (shares) | |
| `$bench_close` | VNINDEX close | Broadcast identically across instruments; the benchmark |
| `$bench_return` | VNINDEX daily return | Used for excess-return / IR computation |
| `$industry` | HOSE sector string | Used by `INDUSTRY_NEUTRALIZE` |

Note the `$` prefix — it is part of the DSL identifier (see
`03_expression_language.md`). Inside `factor_executor.py` the `$` is
stripped before executing in Python, and names that collide with Python
keywords are prefixed with `col_` (e.g. `$return` → `col_return`).

### Loading

Read once, cached in memory by `backtest/factor_executor.py`:

```python path=F:\projects\AlphaAgentEvo\backtest\factor_executor.py start=83
def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load daily_pv.h5 into memory (cached on first call)."""
```

The API server calls `load_data()` at startup. Each individual column is
pre-extracted into a dict `_cached_columns[col]` for fast lookup during
expression execution.

### Temporal splits

Configured in `configs/grpo_config.yaml` (`backtest.periods`) and in
`deploy/api_server_verl.py` (hardcoded). Default splits:

| Period | Start | End |
|---|---|---|
| train | 2016-01-01 | 2023-12-31 |
| val | 2024-01-01 | 2024-12-31 |
| test | 2025-01-01 | 2026-12-31 |

Factor values are always computed on **all** data (TS_* operators need
lookback window). Only the **IC/IR metric** is computed on the
requested period. Masks per period are built once at startup.

## 2. Seed factor library

Seed factors are domain-expert-curated alpha expressions along with
baseline metrics. They seed the LLM's "evolution" process.

### `data/seed_factors.json`

Small hand-curated list (3 factors) used as the *base* seed pool for the
legacy `training/generate_dataset.py` augmentation pipeline.

Schema per entry:

```json path=null start=null
{
  "name": "Adaptive_Momentum_Volatility_Ratio_20D",
  "expression": "RANK(TS_MEAN($return, 5) / (TS_STD($return, 20) + 1e-8)) * SIGN(DELTA($close, 5) / ($close + 1e-8))",
  "hypothesis": "Short-term momentum normalized by longer-term volatility ...",
  "ic": 0.042,
  "ir": 1.15,
  "ar": 0.128,
  "max_dd": -0.085,
  "source_run": "run4",
  "round": 7
}
```

### `data/seed_factors.jsonl`

Larger (389 entries) JSON-lines file produced by augmentation
(`generate_programmatic_factors()` +
`generate_mutations()` in `training/generate_dataset.py`). Same schema
as above plus `source` column ("seed" | "mutation_*" | "programmatic").

### `data/vn_seeds_300.jsonl`, `val_seeds.jsonl`, `test_seeds.jsonl`

Vietnam-specific seed splits used for the **Verl / paper-parity**
pipeline:

- `vn_seeds_300.jsonl` — 300 seeds, 67% conditional
  (`(cond)? value : nan` form), matching the paper's 3-layer signal
  architecture.
- `val_seeds.jsonl` — 30 seeds.
- `test_seeds.jsonl` — 99 seeds.

Schema per entry:

```json path=null start=null
{"name": "LIQ_NET_FLOW_PER_RANGE",
 "expr": "(((TS_SUM($net_foreign_val,5) > 0) & ...) ? RANK(INDUSTRY_NEUTRALIZE(...)) : nan)"}
```

Note the different key (`expr` here, `expression` in the legacy JSON).
These seeds also use a **richer operator set** (`ZIGZAG_*`, `BARSLAST`,
`WR`, `CCI`, `R_SQUARE`, `BBI`, `RESI`) — several of which are **not
implemented** in `function_lib.py` yet. Running one of those through the
backtester will fail with an exception (reported as `error` in the API
response). This is known and documented in the paper-seeds file, which
mixes "already working" seeds and "unsupported operator" seeds.

### `data/paper_seeds_converted.jsonl`

The first 270 seeds translated from the paper's reference implementation.

### `data/seed_backtest_results.json`

Pre-computed IR for every seed, used to populate the `seed_ir` field of
the training parquet without calling the backtest API each time.

### `data/sector_map.json`

Ticker → HOSE sector mapping (476 tickers across 19 sectors). Useful if
you regenerate `daily_pv.h5` or want to validate `$industry`.

## 3. Training parquet

The training loop consumes parquet files produced by
`training/generate_dataset.py` (local `trl` track) or by the paper's
toolchain (Verl track).

### Location

- Local: `data/train.parquet`, `data/val.parquet`, `data/test.parquet`.
- Verl: the same files but also copied into `deploy/v2/`.

### Schema

| Column | Type | Description |
|---|---|---|
| `prompt` | `list[dict]` | Conversation prefix: `[{"role": "system", "content": ...}, {"role": "user", "content": "Evolve this seed..."}]` |
| `seed_expr` | `str` | The seed expression — passed to the reward function |
| `seed_ir` | `float` | Baseline IR — passed to the reward function |
| `seed_name` | `str` | Human-friendly seed name |

The **system** message is the contents of
`training/system_prompt.md` (operator reference, strategy hints,
`/no_think` directive). The **user** message is built by
`build_conversation()` in `training/generate_dataset.py` and follows the
template:

```
Evolve this seed alpha factor to achieve a higher IR.

Factor: <name>
Expression: <expression>
Hypothesis: <hypothesis>
Baseline IR: <ir>

Evaluate the seed first, then try at least 2 variations. Report the best result.
```

The trainer converts `prompt` into a chat-formatted token sequence using
the model's `apply_chat_template`; tools are attached via the `tools=`
argument of `GRPOTrainer` (local) or the Hydra config (Verl).

### Regenerating the dataset

```bash path=null start=null
# Local track
bash start_api.sh           # keep running in one terminal
python training/generate_dataset.py --augment --evaluate \
    --backtest-url http://localhost:8001 \
    --output-dir data
```

Flags:

- `--augment` — also include mutations (window tweaks, RANK/ZSCORE
  wrappers, variable swaps, additive/multiplicative combinations) and
  programmatic factors (10 categories, ~150 factors). Without this
  flag, only `seed_factors.json` is used (very small).
- `--evaluate` — call the backtest API to overwrite `ir` with the real
  value. Expensive — ~1 minute / factor serially.
- `--output-dir data` — where to write `train.parquet`, `val.parquet`,
  and `seed_factors.jsonl`.

### Known issue: parquet version skew

The current parquet files in `data/` were produced by an older pandas /
pyarrow and may fail to read on newer pyarrow with a "Repetition level
histogram size mismatch" error. If you hit this, **regenerate** with
the current environment or read the equivalent JSONL files. See
`10_AGENT_GUIDE.md` §"Known traps" for a workaround.

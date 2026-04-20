"""Tests for backtest/factor_executor.py — specifically the ic_mean/ic_std bug fix.

Bug: execute_expression() success-path return dict was missing ic_mean and ic_std,
causing pydantic_core.ValidationError when EvaluateResponse(**result) was called
in api_server.py.

Regression tests verify:
1. _compute_ic() returns valid (float, float) from synthetic data.
2. _compute_ic() degrades gracefully when $close is absent or data is sparse.
3. EvaluateResponse pydantic model requires ic_mean and ic_std.
4. execute_expression() always returns a dict containing both fields.
5. The dict produced by execute_expression() passes EvaluateResponse validation.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.factor_executor import _compute_ic
from backtest.api_server import EvaluateResponse


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_df(n_dates: int = 60, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic daily_pv multi-index DataFrame for testing.

    Index: (datetime, instrument)  — matching the real data layout.
    Columns: $close, $open, $return, $bench_return
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    stocks = [f"VN{i:03d}" for i in range(n_stocks)]

    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    n = len(idx)

    close = rng.uniform(10, 100, n)
    open_ = close * rng.uniform(0.995, 1.005, n)
    ret = rng.normal(0, 0.01, n)
    bench = rng.normal(0, 0.005, n_dates)
    bench_rep = np.repeat(bench, n_stocks)

    return pd.DataFrame(
        {
            "$close": close,
            "$open": open_,
            "$return": ret,
            "$bench_return": bench_rep,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests for _compute_ic
# ---------------------------------------------------------------------------

class TestComputeIC:
    def test_returns_float_tuple(self):
        """_compute_ic should return a (float, float) with valid data."""
        df = _make_df()
        factor = df["$close"]
        ic_mean, ic_std = _compute_ic(factor, df)
        assert isinstance(ic_mean, float)
        assert isinstance(ic_std, float)
        assert not np.isnan(ic_mean)
        assert not np.isnan(ic_std)

    def test_ic_range(self):
        """IC values must be in [-1, 1] and std must be non-negative."""
        df = _make_df()
        factor = df["$close"]
        ic_mean, ic_std = _compute_ic(factor, df)
        assert -1.0 <= ic_mean <= 1.0
        assert ic_std >= 0.0

    def test_missing_close_column_returns_zeros(self):
        """Returns (0.0, 0.0) when the DataFrame has no $close column."""
        df = _make_df().drop(columns=["$close"])
        factor = df["$return"]
        ic_mean, ic_std = _compute_ic(factor, df)
        assert ic_mean == 0.0
        assert ic_std == 0.0

    def test_period_filter_start(self):
        """start_date filter should restrict IC computation."""
        df = _make_df(n_dates=100)
        factor = df["$close"]

        ic_full_mean, _ = _compute_ic(factor, df)
        ic_half_mean, _ = _compute_ic(
            factor, df, start_date="2020-06-01", end_date="2020-09-30"
        )
        # Both must be valid floats (period filter shouldn't crash)
        assert isinstance(ic_full_mean, float)
        assert isinstance(ic_half_mean, float)

    def test_insufficient_stocks_returns_zeros(self):
        """Returns (0.0, 0.0) when fewer than 5 stocks have data on each date."""
        df = _make_df(n_dates=30, n_stocks=3)  # only 3 stocks — below threshold
        factor = df["$close"]
        ic_mean, ic_std = _compute_ic(factor, df)
        assert ic_mean == 0.0
        assert ic_std == 0.0

    def test_single_date_std_is_zero(self):
        """With only one valid date, ic_std should be 0.0."""
        df = _make_df(n_dates=2, n_stocks=20)  # only 1 date gets forward return
        factor = df["$close"]
        _, ic_std = _compute_ic(factor, df)
        assert ic_std == 0.0


# ---------------------------------------------------------------------------
# Tests for EvaluateResponse pydantic model
# ---------------------------------------------------------------------------

class TestEvaluateResponseSchema:
    def test_validation_fails_without_ic_mean(self):
        """EvaluateResponse must raise if ic_mean is missing."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ic_mean"):
            EvaluateResponse(
                success=True,
                score=1.23,
                ic_std=0.02,
                ir=1.23,
                error=None,
                exec_time=0.5,
            )

    def test_validation_fails_without_ic_std(self):
        """EvaluateResponse must raise if ic_std is missing."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="ic_std"):
            EvaluateResponse(
                success=True,
                score=1.23,
                ic_mean=0.05,
                ir=1.23,
                error=None,
                exec_time=0.5,
            )

    def test_validation_succeeds_with_both_ic_fields(self):
        """EvaluateResponse should validate when both ic fields are present."""
        resp = EvaluateResponse(
            success=True,
            score=1.23,
            ic_mean=0.05,
            ic_std=0.02,
            ir=1.23,
            error=None,
            exec_time=0.5,
        )
        assert resp.ic_mean == pytest.approx(0.05)
        assert resp.ic_std == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Integration: execute_expression always returns ic_mean / ic_std
# ---------------------------------------------------------------------------

class TestExecuteExpressionICFields:
    """Test that execute_expression() always returns ic_mean and ic_std.

    We patch the module-level cache so no real HDF5 file is needed.
    """

    @staticmethod
    def _patch_cache(df: pd.DataFrame):
        """Return a dict of patches for the executor module globals."""
        import backtest.factor_executor as fe

        return {
            "_cached_df": df,
            "_cached_columns": {col: df[col] for col in df.columns},
            "_period_ranges": {},
            "_period_masks": {},
        }

    def test_success_path_has_ic_keys(self):
        """execute_expression success result must contain ic_mean and ic_std."""
        import backtest.factor_executor as fe

        df = _make_df()
        patches = self._patch_cache(df)

        with (
            patch.object(fe, "_cached_df", patches["_cached_df"]),
            patch.object(fe, "_cached_columns", patches["_cached_columns"]),
            patch.object(fe, "_period_ranges", patches["_period_ranges"]),
            patch.object(fe, "_period_masks", patches["_period_masks"]),
        ):
            result = fe.execute_expression("$close")

        assert "ic_mean" in result, "ic_mean missing from execute_expression result"
        assert "ic_std" in result, "ic_std missing from execute_expression result"
        assert isinstance(result["ic_mean"], float)
        assert isinstance(result["ic_std"], float)

    def test_result_validates_as_evaluate_response(self):
        """execute_expression output must pass EvaluateResponse pydantic validation."""
        import backtest.factor_executor as fe

        df = _make_df()
        patches = self._patch_cache(df)

        with (
            patch.object(fe, "_cached_df", patches["_cached_df"]),
            patch.object(fe, "_cached_columns", patches["_cached_columns"]),
            patch.object(fe, "_period_ranges", patches["_period_ranges"]),
            patch.object(fe, "_period_masks", patches["_period_masks"]),
        ):
            result = fe.execute_expression("$close")

        # This is the exact line that was raising ValidationError before the fix
        resp = EvaluateResponse(**result)
        assert resp is not None

    def test_error_path_has_ic_keys(self):
        """execute_expression error (exception) result must also contain ic fields."""
        import backtest.factor_executor as fe

        df = _make_df()
        patches = self._patch_cache(df)

        with (
            patch.object(fe, "_cached_df", patches["_cached_df"]),
            patch.object(fe, "_cached_columns", patches["_cached_columns"]),
            patch.object(fe, "_period_ranges", patches["_period_ranges"]),
            patch.object(fe, "_period_masks", patches["_period_masks"]),
        ):
            # Deliberately invalid expression triggers the except branch
            result = fe.execute_expression("INVALID_FUNC_THAT_DOES_NOT_EXIST($close)")

        assert "ic_mean" in result
        assert "ic_std" in result
        assert result["success"] is False

    def test_result_with_period_validates(self):
        """execute_expression with a named period still validates correctly."""
        import backtest.factor_executor as fe

        df = _make_df(n_dates=80)
        patches = self._patch_cache(df)
        period_ranges = {"train": ("2020-01-01", "2020-06-30")}

        with (
            patch.object(fe, "_cached_df", patches["_cached_df"]),
            patch.object(fe, "_cached_columns", patches["_cached_columns"]),
            patch.object(fe, "_period_ranges", period_ranges),
            patch.object(fe, "_period_masks", patches["_period_masks"]),
        ):
            result = fe.execute_expression("$close", period="train")

        resp = EvaluateResponse(**result)
        assert resp is not None
        assert isinstance(resp.ic_mean, float)
        assert isinstance(resp.ic_std, float)

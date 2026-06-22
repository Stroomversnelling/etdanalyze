"""Cross-path schema parity test (parent ADR-018).

Asserts that ``etdanalyze.compute_stats_table`` (the per-group vectorised
path) emits a DataFrame whose stat columns are a superset of
``etdmap.NUMERIC_STATS_SCHEMA`` and whose values match the kernel
``etdmap.compute_numeric_column_stats`` for a single-group input.

This test cannot live in etdmap (per ADR-004 etdmap must not import from
etdanalyze). It lives here, in etdanalyze tests, because etdanalyze can
freely import from etdmap.
"""

import pandas as pd
import pytest

import ibis

from etdmap.data_stats import (
    NUMERIC_STATS_SCHEMA,
    compute_numeric_column_stats,
)
from etdanalyze.analysis_helpers import compute_stats_table, compute_stats_table_lazy


def _fixture_df():
    """One group ('A'), one variable, deterministic Float64 values.

    A single group keeps the parity comparison simple: per-group stats
    on group 'A' must equal the kernel's stats on the same series.
    """
    return pd.DataFrame({
        "g": ["A"] * 5,
        "x": pd.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="Float64"),
    })


class TestComputeStatsTableSchemaParity:

    def test_output_columns_are_superset_of_schema(self):
        """compute_stats_table emits at least every name in
        NUMERIC_STATS_SCHEMA. It may add extras (variable, group keys)
        -- the contract is 'at least these'."""
        out = compute_stats_table(_fixture_df(), ["g"], ["x"])
        missing = set(NUMERIC_STATS_SCHEMA) - set(out.columns)
        assert not missing, (
            f"compute_stats_table missing schema columns: {sorted(missing)}"
        )

    def test_per_group_values_match_kernel_for_single_group(self):
        """For a single-group input, the per-group stats row equals the
        kernel's stats on the underlying series. Drift here means the
        two paths have diverged and the project's schema contract
        (ADR-018) is broken."""
        df = _fixture_df()
        kernel = compute_numeric_column_stats(df["x"])
        out = compute_stats_table(df, ["g"], ["x"])
        assert len(out) == 1, "single group should yield one row"
        row = out.iloc[0]
        for key in NUMERIC_STATS_SCHEMA:
            kv = kernel[key]
            tv = row[key]
            if pd.isna(kv) and pd.isna(tv):
                continue
            assert float(tv) == pytest.approx(float(kv)), (
                f"compute_stats_table value for '{key}' diverged from "
                f"kernel: table={tv}, kernel={kv}"
            )


class TestComputeStatsTableLazyParity:
    """compute_stats_table_lazy (ibis backend) must match the pandas-eager
    compute_stats_table on the same input. Lazy validation brief on
    real project data showed agreement to 1e-9; this test enforces it on
    a small fixture for CI."""

    def test_output_columns_match_eager(self):
        """Same column set as the pandas-eager version."""
        df = _fixture_df()
        eager = compute_stats_table(df, ["g"], ["x"])
        lazy = compute_stats_table_lazy(ibis.memtable(df), ["g"], ["x"])
        # Same stat columns, same group cols, same variable col.
        assert set(eager.columns) == set(lazy.columns), (
            f"column-set drift: eager - lazy = {set(eager.columns) - set(lazy.columns)}, "
            f"lazy - eager = {set(lazy.columns) - set(eager.columns)}"
        )

    def test_values_match_eager_within_tolerance(self):
        """Numerical parity within 1e-9 across all NUMERIC_STATS_SCHEMA columns."""
        df = _fixture_df()
        eager = compute_stats_table(df, ["g"], ["x"]).set_index(["variable", "g"])
        lazy = compute_stats_table_lazy(ibis.memtable(df), ["g"], ["x"]).set_index(["variable", "g"])
        assert eager.index.equals(lazy.index), "group keys differ"
        for key in NUMERIC_STATS_SCHEMA:
            ev, lv = float(eager[key].iloc[0]), float(lazy[key].iloc[0])
            assert lv == pytest.approx(ev, rel=1e-9, abs=1e-9), (
                f"compute_stats_table_lazy diverged from eager on '{key}': "
                f"eager={ev}, lazy={lv}"
            )

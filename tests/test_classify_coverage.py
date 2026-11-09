"""Tests for etdanalyze.analysis_helpers.classify_coverage.

The strict empty-data contract relies on this helper to produce a
per-(variable, group) coverage row so plot callers can skip zero-row
combinations upstream. Tests cover: default groups, custom groups,
n_rows / n_nonnull counting, is_empty and is_constant flags,
column-missing errors.
"""
import pandas as pd
import pytest

from etdanalyze.analysis_helpers import classify_coverage


def _frame():
    """Two projects (1, 2) x two woning types (A, B). Project 2 has
    NA for var_x; project 1 has constant var_y. var_z varies across
    rows in both."""
    return pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "WoningType":   ["A", "A", "B", "A", "B", "B"],
        "var_x":        pd.array(
            [1.0, 2.0, 3.0, pd.NA, pd.NA, pd.NA], dtype="Float64",
        ),
        "var_y":        pd.array([5.0, 5.0, 5.0, 7.0, 8.0, 9.0],
                                 dtype="Float64"),
        "var_z":        pd.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                                 dtype="Float64"),
    })


class TestClassifyCoverage:
    def test_default_group_cols(self):
        df = _frame()
        out = classify_coverage(df, ["var_x"])
        # Default groups = ["ProjectIdBSV", "WoningType"]
        assert {"ProjectIdBSV", "WoningType"} <= set(out.columns)
        # 3 groups: (1,A), (1,B), (2,A), (2,B) -> 4 actually
        assert len(out) == 4

    def test_n_rows_and_n_nonnull(self):
        df = _frame()
        out = classify_coverage(df, ["var_x"])
        proj1_a = out[(out["ProjectIdBSV"] == 1) & (out["WoningType"] == "A")]
        assert int(proj1_a["n_rows"].iloc[0]) == 2
        assert int(proj1_a["n_nonnull"].iloc[0]) == 2
        proj2_a = out[(out["ProjectIdBSV"] == 2) & (out["WoningType"] == "A")]
        # var_x is NA in project 2
        assert int(proj2_a["n_rows"].iloc[0]) == 1
        assert int(proj2_a["n_nonnull"].iloc[0]) == 0
        assert bool(proj2_a["is_empty"].iloc[0]) is True

    def test_is_constant_flag(self):
        df = _frame()
        out = classify_coverage(df, ["var_y"])
        # Project 1 has var_y == 5.0 across all rows -> constant
        proj1_a = out[(out["ProjectIdBSV"] == 1) & (out["WoningType"] == "A")]
        assert bool(proj1_a["is_constant"].iloc[0]) is True
        # Project 2 var_y is 7, 8, 9 -> not constant
        proj2_b = out[(out["ProjectIdBSV"] == 2) & (out["WoningType"] == "B")]
        assert bool(proj2_b["is_constant"].iloc[0]) is False

    def test_is_empty_not_flagged_when_data_present(self):
        df = _frame()
        out = classify_coverage(df, ["var_z"])
        # var_z is fully populated -> no group should be empty
        assert (~out["is_empty"]).all()

    def test_custom_group_cols(self):
        # Group by ProjectIdBSV only (drop WoningType)
        df = _frame()
        out = classify_coverage(df, ["var_x"], group_cols=["ProjectIdBSV"])
        assert "WoningType" not in out.columns
        # 2 projects -> 2 rows
        assert len(out) == 2
        proj2 = out[out["ProjectIdBSV"] == 2]
        # All 3 rows for project 2 have NA in var_x
        assert int(proj2["n_rows"].iloc[0]) == 3
        assert int(proj2["n_nonnull"].iloc[0]) == 0

    def test_multiple_variables_at_once(self):
        df = _frame()
        out = classify_coverage(df, ["var_x", "var_y", "var_z"])
        # 4 (project, woning) groups x 3 variables = 12 rows
        assert len(out) == 12
        assert set(out["variable"].unique()) == {"var_x", "var_y", "var_z"}

    def test_missing_column_raises(self):
        df = _frame()
        with pytest.raises(KeyError, match="not in df"):
            classify_coverage(df, ["nonexistent_var"])

    def test_missing_group_col_raises(self):
        df = _frame()
        with pytest.raises(KeyError, match="not in df"):
            classify_coverage(df, ["var_x"], group_cols=["NoSuchCol"])

    def test_empty_value_cols_returns_empty_frame(self):
        df = _frame()
        out = classify_coverage(df, [])
        # Empty value_cols -> empty frame with the expected schema
        assert list(out.columns) == [
            "variable", "ProjectIdBSV", "WoningType",
            "n_rows", "n_nonnull", "is_empty", "is_constant",
        ]
        assert len(out) == 0

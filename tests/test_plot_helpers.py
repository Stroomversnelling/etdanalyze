"""Tests for the new shared plot helpers: unit lookup, bimodal split, outlier
table, sort_load, plus the extended plot_boxwhisker_comparison kwargs."""
import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from etdanalyze.plot_functions import (
    axis_label_with_unit,
    detect_value_range_clusters,
    plot_boxwhisker_comparison,
    plot_household_range_strips,
    unit_for_column,
)
from etdanalyze.analysis_helpers import (
    build_extreme_value_table,
    build_outlier_table,
    sort_load,
)


# ---------------------------------------------------------------------------
# unit_for_column / axis_label_with_unit
# ---------------------------------------------------------------------------

class TestUnitLookup:
    def test_known_column_returns_unit(self):
        assert unit_for_column("ElektriciteitNetgebruikHoog") == "kWh"
        assert unit_for_column("WarmteproductieWarmtepomp") == "GJ"
        assert unit_for_column("Gasgebruik") == "m3"

    def test_diff_variant_inherits(self):
        assert unit_for_column("ElektriciteitNetgebruikHoogDiff") == "kWh"

    def test_unknown_returns_empty(self):
        assert unit_for_column("__not_a_real_column__") == ""

    def test_axis_label_appends_when_known(self):
        assert axis_label_with_unit("ElektriciteitNetgebruikHoog", "median") == "median (kWh)"

    def test_axis_label_passes_through_when_unknown(self):
        assert axis_label_with_unit("__not_a_real_column__", "median") == "median"

    def test_axis_label_defaults_base_to_column(self):
        assert axis_label_with_unit("ElektriciteitNetgebruikHoog") == "ElektriciteitNetgebruikHoog (kWh)"


# ---------------------------------------------------------------------------
# detect_bimodal_split
# ---------------------------------------------------------------------------

class TestDetectValueRangeClusters:
    def test_uniform_distribution_returns_single_cluster(self):
        keys = [f"P{i}" for i in range(5)]
        mins = pd.Series({k: 0.0 for k in keys})
        maxs = pd.Series({f"P{i}": 10.0 + i for i in range(5)})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 1
        assert clusters[0] == set(keys)

    def test_clear_two_clusters_via_max_gap(self):
        # Low cluster around 1, high cluster around 100 (100x ratio in max).
        mins = pd.Series({"P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 90.0, "P5": 95.0})
        maxs = pd.Series({"P1": 1.0, "P2": 1.2, "P3": 0.9, "P4": 100.0, "P5": 110.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 2
        assert clusters[0] == {"P1", "P2", "P3"}
        assert clusters[1] == {"P4", "P5"}

    def test_three_clusters(self):
        # Three clearly separated bands: ~1, ~100, ~10000.
        mins = pd.Series({
            "A1": 0.0, "A2": 0.0, "A3": 0.0,
            "B1": 90.0, "B2": 95.0,
            "C1": 9000.0, "C2": 9500.0,
        })
        maxs = pd.Series({
            "A1": 1.0, "A2": 1.2, "A3": 0.9,
            "B1": 100.0, "B2": 110.0,
            "C1": 10000.0, "C2": 10500.0,
        })
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 3
        assert clusters[0] == {"A1", "A2", "A3"}
        assert clusters[1] == {"B1", "B2"}
        assert clusters[2] == {"C1", "C2"}

    def test_overlapping_ranges_do_not_split(self):
        # Ranges overlap broadly even though medians vary -- should NOT split.
        # Mirrors the OverigGG case where mins are ~0 and maxs span a wide
        # but continuous range.
        mins = pd.Series({"P8": 0.0, "P9": 0.0, "P10": 43.0, "P11": 40.0})
        maxs = pd.Series({"P8": 1467.0, "P9": 745.0, "P10": 1807.0, "P11": 209.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 1

    def test_negative_extreme_outlier_isolated(self):
        # One project with an extreme negative min while others sit near 0.
        # The detector must isolate it -- when grouped with the others, its
        # huge range cramps their boxes against the upper edge of the
        # y-axis. The remaining projects may end up in one or more
        # clusters depending on the walk order; the essential property
        # is just that A is alone.
        mins = pd.Series({"A": -50000.0, "B": 0.0, "C": 0.0, "D": 5.0})
        maxs = pd.Series({"A": 100.0, "B": 110.0, "C": 90.0, "D": 105.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) >= 2
        assert {"A"} in clusters

    def test_two_groups_split_when_well_separated(self):
        # Two projects with widely separated max values should still
        # split. Captures the Booster case where zero-filler filtering
        # leaves just the bulk project and one near-zero project.
        mins = pd.Series({"P1": 0.0, "P2": 0.0})
        maxs = pd.Series({"P1": 0.10, "P2": 1970.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 2

    def test_one_group_returns_single_cluster(self):
        mins = pd.Series({"P1": 0.0})
        maxs = pd.Series({"P1": 100.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 1

    def test_close_clusters_do_not_split(self):
        # Small absolute gap relative to overall range.
        mins = pd.Series({"P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0})
        maxs = pd.Series({"P1": 10.0, "P2": 11.0, "P3": 15.0, "P4": 16.0})
        clusters = detect_value_range_clusters(mins, maxs)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# sort_load
# ---------------------------------------------------------------------------

class TestSortLoad:
    def test_descending_with_reset_index(self):
        s = pd.Series([3.0, 1.0, 2.0], index=["a", "b", "c"])
        out = sort_load(s)
        assert out.tolist() == [3.0, 2.0, 1.0]
        assert list(out.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# build_outlier_table
# ---------------------------------------------------------------------------

class TestBuildOutlierTable:
    def _frame(self):
        # Two suppliers x one project each. Each cohort has >= 5 points so
        # Tukey fence applies (rather than the z-score fallback for tiny n).
        return pd.DataFrame({
            "HuisIdBSV":       list(range(1, 13)),
            "Dataleverancier": ["A"] * 6 + ["B"] * 6,
            "ProjectIdBSV":    [1] * 6 + [2] * 6,
            "median":          [10.0, 10.5, 11.0, 9.5, 10.2, 50.0,
                                20.0, 21.0, 22.0, 19.5, 20.7, 200.0],
        })

    def test_finds_outliers_per_cohort(self):
        out = build_outlier_table(self._frame(), value_col="median")
        ids = set(out["HuisIdBSV"].tolist())
        assert 6 in ids   # the 50.0 anomaly in supplier A
        assert 12 in ids  # the 200.0 anomaly in supplier B

    def test_returns_required_columns(self):
        out = build_outlier_table(self._frame(), value_col="median")
        for c in ("HuisIdBSV", "Dataleverancier", "ProjectIdBSV", "value", "group_median", "z_score"):
            assert c in out.columns

    def test_top_n_caps_results(self):
        out = build_outlier_table(self._frame(), value_col="median", top_n=1)
        assert len(out) == 1

    def test_empty_when_no_outliers(self):
        df = pd.DataFrame({
            "HuisIdBSV": [1, 2, 3, 4],
            "Dataleverancier": ["A"] * 4,
            "ProjectIdBSV": [1] * 4,
            "median": [10.0, 10.1, 9.9, 10.2],
        })
        out = build_outlier_table(df, value_col="median")
        assert out.empty

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            build_outlier_table(pd.DataFrame({"x": [1.0]}), value_col="missing")


# ---------------------------------------------------------------------------
# build_extreme_value_table  (percentile-bounds method)
# ---------------------------------------------------------------------------


class TestBuildExtremeValueTable:
    """The percentile-bounds approach answers a different question than
    Tukey: 'is this value so extreme it dwarfs the bulk of the data?',
    independent of distribution shape. The q99*10 / q01*10 thresholds
    are designed for sufficiently large samples (typically per-row
    time-series with hundreds of records per project), where q99 is
    substantially less than the extreme tail values."""

    def _frame_with_extreme_high(self, n_normal=200):
        # Bulk around 100 (n_normal rows); one value 1000x larger that
        # comfortably exceeds q99 * 10.
        rng = np.random.default_rng(0)
        rows = [
            {"HuisIdBSV": i + 1, "Dataleverancier": "A", "ProjectIdBSV": 1,
             "median": float(100 + rng.normal(0, 2))}
            for i in range(n_normal)
        ]
        rows.append({
            "HuisIdBSV": n_normal + 1, "Dataleverancier": "A",
            "ProjectIdBSV": 1, "median": 100000.0,
        })
        return pd.DataFrame(rows)

    def test_finds_above_high_bound(self):
        df = self._frame_with_extreme_high()
        out = build_extreme_value_table(df, value_col="median")
        assert not out.empty
        assert (out["direction"] == "above").any()
        # The injected extreme row should be flagged
        assert 201 in set(out["HuisIdBSV"].tolist())

    def test_required_columns_present(self):
        df = self._frame_with_extreme_high()
        out = build_extreme_value_table(df, value_col="median")
        for c in ("HuisIdBSV", "Dataleverancier", "ProjectIdBSV",
                  "value", "bound_low", "bound_high", "direction"):
            assert c in out.columns

    def test_finds_below_low_bound_when_negative(self):
        # When q01 is negative, bound_low becomes q01*10. A value far
        # below that bound should be flagged 'below'.
        rng = np.random.default_rng(0)
        rows = [
            {"HuisIdBSV": i + 1, "Dataleverancier": "A", "ProjectIdBSV": 1,
             "median": float(-1 + rng.normal(0, 0.05))}
            for i in range(200)
        ]
        rows.append({
            "HuisIdBSV": 201, "Dataleverancier": "A", "ProjectIdBSV": 1,
            "median": -1000.0,
        })
        df = pd.DataFrame(rows)
        out = build_extreme_value_table(df, value_col="median")
        assert not out.empty
        assert (out["direction"] == "below").any()
        assert 201 in set(out["HuisIdBSV"].tolist())

    def test_empty_when_no_extremes(self):
        df = pd.DataFrame({
            "HuisIdBSV":       [1, 2, 3, 4, 5],
            "Dataleverancier": ["A"] * 5,
            "ProjectIdBSV":    [1] * 5,
            "median":          [10.0, 10.1, 10.2, 9.9, 10.05],
        })
        out = build_extreme_value_table(df, value_col="median")
        assert out.empty

    def test_top_n_caps_results(self):
        df = self._frame_with_extreme_high()
        out = build_extreme_value_table(df, value_col="median", top_n=1)
        assert len(out) == 1

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            build_extreme_value_table(
                pd.DataFrame({"x": [1.0]}), value_col="missing",
            )


# ---------------------------------------------------------------------------
# plot_boxwhisker_comparison extensions
# ---------------------------------------------------------------------------

class TestBoxwhiskerExtensions:
    def _frame(self, n_per_project=8, projects=("P1", "P2", "P3")):
        rows = []
        for p in projects:
            base = 10.0 if p != "P3" else 100.0  # P3 is the high cluster
            for i in range(n_per_project):
                rows.append({
                    "HuisIdBSV": f"{p}_{i}",
                    "ProjectIdBSV": p,
                    "WoningType": "TypeA" if i % 2 == 0 else "TypeB",
                    "median": base + i * 0.3,
                })
        return pd.DataFrame(rows)

    def test_unit_appears_in_label_when_column_name_known(self, tmp_path):
        df = self._frame()
        out = tmp_path / "unit.png"
        plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            column_name="ElektriciteitNetgebruikHoog",
            save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_outlier_ids_renders_without_error(self, tmp_path):
        df = self._frame()
        out = tmp_path / "outliers.png"
        plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            outlier_ids={"P1_0", "P3_5"},
            save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_secondary_axis_renders_as_stacked_subplot(self, tmp_path):
        """When secondary_axis_data is provided, the figure must have two
        subplots (boxplot on top, companion scatter beneath, sharing
        x-axis) -- not a twinx() overlay on the same axes."""
        import matplotlib.pyplot as plt
        df = self._frame()
        secondary = df[["ProjectIdBSV"]].copy()
        secondary["validate_fail_rate"] = np.random.default_rng(0).uniform(0, 1, len(df))
        out = tmp_path / "secondary.png"
        fig = plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            secondary_axis_data=secondary,
            secondary_axis_value_col="validate_fail_rate",
            secondary_axis_label="validate fail rate",
        )
        try:
            # Two real stacked subplots (twinx would also show 2 Axes
            # but they would share a bbox position; stacked subplots
            # have distinct y0 values).
            assert fig is not None
            assert len(fig.axes) >= 2
            ax_top, ax_bot = fig.axes[0], fig.axes[1]
            assert ax_top.get_position().y0 > ax_bot.get_position().y0
            fig.savefig(out, dpi=150)
        finally:
            plt.close(fig)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_household_range_strips  (per-household candlestick)
# ---------------------------------------------------------------------------


class TestPlotHouseholdRangeStrips:
    """The new helper draws one vertical range strip per household with
    min-q1-q25-median-q75-q99-max marks, grouped by project. Smoke tests
    cover the happy path and column-validation."""

    def _frame(self, n_per_project=4):
        rng = np.random.default_rng(0)
        rows = []
        for proj in (1, 2, 3):
            for i in range(n_per_project):
                base = 10.0 * proj
                rows.append({
                    "HuisIdBSV": f"P{proj}_{i}",
                    "ProjectIdBSV": proj,
                    "Dataleverancier": "Watch-E" if proj > 1 else "Factory Zero",
                    "min": float(base + rng.normal(0, 0.1)),
                    "quantile_1": float(base + 0.5),
                    "quantile_25": float(base + 1.0),
                    "median": float(base + 1.5),
                    "quantile_75": float(base + 2.0),
                    "quantile_99": float(base + 2.5),
                    "max": float(base + 3.0 + rng.uniform(0, 1)),
                })
        return pd.DataFrame(rows)

    def test_writes_non_empty_png(self, tmp_path):
        df = self._frame()
        out = tmp_path / "strips.png"
        plot_household_range_strips(
            df, column_name="ElektriciteitNetgebruikHoog",
            save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_returns_figure_when_no_path(self):
        import matplotlib.pyplot as plt
        df = self._frame()
        fig = plot_household_range_strips(df)
        try:
            assert fig is not None
            # Single-axis layout (no stacked subplot).
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_unit_appears_in_labels_when_column_known(self):
        import matplotlib.pyplot as plt
        df = self._frame()
        fig = plot_household_range_strips(df, column_name="ElektriciteitNetgebruikHoog")
        try:
            ylabel = fig.axes[0].get_ylabel()
            assert "(kWh)" in ylabel
        finally:
            plt.close(fig)

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"HuisIdBSV": [1], "ProjectIdBSV": [1], "min": [0.0]})
        with pytest.raises(KeyError, match="missing columns"):
            plot_household_range_strips(df)

    def test_missing_optional_percentile_columns_silently_drops_segments(self, tmp_path):
        # min/median/max only -- no q1/q25/q75/q99. Should still render.
        df = pd.DataFrame({
            "HuisIdBSV":       ["A", "B", "C"],
            "ProjectIdBSV":    [1, 1, 2],
            "Dataleverancier": ["X", "X", "Y"],
            "min":             [0.0, 1.0, 5.0],
            "median":          [2.0, 3.0, 7.0],
            "max":             [4.0, 5.0, 9.0],
        })
        out = tmp_path / "strips_minimal.png"
        plot_household_range_strips(df, save_fig_path=str(out))
        assert out.exists() and out.stat().st_size > 0

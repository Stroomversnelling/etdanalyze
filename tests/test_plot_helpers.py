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
    plot_range_strips_blocks,
    plot_range_strips_candlestick,
    unit_for_column,
)
from etdanalyze.analysis_helpers import (
    build_distribution_table,
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

    def test_diff_variant_has_explicit_row(self):
        # Diff variants are explicit rows in etdmodel.csv (Diff='ja'),
        # not synthesised from a name suffix.
        assert unit_for_column("ElektriciteitNetgebruikHoogDiff") == "kWh"

    def test_unknown_returns_empty(self):
        # Low-level helper stays silent on unknown -- axis_label_with_unit
        # is the layer that raises.
        assert unit_for_column("__not_a_real_column__") == ""

    def test_axis_label_appends_when_known(self):
        assert axis_label_with_unit("ElektriciteitNetgebruikHoog", "median") == "median (kWh)"

    def test_axis_label_raises_on_unknown(self):
        # Loud failure: callers must pass an override for non-data-model columns.
        with pytest.raises(KeyError, match="[Nn]o unit known"):
            axis_label_with_unit("__not_a_real_column__", "median")

    def test_axis_label_defaults_base_to_column(self):
        assert axis_label_with_unit("ElektriciteitNetgebruikHoog") == "ElektriciteitNetgebruikHoog (kWh)"

    def test_overrides_unknown_column(self):
        # Override allows labelling a column that isn't in the data model.
        assert axis_label_with_unit(
            "COP", "COP", overrides={"COP": ""}
        ) == "COP"

    def test_overrides_take_precedence_over_data_model(self):
        # When a Diff column has been display-converted to kW (post-helper),
        # the caller overrides the data-model unit (kWh) with the
        # display unit.
        assert axis_label_with_unit(
            "ElektriciteitNetgebruikHoogDiff", "median",
            overrides={"ElektriciteitNetgebruikHoogDiff": "kW"},
        ) == "median (kW)"

    def test_unitless_override_returns_bare_base(self):
        # An empty-string override is the explicit unitless marker.
        assert axis_label_with_unit(
            "SomeUnitlessRatio", "median",
            overrides={"SomeUnitlessRatio": ""},
        ) == "median"


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
    independent of distribution shape. The p99*10 / p01*10 thresholds
    are designed for sufficiently large samples (typically per-row
    time-series with hundreds of records per project), where p99 is
    substantially less than the extreme tail values."""

    def _frame_with_extreme_high(self, n_normal=200):
        # Bulk around 100 (n_normal rows); one value 1000x larger that
        # comfortably exceeds p99 * 10.
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
        # When p01 is negative, bound_low becomes p01*10. A value far
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

    def test_x_order_overrides_default_alphabetical(self):
        """x_order is opt-in. Omitted -> the default alphabetical (string)
        order, so numeric ProjectIdBSV reads 1, 10, 2. Supplied -> the
        caller's exact left-to-right order, e.g. 1, 2, 10. The default must be
        unchanged (it is an intended feature, not a bug)."""
        import matplotlib.pyplot as plt
        rows = [
            {"HuisIdBSV": f"{p}_{i}", "ProjectIdBSV": p, "WoningType": "A",
             "median": float(p + i)}
            for p in (1, 2, 10) for i in range(6)
        ]
        df = pd.DataFrame(rows)
        # default unchanged: alphabetical by str -> "1", "10", "2"
        fig = plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV",
            color_group="ProjectIdBSV",
        )
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "10", "2"]
        plt.close(fig)
        # explicit numeric order honoured
        fig = plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV",
            color_group="ProjectIdBSV", x_order=[1, 2, 10],
        )
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "2", "10"]
        plt.close(fig)

    def test_x_order_keeps_unlisted_categories(self):
        """Categories present in the data but absent from x_order are appended
        (alphabetically) so none are silently dropped."""
        import matplotlib.pyplot as plt
        rows = [
            {"HuisIdBSV": f"{p}_{i}", "ProjectIdBSV": p, "WoningType": "A",
             "median": float(p + i)}
            for p in (1, 2, 3) for i in range(6)
        ]
        df = pd.DataFrame(rows)
        fig = plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV",
            color_group="ProjectIdBSV", x_order=[3],  # only 3 listed
        )
        # 3 first (listed), then 1, 2 appended alphabetically
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["3", "1", "2"]
        plt.close(fig)

    def test_raises_on_all_na_value_col(self):
        """Strict empty-data contract: when every row's value_col is NA
        the helper raises ValueError pointing the caller at upstream
        coverage filtering. No silent return / no empty plot."""
        df = self._frame()
        df["median"] = pd.NA
        with pytest.raises(ValueError, match="no non-NA rows"):
            plot_boxwhisker_comparison(
                df, value_col="median", x_group="ProjectIdBSV",
                color_group="WoningType",
            )

    def test_raises_on_empty_input(self):
        """Same guard, different shape -- zero rows in df."""
        df = pd.DataFrame(columns=["HuisIdBSV", "ProjectIdBSV", "WoningType", "median"])
        with pytest.raises(ValueError, match="no non-NA rows"):
            plot_boxwhisker_comparison(
                df, value_col="median", x_group="ProjectIdBSV",
                color_group="WoningType",
            )

    def test_scatter_all_renders(self, tmp_path):
        """Explicit scatter='all' draws the whole-population swarm."""
        df = self._frame()
        out = tmp_path / "scatter_all.png"
        plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            scatter="all", save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scatter_none_renders(self, tmp_path):
        """scatter='none' draws boxes only, no overlay markers."""
        df = self._frame()
        out = tmp_path / "scatter_none.png"
        plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            scatter="none", save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scatter_matplotlib_renders(self, tmp_path):
        """scatter='matplotlib' uses native fliers; tinting is applied
        per box when x_group == color_group."""
        df = self._frame()
        out = tmp_path / "scatter_matplotlib.png"
        # Force outliers so there's at least one flier per box: append
        # extreme points to P1 and P3.
        extra = pd.DataFrame([
            {"HuisIdBSV": "P1_X", "ProjectIdBSV": "P1", "WoningType": "TypeA", "median": 9999.0},
            {"HuisIdBSV": "P3_X", "ProjectIdBSV": "P3", "WoningType": "TypeA", "median": -9999.0},
        ])
        df2 = pd.concat([df, extra], ignore_index=True)
        plot_boxwhisker_comparison(
            df2, value_col="median",
            x_group="ProjectIdBSV", color_group="ProjectIdBSV",
            scatter="matplotlib", save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scatter_auto_picks_all_at_low_n(self, tmp_path):
        """At default N (~25 rows per box) auto resolves to 'all'."""
        df = self._frame()
        out = tmp_path / "scatter_auto_low.png"
        plot_boxwhisker_comparison(
            df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
            scatter="auto", save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scatter_auto_picks_matplotlib_at_high_n(self, tmp_path):
        """At high N per box (> 200 rows) auto resolves to 'matplotlib'."""
        df = self._frame(n_per_project=500)
        out = tmp_path / "scatter_auto_high.png"
        plot_boxwhisker_comparison(
            df, value_col="median",
            x_group="ProjectIdBSV", color_group="ProjectIdBSV",
            scatter="auto", save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_scatter_invalid_raises(self):
        """An unknown scatter mode is a programmer error -- raise."""
        df = self._frame()
        with pytest.raises(ValueError, match="unknown scatter mode"):
            plot_boxwhisker_comparison(
                df, value_col="median", x_group="ProjectIdBSV", color_group="WoningType",
                scatter="nonsense",
            )

    def test_grouping_col_drives_per_phrase_in_title(self, tmp_path):
        """grouping_col='WeerstationGroep' renders 'per weerstation' in
        the default title (replacing the historic 'per huishouden')."""
        from etdanalyze.plot_functions import (
            DEFAULT_BOXWHISKER_TITLE_TPL, DEFAULT_BOXWHISKER_YLABEL_TPL,
        )
        # Both templates carry the placeholder so the registry value
        # surfaces in both the title and the y-axis.
        assert "{per_grouping}" in DEFAULT_BOXWHISKER_TITLE_TPL
        assert "{per_grouping}" in DEFAULT_BOXWHISKER_YLABEL_TPL

        df = self._frame()
        out = tmp_path / "grouping.png"
        import matplotlib.pyplot as plt
        fig = plot_boxwhisker_comparison(
            df, value_col="median",
            x_group="ProjectIdBSV", color_group="WoningType",
            grouping_col="WeerstationGroep",
        )
        try:
            assert fig is not None
            ax = fig.axes[0]
            assert "per weerstation" in ax.get_title()
            assert "per weerstation" in ax.get_ylabel()
            fig.savefig(out, dpi=150)
        finally:
            plt.close(fig)
        assert out.exists()

    def test_grouping_col_defaults_to_id_col(self, tmp_path):
        """No grouping_col -> falls back to id_col (HuisIdBSV) ->
        'per huishouden' in the title."""
        df = self._frame()
        import matplotlib.pyplot as plt
        fig = plot_boxwhisker_comparison(
            df, value_col="median",
            x_group="ProjectIdBSV", color_group="WoningType",
        )
        try:
            assert fig is not None
            ax = fig.axes[0]
            assert "per huishouden" in ax.get_title()
        finally:
            plt.close(fig)

    def test_grouping_col_unknown_falls_back_to_per_col(self, tmp_path):
        """Unknown grouping column -> generic 'per <col>' phrase."""
        df = self._frame()
        import matplotlib.pyplot as plt
        fig = plot_boxwhisker_comparison(
            df, value_col="median",
            x_group="ProjectIdBSV", color_group="WoningType",
            grouping_col="WildlyMadeUpColumn",
        )
        try:
            assert fig is not None
            ax = fig.axes[0]
            assert "per WildlyMadeUpColumn" in ax.get_title()
        finally:
            plt.close(fig)

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
# plot_range_strips_candlestick  (per-household candlestick)
# ---------------------------------------------------------------------------


class TestPlotHouseholdRangeStrips:
    """The new helper draws one vertical range strip per household with
    min-p25-q25-median-q75-p99-max marks, grouped by project. Smoke tests
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
                    "Dataleverancier": "SupplierB" if proj > 1 else "SupplierA",
                    "min": float(base + rng.normal(0, 0.1)),
                    "p01": float(base + 0.5),
                    "p25": float(base + 1.0),
                    "median": float(base + 1.5),
                    "p75": float(base + 2.0),
                    "p99": float(base + 2.5),
                    "max": float(base + 3.0 + rng.uniform(0, 1)),
                })
        return pd.DataFrame(rows)

    def test_writes_non_empty_png(self, tmp_path):
        df = self._frame()
        out = tmp_path / "strips.png"
        plot_range_strips_candlestick(
            df, column_name="ElektriciteitNetgebruikHoog",
            save_fig_path=str(out),
        )
        assert out.exists() and out.stat().st_size > 0

    def test_returns_figure_when_no_path(self):
        import matplotlib.pyplot as plt
        df = self._frame()
        fig = plot_range_strips_candlestick(df)
        try:
            assert fig is not None
            # Single-axis layout (no stacked subplot).
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_unit_appears_in_labels_when_column_known(self):
        import matplotlib.pyplot as plt
        df = self._frame()
        fig = plot_range_strips_candlestick(df, column_name="ElektriciteitNetgebruikHoog")
        try:
            ylabel = fig.axes[0].get_ylabel()
            assert "(kWh)" in ylabel
        finally:
            plt.close(fig)

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"HuisIdBSV": [1], "ProjectIdBSV": [1], "min": [0.0]})
        with pytest.raises(KeyError, match="missing columns"):
            plot_range_strips_candlestick(df)

    def test_missing_optional_percentile_columns_silently_drops_segments(self, tmp_path):
        # min/median/max only -- no p25/q25/q75/p99. Should still render.
        df = pd.DataFrame({
            "HuisIdBSV":       ["A", "B", "C"],
            "ProjectIdBSV":    [1, 1, 2],
            "Dataleverancier": ["X", "X", "Y"],
            "min":             [0.0, 1.0, 5.0],
            "median":          [2.0, 3.0, 7.0],
            "max":             [4.0, 5.0, 9.0],
        })
        out = tmp_path / "strips_minimal.png"
        plot_range_strips_candlestick(df, save_fig_path=str(out))
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# build_distribution_table  (report-ready per-(group, period) stats table)
# ---------------------------------------------------------------------------


class TestBuildDistributionTable:
    """Stacked per-(group, period) distribution stats with a population
    roll-up row per period. Returns a tidy frame ready for .to_markdown()."""

    _COLS = ["period", "ProjectIdBSV", "WoningType",
             "n", "gem.", "med.", "p25", "p75", "min", "max"]

    def _frame(self):
        # P1/A/year: 10,20,30 ; P2/B/year: 40,60 ; P1/A/verw: 100,200.
        # One 'zomer' row that is NOT in the periods we request -> dropped.
        return pd.DataFrame({
            "ProjectIdBSV": [1, 1, 1, 2, 2, 1, 1, 1],
            "WoningType":   ["A", "A", "A", "B", "B", "A", "A", "A"],
            "period":       ["year", "year", "year", "year", "year",
                             "verwarmingsperiode", "verwarmingsperiode", "zomer"],
            "val":          [10.0, 20.0, 30.0, 40.0, 60.0, 100.0, 200.0, 999.0],
        })

    def _row(self, out, **eq):
        mask = pd.Series(True, index=out.index)
        for k, v in eq.items():
            mask &= out[k] == v
        sub = out[mask]
        assert len(sub) == 1, f"expected exactly one row for {eq}, got {len(sub)}"
        return sub.iloc[0]

    def test_schema_columns_exact_order(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("year", "verwarmingsperiode"))
        assert list(out.columns) == self._COLS

    def test_empty_when_no_matching_period_preserves_columns(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("no_such_period",))
        assert out.empty
        assert list(out.columns) == self._COLS

    def test_per_group_stats_are_correct(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("year", "verwarmingsperiode"))
        r = self._row(out, period="year", ProjectIdBSV=1, WoningType="A")
        assert int(r["n"]) == 3
        assert r["gem."] == pytest.approx(20.0)
        assert r["med."] == pytest.approx(20.0)
        assert r["p25"] == pytest.approx(15.0)
        assert r["p75"] == pytest.approx(25.0)
        assert r["min"] == pytest.approx(10.0)
        assert r["max"] == pytest.approx(30.0)

    def test_population_rollup_aggregates_across_all_households(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("year", "verwarmingsperiode"))
        # Roll-up row: first group col holds the label, the rest are blank.
        r = self._row(out, period="year", ProjectIdBSV="alle HHs")
        assert r["WoningType"] == ""
        assert int(r["n"]) == 5                      # 10,20,30,40,60
        assert r["gem."] == pytest.approx(32.0)
        assert r["med."] == pytest.approx(30.0)
        assert r["min"] == pytest.approx(10.0)
        assert r["max"] == pytest.approx(60.0)

    def test_out_of_scope_periods_are_dropped(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("year", "verwarmingsperiode"))
        assert "zomer" not in set(out["period"])
        assert 999.0 not in set(out["max"])

    def test_periods_ordered_within_a_group(self):
        out = build_distribution_table(
            self._frame(), "val", periods=("year", "verwarmingsperiode"))
        p1a = out[(out["ProjectIdBSV"] == 1) & (out["WoningType"] == "A")]
        assert list(p1a["period"]) == ["year", "verwarmingsperiode"]

    def test_variadic_single_group_col(self):
        out = build_distribution_table(
            self._frame(), "val",
            group_cols=("ProjectIdBSV",), periods=("year",))
        assert list(out.columns) == [
            "period", "ProjectIdBSV", "n", "gem.", "med.", "p25", "p75", "min", "max"]
        # Roll-up row labels the single group column.
        r = self._row(out, period="year", ProjectIdBSV="alle HHs")
        assert int(r["n"]) == 5

    def test_count_excludes_missing_values(self):
        # ADR-005: pd.NA in a Float64 column is missing; n counts non-NA only.
        df = pd.DataFrame({
            "ProjectIdBSV": [1, 1, 1],
            "WoningType":   ["A", "A", "A"],
            "period":       ["year", "year", "year"],
            "val":          pd.array([10.0, pd.NA, 30.0], dtype="Float64"),
        })
        out = build_distribution_table(df, "val", periods=("year",))
        r = self._row(out, period="year", ProjectIdBSV=1, WoningType="A")
        assert int(r["n"]) == 2
        assert r["gem."] == pytest.approx(20.0)

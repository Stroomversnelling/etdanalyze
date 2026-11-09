# Plotting helpers: design and conventions

How the plotting helpers in `etdanalyze.plot_functions` and the data
utilities in `etdanalyze.analysis_helpers` fit together, and the
conventions a new helper should follow. These are conventions, not laws:
they exist so plots look consistent and callers can guess a signature
before reading it. Depart from them when you have a reason, and say why.

## Quick start

Pick a numeric column to plot, the categorical column for the x-axis, and
the categorical column for colour. The helper resolves the unit from the
ETD data model and labels the axes for you.

```python
import pandas as pd
from etdanalyze.plot_functions import plot_boxwhisker_comparison

# df: one row per household, with the value plus its grouping columns.
plot_boxwhisker_comparison(
    df,
    value_col="median_kwh",          # y-axis numeric
    x_group="ProjectIdBSV",          # categories along x
    color_group="WoningType",        # colour coding
    column_name="ElektriciteitNetgebruikHoog",  # -> unit + title text
    x_order=sorted(df["ProjectIdBSV"].unique()),  # numeric order, not 1,10,2
    save_fig_path="plots/elek__annual.png",
)
```

`save_fig_path` is the universal switch: pass a path and the helper writes
a PNG and returns `None`; omit it and the helper returns the matplotlib
`Figure` so you can keep tweaking it.

## 1. Two modules, split by side effect

- **`plot_functions.py` -- rendering.** Functions that build matplotlib
  figures and usually save them as PNG: boxplots, daily/weekly/annual
  profiles, scatter plots, energy signatures, load-duration curves.
- **`analysis_helpers.py` -- data.** Functions that filter, aggregate,
  normalise, or summarise dataframes. No figures, no files, no printing.

The dividing question for a new function is "does it draw or save
anything?" If no, it belongs in `analysis_helpers`, even when it exists
only to feed a plot (outlier detection is the classic example: detection
returns a table in `analysis_helpers`; the decision to draw a marker for
it lives in `plot_functions`).

## 2. Helper signature conventions

Most rendering helpers share the same argument vocabulary, so once you
know one you can read the rest:

| Argument | Role |
| --- | --- |
| `df` | Input frame, one row per observation unit (per household, or per `(household, ReadingDate)`). |
| `value_col` | The y-axis numeric. |
| `x_group` | Categorical laid out along x (e.g. `"ProjectIdBSV"`). |
| `color_group` | Categorical that drives colour. |
| `id_col` | Identifier for a single observation (default `"HuisIdBSV"`); used by overlays such as outlier markers. |
| `column_name` | The ETD model column being plotted. Drives unit lookup and title/label text (see §3). |
| `overrides` | Per-call `{column: unit}` map for columns outside the data model (see §3). |
| `save_fig_path` | Set -> save PNG, return `None`. Unset -> return the `Figure`. |

`plot_boxwhisker_comparison` is the richest example and exercises the full
set. Simpler helpers take a subset -- `plot_load_duration_curve` only
needs the frame, a diff column, and an interval:

```python
plot_load_duration_curve(
    df,
    diff_column="ElektriciteitsgebruikTotaalNettoDiff",
    interval="hourly",
    project_id=3,
    save_fig_path="plots/ldc__project3.png",
)
```

## 3. Units and labels

Units come from one source of truth: `etdmap.data_model.load_unit_map()`,
a `{column: unit}` dict that `plot_functions` caches once at import time
as `_UNIT_MAP`. Three helpers sit on top of it:

- `unit_for_column(col) -> str` -- low-level lookup; returns `""` for an
  unknown column.
- `resolve_unit(col, overrides=None) -> str` -- the helper to use. Strict
  by default and override-aware.
- `axis_label_with_unit(col, base_label, overrides=None) -> str` -- builds
  `"base_label (unit)"`, or just `base_label` when the unit is `""`.

`resolve_unit` is **strict**: it checks `overrides` first, then the data
model, and **raises `KeyError`** if neither knows the column. There is no
silent blank fallback -- a missing unit is a bug that should surface
loudly, not a label that quietly drops its unit. The two reasons to pass
`overrides` are (a) a derived variable that is not in the data model, and
(b) a display-unit variant whose unit changed upstream (e.g. a `Diff`
column converted from kWh to kW). An empty-string override is meaningful:
it declares the value genuinely unitless.

```python
>>> axis_label_with_unit("ElektriciteitNetgebruikHoog", "median value")
'median value (kWh)'

# derived variant now in kW after convert_diff_to_power:
>>> axis_label_with_unit(
...     "ElektriciteitNetgebruikHoogDiff", "median",
...     overrides={"ElektriciteitNetgebruikHoogDiff": "kW"},
... )
'median (kW)'

# explicitly unitless:
>>> axis_label_with_unit("COP", "COP", overrides={"COP": ""})
'COP'
```

Helpers opt in by accepting `column_name` (and `overrides`) and feeding
them to `resolve_unit`. When a helper owns the unit this way, do **not**
pre-stamp a unit into the title text yourself -- you will get a doubled
unit, which is the intended loud failure mode.

## 4. Outlier detection: two styles for two questions

Detection and rendering are separate. `analysis_helpers` returns a table
of flagged rows; the caller decides whether to draw markers, list them in
a side table, annotate them, or ignore them. There are two detection
helpers because there are two different questions:

**`build_outlier_table` -- "is this value unusual relative to its
cohort?"** Tukey fence (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) within each group,
falling back to a z-score test (`> 2.5`) when a group has fewer than four
points or a zero IQR. Best when the cohort is roughly symmetric and a few
points stand out from their neighbours.

```python
out = build_outlier_table(
    df,
    value_col="median_kwh",
    group_cols=("Dataleverancier", "ProjectIdBSV"),
    top_n=10,
)
# columns: id_col, *group_cols, value, group_median, z_score
# sorted by |z_score| desc; empty frame when nothing is flagged.
```

**`build_extreme_value_table` -- "is this value so extreme it dwarfs the
bulk of the data?"** Percentile bounds: high bound `q99 * 10`, low bound
`q01 * 10` when `q01` is negative (otherwise `0`). Robust to skew, so it
suits right-skewed energy variables where Tukey would flag every healthy
peak. It needs a reasonably large sample (hundreds of rows per group); on
a ~10-row-per-project summary frame the `q99` coincides with the outlier
and nothing useful comes out.

```python
ext = build_extreme_value_table(df, value_col="ElektriciteitsgebruikTotaalNettoDiff")
# columns: id_col, *group_cols, value, bound_low, bound_high, direction
# direction is "above" / "below"; sorted by distance from the violated bound.
```

The two are not interchangeable: reach for `build_outlier_table` to flag
rows that stand out from neighbours, and `build_extreme_value_table` to
catch values that would visually dominate a chart or a downstream sum.

Both helpers are pure -- they return a table and never print or mutate the
input. A caller that wants side effects (tagging an `is_outlier` column,
printing which timestamps were dropped) builds that on top.

## 5. Range clusters: detection as metadata

`detect_value_range_clusters(per_group_min, per_group_max)` answers yet
another question: "do these groups fall into separate value ranges that a
single shared y-axis would squash?" It takes a per-group min and a
per-group max Series and returns a list of key-sets:

```python
mins = df.groupby("ProjectIdBSV")["value"].min()
maxs = df.groupby("ProjectIdBSV")["value"].max()
clusters = detect_value_range_clusters(mins, maxs)
# [] -> no groups
# [{1, 2, 3}] -> one range, plot normally
# [{1, 2}, {3, 4}] -> two ranges, ordered by ascending cluster max
```

A group is judged "cramped" when its own range is a tiny fraction of its
cluster's y-axis span, or when an all-positive cluster spans more than two
orders of magnitude (the small boxes would collapse to a line at the axis
edge). The function only reports the split; it does not act on it. A
caller can emit one figure per cluster (each with its own y-axis), switch
to a log scale, annotate the largest gap, or ignore the result entirely.

## 6. Layout: stacked subplots, twinx, constrained_layout

- Create figures with `plt.subplots(..., constrained_layout=True)` rather
  than calling `plt.tight_layout()` afterwards. `tight_layout` fights with
  secondary axes, external legends, and multi-row figures, and emits noisy
  `UserWarning`s.
- To show a companion metric alongside the main plot, use a **stacked
  subplot row** that shares the x-axis, not `twinx()`. Give the primary
  plot the larger share with `gridspec_kw={"height_ratios": [3, 1]}`:

  ```python
  fig, (ax_main, ax_aux) = plt.subplots(
      2, 1, sharex=True,
      gridspec_kw={"height_ratios": [3, 1]},
      constrained_layout=True,
  )
  ```

  `plot_boxwhisker_comparison` does exactly this when
  `secondary_axis_data` is passed: the box distribution sits on top and
  the companion metric (e.g. a validation fail-rate) sits beneath.
- `twinx()` is not banned, but reserve it for the rare case where two
  metrics genuinely share one coordinate space and must overlay. A helper
  that uses it should be a dedicated plot type or gate it behind an
  explicit kwarg.

## 7. Style defaults

Helpers tend to converge on these; override locally when a plot type needs
a different look (heatmaps, time series, energy signatures do).

- **DPI:** 150 for report-oriented output, 300 where print quality
  matters.
- **Colormap:** `tab10` for up to 10 categorical groups, cycling beyond
  that.
- **Figure width:** scale with the number of x-categories, e.g.
  `max(8.0, n_x * 1.6 + 2.0)`.

## 8. Out of scope (deliberately)

What this library does not do yet, so a reader knows where the edges are.
Each is a fair candidate for future work once a concrete caller needs it.

- **Faceting.** No general `facet_by` wrapper; a caller that needs several
  plots from one frame writes its own loop.
- **Central theme.** No shared matplotlib `rcParams`; each helper sets its
  own colours, fonts, and grid inline.
- **Accessibility / dark mode.** No colour-blind-safe or dark palettes.
- **Interactivity.** Static PNG only -- no plotly/bokeh.
- **Legend placement at scale.** The external
  `bbox_to_anchor=(1.01, 1)` legend works for a handful of categories and
  overflows past ~10.
- **Caching.** Helpers re-render on every call.

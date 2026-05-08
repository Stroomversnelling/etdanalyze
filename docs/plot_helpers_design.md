# Plot helpers design notes

This is a descriptive walk-through of how the plot helpers in
`etdanalyze.plot_functions` and the supporting utilities in
`etdanalyze.analysis_helpers` have been put together up to this point.
It is not a set of rules. It is meant to give a future contributor
enough context to decide where it makes sense to follow the established
pattern and where there is a good reason to depart from it.

## 1. Module shape

There are two modules involved:

- `etdanalyze/plot_functions.py` -- rendering. Functions that produce
  matplotlib figures and (typically) save them as PNG. This is where
  boxplots, daily/weekly/annual profiles, scatter plots, energy
  signatures, and load-duration curves live.
- `etdanalyze/analysis_helpers.py` -- data utilities. Functions that
  filter, aggregate, normalise, or summarise dataframes. They have no
  side effects on figures or files.

The split is by side effect: `plot_functions` writes images and prints
analysis lines; `analysis_helpers` returns dataframes. When a piece of
logic could fit either module (e.g. outlier detection), the test is
"does it draw or save anything?" -- if no, it goes in
`analysis_helpers`.

## 2. Helper signature conventions

Most of the existing rendering helpers share a similar argument
shape:

- `df` -- the input DataFrame, one row per observation unit
  (typically per household or per (household, ReadingDate)).
- `value_col`, `x_group`, `color_group` -- the three semantic
  axes of the plot. `value_col` is the y-axis numeric. `x_group`
  is the categorical column laid out along x (e.g. `ProjectIdBSV`).
  `color_group` is the categorical that drives colour coding.
- `id_col` -- identifier for individual observations (default
  `"HuisIdBSV"`). Used by helpers that overlay specific
  observations (e.g. outlier markers).
- `column_name` (optional) -- the ETD model column being plotted.
  When given, the helper looks up its unit from
  `etdmap.data_model.load_unit_map()` and appends the unit to the
  y-axis label and title.
- `save_fig_path` -- when set, the helper saves the figure to that
  path and returns `None`. When unset, it returns the figure
  object so the caller can do its own thing.

`plot_boxwhisker_comparison` is the most parameter-rich helper and
illustrates the convention. The simpler helpers (e.g.
`plot_load_duration_curve`) take a subset of these parameters.

## 3. Units and labels

`etdmap/data/thresholds.csv` is the project-wide source of truth for
variable units (column `Eenheid`). It is loaded once via
`etdmap.data_model.load_unit_map()`, which returns a
`{variable_name: unit_string}` dict. `plot_functions.py` caches it at
import time as `_UNIT_MAP` and exposes two small helpers:

- `unit_for_column(col) -> str` -- the unit, or `""` if unknown.
- `axis_label_with_unit(col, base_label) -> str` -- the base label
  with `" (unit)"` appended when a unit is known.

Helpers can opt in by accepting a `column_name` argument and using
either of the above. Today not every helper does; the ones that have
been touched recently (`plot_boxwhisker_comparison`,
`plot_load_duration_curve`) do.

## 4. Outlier handling: detection vs rendering, and the two detection styles

Outlier *detection* and outlier *rendering* are separated. Detection
returns a dataframe; rendering decides whether to draw markers, list
them in a side table, annotate inline, or omit entirely.

There are **two complementary detection styles** in `analysis_helpers`,
named for the question they answer:

### `build_outlier_table` (Tukey + z-score fallback)

- Question: "is this value unusual *relative to its cohort
  distribution*?"
- Method: Tukey fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR) within each
  group. Falls back to a z-score test (>2.5) when the group has
  fewer than 4 points or the IQR is zero.
- Best for: spotting per-project anomalies where the cohort has a
  reasonably symmetric distribution and a small number of unusual
  observations stand out from neighbours.
- Output: long-form table with columns `id_col`, `*group_cols`,
  `value`, `group_median`, `z_score`, sorted by `|z_score|`
  descending.

### `build_extreme_value_table` (percentile-bounds)

- Question: "is this value so extreme that it dwarfs the bulk of
  the data, regardless of distribution shape?"
- Method: q99 * 10 for the high bound, q01 * 10 for the low bound
  when q01 is itself negative (otherwise the low bound is 0).
  Mirrors `filter_between_upper_lower_bounds` (which is the
  underlying filter that returns surviving rows rather than
  flagging outliers).
- Best for: energy variables. They are typically right-skewed,
  with legitimate large values at the tails. Tukey would flag
  every healthy peak; q99*10 only flags the truly abnormal.
  Robust to non-normality. Designed for sufficiently large
  samples (hundreds of observations per group); will not work
  meaningfully on per-project summary frames with ~10 rows per
  project, because the q99 itself coincides with the outlier.
- Output: long-form table with columns `id_col`, `*group_cols`,
  `value`, `bound_low`, `bound_high`, `direction`
  (`"above"` / `"below"`), sorted by absolute distance from the
  violated bound.

The two are not interchangeable. Use `build_outlier_table` when you
want to flag rows that stand out from neighbours. Use
`build_extreme_value_table` when you want to clip values that would
visually dominate downstream calculations.

`etdanalyze-internal/analysis/analysis_helpers.py
.filter_and_report_outliers` uses the percentile-bounds method and
adds two side effects: it tags an `is_outlier` boolean column on the
input dataframe and prints a summary of removed values to stdout.
That is appropriate for the load-duration curve workflow it serves
(reviewers want to see exactly which timestamps were dropped). The
generic `build_*` helpers in `analyze` deliberately do not print or
mutate; they return tables and let the caller decide.

Rendering choices -- markers, side tables, inline annotations -- are
made by each caller. The mapping report draws X markers on the
boxplot for outliers and emits a markdown table beneath the image;
the load-duration curve in etdanalyze-internal draws red dashed
verticals at outlier positions. These are local choices, not rules.

## 5. Range / cluster detection as metadata

`detect_value_range_clusters(per_group_values)` answers a different
question: "do the per-group values fall into N visibly separate
ranges (orders of magnitude apart)?" It returns a list of cluster
key sets ordered by ascending cluster median:

- length 1: no meaningful split, all groups are one range
- length 2+: N clusters, each with its own member set

The detection itself is just metadata. It does not decide what to
do about the split. Callers can:

- emit one PNG per cluster (the mapping report's choice -- so each
  cluster gets its own y-axis and the smaller boxes are not
  squashed by the largest);
- use a log y-scale on a single plot (loses relative-magnitude
  intuition but keeps everything in one image);
- annotate the largest gap on a single plot;
- ignore it entirely.

The mapping report's choice is to emit one PNG per cluster with
sequence-numbered filenames `<column>__<season>__<i>.png` in
ascending median order. That is a *report* decision; nothing in
`detect_value_range_clusters` requires it.

A future direction would be `summarize_value_ranges` returning a
richer metadata dict (per-cluster boundaries, spreads, separation
score) plus a `facet_by(plot_fn, ...)` wrapper that takes any plot
helper and an arbitrary partitioning rule. Both are deferred until a
second consumer needs them.

## 6. Layout: stacked subplots, twinx, constrained_layout

The current convention for layout:

- Figures use `constrained_layout=True` at `plt.subplots()` time
  rather than a post-hoc `plt.tight_layout()` call. The latter
  fights with secondary axes, externally placed legends, and
  multi-row figures, and produces UserWarnings that get noisy.
- When a helper needs to show a companion metric (e.g.
  validate_fail_rate next to value distribution), the convention
  is a **stacked subplot row** sharing the x-axis (typically with
  `gridspec_kw={"height_ratios": [3, 1]}` so the primary plot
  dominates), not `twinx()`.
- `twinx()` is not forbidden. It is the right choice when two
  metrics share a meaningful coupling that a separate subplot
  would lose -- for instance a journalism-style explanatory chart
  where two scales must literally overlay. A helper that uses
  `twinx` should either be its own dedicated plot type or expose
  the choice through an explicit kwarg.

`plot_boxwhisker_comparison` switches between single-axis and
two-row stacked layout based on whether `secondary_axis_data` is
provided. It does not use `twinx`.

## 7. Filename conventions (mapping report)

The mapping report writes plot PNGs into a `plots/` subfolder of
its output directory. The schema has two cases:

- `<column>__<season>.png` -- one plot for the whole cohort.
- `<column>__<season>__<i>.png` for `i = 1..N` -- N plots when
  the report decided to split a multi-range cohort. Sequence
  ordered by ascending cluster median (1 = lowest range).

`<season>` is one of `annual`, `cold`, `warm`. The mapping report
runs a hygiene pass at the start of `main()` that deletes any PNG
in the plot folder that does not match this schema, so the folder
stays self-consistent across runs.

Other consumers are not bound by this. Any plot output produced by
`etdanalyze.plot_functions` callers can use whatever filename
convention makes sense for that consumer; only the mapping report's
plot folder enforces the schema.

## 8. Style constants

Defaults that the helpers tend to converge on:

- DPI 150 for `savefig` (the mapping report uses 150; some other
  helpers use 300 for higher-quality print output).
- `tab10` colormap for categorical groupings up to 10 categories.
  Helpers cycle wrap when `color_group` has more than 10
  categories.
- Figure widths scale with the number of x-categories
  (`max(8.0, n_x * 1.6 + 2.0)` is a common pattern).

These are conventions, not requirements. Helpers that need a
different look (heatmaps, time series, energy signatures) override
locally.

## 9. What this codebase does not solve yet

For honest scoping:

- **Faceting.** No general `facet_by` wrapper. When a helper
  needs to produce several plots from one input (e.g. one per
  cluster), the caller currently writes the iteration loop.
- **Project-wide style theme.** No central matplotlib rcParams
  preset. Each helper sets its own colours, font sizes, grid
  styles inline.
- **Dark mode / accessibility.** No alternative palettes for
  colour-blind viewers; no dark-background variant.
- **Interactive plots.** Everything is static PNG. No plotly /
  bokeh helpers.
- **Automatic legend placement** at `n_color_group > 10`. The
  current `bbox_to_anchor=(1.01, 1)` external legend works for a
  handful of categories; large category counts overflow.
- **Caching.** Helpers re-render every time. Some report
  generators wipe and rebuild their entire plot folder per run.

These are flagged so a reader knows what is intentionally
out-of-scope. They are good candidates for future work when a
concrete consumer needs them.

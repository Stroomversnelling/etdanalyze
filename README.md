# etdanalyze

Plotting helpers and dataframe utilities for analysing
[ETD](https://energietransitiedataset.nl/) household energy data, built on
the [etdmap](https://energietransitiedataset.nl/) data model.

For an overview of the ETD and all documentation, see
<https://energietransitiedataset.nl/>.

> N.B. This documentation (and the repo as a whole) is a work in progress.

## What's in the package

- **`etdanalyze.plot_functions`** -- rendering. Builds matplotlib figures
  (boxplots, daily/weekly/annual profiles, scatter plots, energy
  signatures, load-duration curves) and usually saves them as PNG.
- **`etdanalyze.analysis_helpers`** -- data. Filters, aggregates,
  normalises, and summarises dataframes. No figures, no files.
- **`etdanalyze.units`** -- the single source of truth for unit
  resolution and unit-aware conversions between energy and power.

Units come from the etdmap data model: every plotted value carries an
explicit unit, and helpers raise rather than guess when a unit is unknown.

## Setup

Each repo has its own virtual environment. Activate it before running any
Python command:

```bash
source .venv/Scripts/activate
```

## Quick start

Pick a numeric column to plot, the categorical column for the x-axis, and
the categorical column for colour. The helper resolves the unit and labels
the axes for you.

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

## Documentation

- [`docs/plot_helpers_design.md`](docs/plot_helpers_design.md) -- design
  and conventions for the plotting helpers and data utilities.
- [`DECISIONS.md`](DECISIONS.md) -- architecture decision records for this
  repo (units as first-class values, strict empty-data guards, the
  variable-type override schema).

# License

`etdanalyze` is released under the [BSD 3-Clause License](LICENSE). The licence covers this code only.

Please cite this package when you use it in publications, analyses or other software (see below and [CITATION.cff](CITATION.cff)).

# Citation

If you use this package or any code in this package or refer to output, please use the following citations for attribution:

_Witkamp, Dickinson, Izeboud (2024). etdanalyze: A Python package for analyzing data using the Energietransitie Dataset (ETD) model._

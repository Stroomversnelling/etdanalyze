import textwrap
from typing import Optional

import etdtransform
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from etdmap.data_model import load_unit_map

from . import analysis_helpers

# ---------------------------------------------------------------------------
# Unit registry — single source of truth for variable units across plots.
# Loaded once at module import from etdmap/data/thresholds.csv.
# ---------------------------------------------------------------------------

_UNIT_MAP = load_unit_map()


def unit_for_column(col: str) -> str:
    """
    Return the unit string for a known model column, '' otherwise.

    Low-level lookup against ``etdmap.data_model.load_unit_map``. Handles
    Diff variants (which inherit unit from parent) via the data-model
    helper. Returns '' on unknown columns -- callers that need the
    raise-on-missing contract should use ``axis_label_with_unit`` instead.
    """
    return _UNIT_MAP.get(col, "")


def resolve_unit(col: str, overrides: Optional[dict] = None) -> str:
    """
    Return the unit string for a column, with optional per-call overrides.

    Resolution order, strict-by-default:

      1. ``overrides[col]`` -- caller-supplied. ``""`` is a valid value
         that explicitly marks the column unitless. Use this for
         derived / analysis-time variables that are not in the etdmap
         data model (e.g. ``COP_system`` -> ``""``,
         ``ZonopwekBrutoPer100M2KW`` -> ``"kW per 100m2"``) and for
         display-unit variants produced upstream (e.g. a Diff column
         after ``convert_diff_to_power`` is now kW, not kWh).
      2. ``etdmap.data_model.load_unit_map`` via ``unit_for_column``.
      3. If neither yields a unit, ``KeyError`` is raised. There is no
         silent fallback -- callers must declare their unit.

    Returns
    -------
    str
        The unit string (possibly empty ``""`` for explicitly unitless).

    Raises
    ------
    KeyError
        Column is in neither ``overrides`` nor the data model.
    """
    if overrides is not None and col in overrides:
        return overrides[col]
    unit = unit_for_column(col)
    if unit:
        return unit
    raise KeyError(
        f"resolve_unit: no unit known for column {col!r}. "
        f"Pass overrides={{{col!r}: '<unit>'}} (use '' for unitless)."
    )


def axis_label_with_unit(
    col: str,
    base_label: Optional[str] = None,
    *,
    overrides: Optional[dict] = None,
) -> str:
    """
    Build a unit-suffixed axis label for a column.

    Lookup order for the unit:
      1. ``overrides[col]`` if ``overrides`` is provided and has the column.
         An empty-string override is meaningful: it declares the value
         genuinely unitless (e.g. COP) and the base label is returned bare.
      2. ``etdmap.data_model.load_unit_map`` via ``unit_for_column``.
      3. If neither yields a unit, ``KeyError`` is raised. There is no
         silent bare-label fallback -- callers must pass an override for
         any column outside the data model.

    Parameters
    ----------
    col : str
        Column name to label. Must either be in the data model or appear
        in ``overrides``.
    base_label : str, optional
        Text before the unit suffix. Defaults to ``col``.
    overrides : dict, optional
        Per-call ``{col: unit}`` map. Pass ``""`` as the unit value for
        explicitly unitless quantities. Per-call overrides are required
        when the value being labelled is a display-unit variant
        (e.g. ``"kW"`` after ``convert_diff_to_power`` produced kW from a
        kWh Diff column) or when the column is not in the data model.

    Examples
    --------
    >>> axis_label_with_unit("ElektriciteitNetgebruikHoog", "median value")
    'median value (kWh)'
    >>> axis_label_with_unit(
    ...     "ElektriciteitNetgebruikHoogDiff", "median",
    ...     overrides={"ElektriciteitNetgebruikHoogDiff": "kW"},
    ... )
    'median (kW)'
    >>> axis_label_with_unit("COP", "COP", overrides={"COP": ""})
    'COP'
    """
    base = base_label if base_label is not None else col
    unit = resolve_unit(col, overrides=overrides)
    return f"{base} ({unit})" if unit else base


# ---------------------------------------------------------------------------
# Bimodal y-axis split detector — generalises the "two clusters of projects
# with very different value ranges" pattern that squashes shared-axis boxplots.
# ---------------------------------------------------------------------------

# Heuristic constants -- kept at module scope so a future tuning pass is one edit.
_RANGE_CLUSTER_MIN_GROUPS = 2        # need at least N groups to consider a split
_RANGE_CLUSTER_MIN_VISIBLE_RATIO = 0.05  # a group's range / cluster y-axis must
                                          # be at least this much; below this
                                          # the group's box is visually cramped
                                          # and belongs in its own cluster


def detect_value_range_clusters(
    per_group_min: pd.Series,
    per_group_max: pd.Series,
) -> list:
    """
    Split groups into N clusters so that no group's value range is
    visually cramped inside its cluster's y-axis span.

    Each group has a range [min, max]. A cluster's y-axis span is the
    union of its members' ranges, i.e. (max of maxes) - (min of mins).
    A group is "cramped" within a cluster when its own range
    (group.max - group.min) is a tiny fraction (< MIN_VISIBLE_RATIO)
    of the cluster's y-axis span: visually its box would be flattened
    against either edge of the chart and unreadable.

    Algorithm:
      1. Sort groups by max ascending.
      2. Walk through. Maintain the current cluster's (cluster_min,
         cluster_max). For each new group, tentatively add it and
         check whether ANY group in the resulting cluster (existing
         members or the new arrival) would be visually cramped.
      3. If yes, finalise the current cluster and start a new one
         with the new group. If no, accept it.

    This frames cluster boundaries entirely in visual terms. Both
    extreme high outliers (their inclusion blows out the y-axis and
    cramps everyone else) AND extreme low / very negative outliers
    (symmetric: they push the cluster_min far below) trigger
    splits the same way.

    Returns
    -------
    list[set]
        Cluster key sets, ordered by ascending cluster max value.
        Length 1 when no meaningful split. Length N >= 2 when the
        data has N visibly separate value ranges.
    """
    common = per_group_min.dropna().index.intersection(per_group_max.dropna().index)
    if len(common) == 0:
        return []
    if len(common) < _RANGE_CLUSTER_MIN_GROUPS:
        return [set(common)]

    mins = per_group_min.loc[common].astype(float)
    maxs = per_group_max.loc[common].astype(float)

    sorted_keys = list(maxs.sort_values().index)
    ranges = {k: float(maxs[k]) - float(mins[k]) for k in sorted_keys}

    def _any_cramped(member_keys, span: float) -> bool:
        if span <= 0:
            return False  # all members are at the same point; nothing cramped
        for k in member_keys:
            if ranges[k] / span < _RANGE_CLUSTER_MIN_VISIBLE_RATIO:
                return True
        return False

    clusters = []
    current = []
    cur_min = None
    cur_max = None
    for key in sorted_keys:
        kmin = float(mins[key])
        kmax = float(maxs[key])
        if not current:
            current = [key]
            cur_min = kmin
            cur_max = kmax
            continue
        new_min = min(cur_min, kmin)
        new_max = max(cur_max, kmax)
        new_span = new_max - new_min
        if _any_cramped(current + [key], new_span):
            clusters.append(set(current))
            current = [key]
            cur_min = kmin
            cur_max = kmax
        else:
            current.append(key)
            cur_min = new_min
            cur_max = new_max
    if current:
        clusters.append(set(current))

    return clusters


# def plot_simple_vars_time(df, plot_var, title=None, plot_var_name=None, project_id=None, save_fig_path=None)

def plot_var_vs_temp(
        df: pd.DataFrame,
        var: str,
        temp_col: str,
        *,
        title: str = "",
        plot_var_name: Optional[str] = None,
        project_id: Optional[str | int] = None,
        overrides: Optional[dict] = None,
        ):
    """
    Plot ``var`` and ``temp_col`` against ``ReadingDate`` on a twinx axes.

    Strict-contract version: the helper does NOT guess data semantics
    from column names. It plots ``df[var]`` on the left axis and
    ``df[temp_col]`` on the right axis, sourcing units from the etdmap
    data model (or ``overrides``). If a caller wants a derived display
    (e.g. kW power on the left axis instead of the kWh-per-row stored
    in the data model), they convert at the call site (see
    ``etdanalyze.units.convert_diff_to_power``) AND declare the
    post-conversion unit via ``overrides={var: "kW"}``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the dataset; must have ``ReadingDate``,
        ``var`` and ``temp_col`` columns.
    var : str
        The variable on the left y-axis (e.g. ``"ElektriciteitsgebruikTotaalNetto"``).
    temp_col : str
        The temperature column on the right y-axis (e.g.
        ``"TemperatuurBuitenWeerstation"``). No default -- the caller
        is explicit about which temperature source they're plotting.
    title : str, optional
        Plot title.
    plot_var_name : str, optional
        Override the base label for ``var``. Defaults to ``var``.
    project_id : int or str, optional
        Filter ``df`` to that project before plotting.
    overrides : dict, optional
        ``{col: unit}`` mapping forwarded to ``axis_label_with_unit``
        for both ``var`` and ``temp_col``. Required when either column
        is not in the etdmap data model. Use ``""`` for unitless.
    """
    if project_id is not None:
        df = df[df['ProjectIdBSV'] == project_id].copy()
    else:
        df = df.copy()

    if plot_var_name is None:
        plot_var_name = var

    # Strict empty-data guard. plot_var_vs_temp needs both axes' columns;
    # a row missing either is unusable. Caller is responsible for
    # ensuring coverage upstream.
    paired = df[[var, temp_col]].dropna()
    if paired.empty:
        raise ValueError(
            f"plot_var_vs_temp: no rows with both var={var!r} and "
            f"temp_col={temp_col!r} non-NA"
            f"{f' for project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    df = df.sort_values('ReadingDate')

    # Layer 3 strict-unit assertion on both axes.
    from etdanalyze.units import assert_unit_in_classes
    var_unit = resolve_unit(var, overrides=overrides)
    assert_unit_in_classes(
        var_unit, var,
        accepted_classes={"power", "temperature", "flow_rate", "state",
                          "ratio", "volume_per_interval"},
        plot_type="plot_var_vs_temp", axis="left y",
    )
    temp_unit_resolved = resolve_unit(temp_col, overrides=overrides)
    assert_unit_in_classes(
        temp_unit_resolved, temp_col,
        accepted_classes={"temperature"},
        plot_type="plot_var_vs_temp", axis="right y",
    )

    fig, ax1 = plt.subplots(figsize=(16, 12))
    ax2 = ax1.twinx()

    ax1.plot(df['ReadingDate'], df[var], linestyle='-', linewidth=0.75,
             label=f'Project {project_id}' if project_id is not None else var)
    ax1.set_ylabel(axis_label_with_unit(var, plot_var_name, overrides=overrides))

    ax2.plot(df['ReadingDate'], df[temp_col], color='orange', linewidth=0.75,
             label=axis_label_with_unit(temp_col, overrides=overrides))
    ax2.set_ylabel(
        axis_label_with_unit(temp_col, overrides=overrides),
        color='orange',
    )

    ax1.set_xlabel('Datum')

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate(rotation=45)

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_daily_profile(df, plot_var, title=None, plot_var_name=None, project_id=None, overrides=None, save_fig_path=None, convert_from_energy_per_interval=None):
    """
    Plot seasonal data for a specific project, showing statistical ranges and mean values.

    Unit handling is strict (Layer 3): the helper resolves ``plot_var``'s
    declared unit via the data model (or ``overrides``) and asserts it
    belongs to an instantaneous unit class (power / temperature /
    flow_rate / state / ratio / volume_per_interval). Energy-per-interval
    (kWh / Wh / GJ) is **not** accepted as a plot input -- a daily
    profile reads as power, not as "kWh per row".

    To plot an energy-per-interval column as a power profile, pass
    ``convert_from_energy_per_interval=<pd.Timedelta>``. The helper then
    L3-asserts the column is in the ``energy_per_interval`` class,
    delegates to ``etdanalyze.units.convert_energy_per_interval_to_power``
    (single source of truth for the conversion math), attaches the
    converted Series as a new derived column on a shallow copy of ``df``
    via ``df.assign`` (the input DataFrame is not mutated), and proceeds
    with the converted column. Memory cost: one new Series the length
    of the post-filter DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing daily profile with at least column "ProjectIdBSV".
    plot_var : str
        The column name representing the variable to plot.
    title : str
        Optional. Title to be used in the plot. If not defined the plot_var_name or plot_var is used.
    plot_var_name : str
        A descriptive name for the plot variable (for axis labeling). When
        omitted, the y-label is derived from etdmap unit metadata (or
        ``overrides``).
    project_id: int|str:
        Optional: when provided the df is filtered to only contain data of that project.
    overrides : dict, optional
        ``{col: unit}`` map forwarded to ``resolve_unit`` /
        ``axis_label_with_unit``. Required when ``plot_var`` is a
        derived variant not in the etdmap data model (use ``""`` for
        unitless).
    save_fig_path : str, optional
        File path to save the plot. If None, the figure is returned.
    convert_from_energy_per_interval : pd.Timedelta or str, optional
        Caller-declared row spacing of ``plot_var`` (e.g.
        ``pd.Timedelta(hours=1)`` for a frame loaded from
        ``household_60min.parquet``). When set, ``plot_var`` is
        converted from energy-per-interval to mean power before
        plotting. The caller passes the interval explicitly because
        they know it from the data's provenance; helpers do not
        infer.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided.
    """

    # add time_of_day column that sets the times correctly.
    df_normalized_dt = etdtransform.calculated_columns.add_normalized_datetime(df).copy()

    # Optional caller-declared conversion: energy-per-interval -> power.
    # Conversion math lives in etdanalyze.units; this helper just opts
    # the column in. Returns df unchanged when the kwarg is None.
    from etdanalyze.units import (
        assert_unit_in_classes,
        apply_optional_energy_conversion,
    )
    df_normalized_dt, plot_var, overrides = apply_optional_energy_conversion(
        df_normalized_dt, plot_var, convert_from_energy_per_interval,
        overrides, plot_type="plot_daily_profile",
    )

    # Layer 3 strict-unit assertion: a daily profile reads an
    # instantaneous-style value at each minute-of-day, so the y-axis
    # values must belong to an instantaneous unit class. The
    # conversion above, if any, has already replaced the working
    # column with a power Series so the assertion passes naturally.
    unit = resolve_unit(plot_var, overrides=overrides)
    assert_unit_in_classes(
        unit, plot_var,
        accepted_classes={"power", "power_per_area", "temperature",
                          "flow_rate", "state", "ratio",
                          "volume_per_interval"},
        plot_type="plot_daily_profile", axis="y",
    )

    # Strict empty-data guard: never render a silent empty plot. Caller
    # is responsible for upstream filtering / coverage-checking; helper
    # fails fast if the inputs would produce nothing to plot. Same
    # contract shape as the L3 unit assertion above -- caller declares
    # intent (via plot_var and optional project_id) and helper rejects
    # when intent doesn't match reality.
    if df_normalized_dt[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_daily_profile: no non-NA rows for plot_var={plot_var!r} "
            f"in the input. Check coverage upstream and skip the call."
        )
    if project_id is not None:
        proj_mask = df_normalized_dt["ProjectIdBSV"] == project_id
        if df_normalized_dt.loc[proj_mask, plot_var].notna().sum() == 0:
            raise ValueError(
                f"plot_daily_profile: no non-NA rows for plot_var={plot_var!r} "
                f"and project_id={project_id!r}. Check coverage and skip the "
                f"call for this project."
            )

    grouped = df_normalized_dt.groupby(["ProjectIdBSV", "time_of_day"])[plot_var]
    stats = ["mean", "median", "min", "max"]
    summary_df = grouped.agg(stats).reset_index()

    # Add quartiles (25th and 75th percentiles)
    quartiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quartiles.columns = ["ProjectIdBSV", "time_of_day", "p25", "p75"]
    summary_df = summary_df.merge(quartiles, on=["ProjectIdBSV", "time_of_day"], how="left")

    # Filter for the specific project
    if project_id is not None:
        summary_df = summary_df[summary_df["ProjectIdBSV"] == project_id]

    # Set up plot colors for different statistics
    cmap = plt.colormaps.get_cmap("autumn")
    stats_to_plot = ["min", "p25", "median", "p75", "max"]
    stat_colors = [cmap(i / (len(stats_to_plot) - 1) / 2) for i in range(len(stats_to_plot))]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot statistical ranges (min, p25, median, p75, max)
    for i, stat in enumerate(stats_to_plot):
        ax.plot(
            pd.to_datetime(summary_df["time_of_day"], format="%H:%M:%S"),
            summary_df[stat],
            linestyle="-",
            linewidth=0.8,
            color=stat_colors[i],
            alpha=0.5,
            label=stat,
        )

    # Highlight the mean value
    ax.plot(
        pd.to_datetime(summary_df["time_of_day"], format="%H:%M:%S"),
        summary_df["mean"],
        linestyle="-",
        linewidth=2,
        color="purple",
        label="mean",
    )

    # Strict label resolution: data-model unit or caller's overrides.
    # The Layer-3 assertion above has already verified the unit is in
    # an accepted class; this just composes the label.
    plot_var_name = axis_label_with_unit(
        plot_var, plot_var_name, overrides=overrides,
    )

    # Axis labels and formatting
    ax.set_ylabel(plot_var_name, color="purple")
    ax.set_xlabel("Time of Day")
    ax.tick_params(axis="y", labelcolor="purple")

    # Format x-axis to show time
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=45)

    # Add title and legend
    if title is not None:
        plt.title(f"{title}")
    else:
        plt.title(f"Daily profile: {plot_var_name}")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save or return figure
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_daily_profile_bands(
        df: pd.DataFrame,
        plot_var: str,
        title: Optional[str] = None,
        plot_var_name: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
        convert_from_energy_per_interval: "pd.Timedelta | str | None" = None,
):
    """
    Band-style daily profile (companion to ``plot_daily_profile``).

    Shows the same 24-hour-of-day shape, but with the canonical 4-layer
    band stack used by ``plot_weekly_profile`` / ``plot_annual_profile``
    -- min-max, p01-p99, p25-p75, median line, mean line. No per-stat
    line breakdown (use ``plot_daily_profile`` for that, kept side by
    side for callers that want the per-project stat lines).

    Both helpers share the same input contract, unit conventions, and
    strict empty-data guard. Use ``plot_daily_profile`` when you want
    to see each statistic as its own line; use this when you want the
    weekly / annual layout for visual consistency.

    Strict-unit handling: ``plot_var`` must resolve to an instantaneous
    unit class. Pass ``convert_from_energy_per_interval=<pd.Timedelta>``
    for the kWh -> kW opt-in. Same shape as the other profile helpers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'ReadingDate' datetime column and the
        ``plot_var`` column.
    plot_var : str
        Column to render on the y-axis.
    title : str, optional
        Composed by the caller (verbatim render). Defaults to
        ``"Daily profile (bands): {plot_var_name}"``.
    plot_var_name : str, optional
        Y-axis label. Defaults to ``plot_var``.
    project_id : int or str, optional
        When provided, filters df to that project before grouping by
        hour-of-day.
    overrides : dict, optional
        Per-call ``{col: unit}`` -- forwarded to ``resolve_unit`` /
        ``axis_label_with_unit``.
    save_fig_path : str, optional
        File path. If None, returns the figure.
    convert_from_energy_per_interval : pd.Timedelta or str, optional
        Opt-in energy-per-interval -> power conversion. See
        ``plot_daily_profile`` for the full contract.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    # Strict empty-data guard -- mirrors plot_weekly_profile.
    if df[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_daily_profile_bands: no non-NA rows for plot_var={plot_var!r}"
            f"{f' and project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    # Optional caller-declared energy-per-interval -> power conversion.
    from etdanalyze.units import (
        assert_unit_in_classes,
        apply_optional_energy_conversion,
    )
    df, plot_var, overrides = apply_optional_energy_conversion(
        df, plot_var, convert_from_energy_per_interval,
        overrides, plot_type="plot_daily_profile_bands",
    )

    df["hour_of_day"] = df["ReadingDate"].dt.hour
    grouped = df.groupby("hour_of_day")[plot_var]
    summary = grouped.agg(["mean", "min", "max"]).reset_index()
    summary["p01"] = grouped.quantile(0.01).values
    summary["p25"] = grouped.quantile(0.25).values
    summary["p75"] = grouped.quantile(0.75).values
    summary["p99"] = grouped.quantile(0.99).values
    summary["median"] = grouped.median().values

    fig, ax = plt.subplots(figsize=(12, 6))
    # Canonical 4-layer band stack -- see plot_weekly_profile.
    ax.fill_between(summary["hour_of_day"], summary["min"], summary["max"],
                    alpha=0.10, color="steelblue", label="min-max")
    ax.fill_between(summary["hour_of_day"], summary["p01"], summary["p99"],
                    alpha=0.18, color="steelblue", label="p01-p99")
    ax.fill_between(summary["hour_of_day"], summary["p25"], summary["p75"],
                    alpha=0.30, color="steelblue", label="p25-p75")
    ax.plot(summary["hour_of_day"], summary["median"],
            linewidth=0.9, color="steelblue", alpha=0.7, label="median")
    ax.plot(summary["hour_of_day"], summary["mean"],
            linewidth=2, color="purple", label="mean")

    # Layer 3 strict-unit assertion -- same accepted classes as the
    # other profile helpers.
    _unit = resolve_unit(plot_var, overrides=overrides)
    assert_unit_in_classes(
        _unit, plot_var,
        accepted_classes={"power", "power_per_area", "temperature",
                          "flow_rate", "state", "ratio",
                          "volume_per_interval"},
        plot_type="plot_daily_profile_bands", axis="y",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(axis_label_with_unit(plot_var, plot_var_name, overrides=overrides))
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.legend()
    plt.title(title or f"Daily profile (bands): {plot_var_name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_daily_profile_mean_combined(
        df: pd.DataFrame,
        plot_vars: list[str],
        title: Optional[str] = None,
        plot_var_names: Optional[list[str]] = None,
        y_label: Optional[str] = None,
        project_id: Optional[str|int] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
):
    """
    Plot the mean values of multiple variables over time for a specific project (if provided).

    For ``*Diff`` columns the per-row energy is converted to average power
    (kW) via :func:`etdanalyze.units.convert_diff_to_power`. When any Diff
    conversion ran the default y-label is "kW"; otherwise the first
    variable's data-model unit is used. The caller may always override the
    label and title explicitly.

    Note: this function does not divide values by floor area. If the
    caller has pre-divided the input series to produce a per-m2 quantity
    they should pass ``y_label`` to reflect that. (The previous default
    title ``"Variabelen per 100m2 (kW)"`` was misleading -- the function
    never enforced that scaling.)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data with columns including "ProjectIdBSV",
        and the variables to be plotted.
    plot_vars : list of str
        List of column names in `df` representing the variables to plot.
    title : str, optional
        Plot title. When omitted, derived from the first variable.
    plot_var_names : list of str, optional
        Custom display names for the variables in the legend. If not provided,
        the original column names from `plot_vars` will be used.
    y_label : str, optional
        Explicit y-axis label. Use this when the caller has pre-applied a
        transformation (e.g. per-m2 normalisation) that the helper cannot
        infer from the column name.
    project_id : int or str, optional
        Identifier of the project to filter the data. If `None`, all data is plotted.
    save_fig_path : str, optional
        File path to save the plot.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # add time_of_day column that sets the times correctly.
    df = etdtransform.calculated_columns.add_normalized_datetime(df).copy()

    # Strict empty-data guard. The combined view needs at least ONE
    # plot_var with data; if every requested variable is all-NA in df
    # the plot would be empty axes.
    plot_vars_with_data = [
        v for v in plot_vars
        if v in df.columns and df[v].notna().any()
    ]
    if not plot_vars_with_data:
        raise ValueError(
            f"plot_daily_profile_mean_combined: none of plot_vars={plot_vars!r} "
            f"have non-NA rows in df"
            f"{f' (project_id={project_id!r})' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    # Layer 3 strict-unit assertion (per variable). A combined daily
    # profile reads as a stacked instantaneous view; energy-per-
    # interval is rejected here for the same reason as in
    # plot_daily_profile -- callers convert upstream and declare the
    # resulting power unit via ``overrides``.
    from etdanalyze.units import assert_unit_in_classes
    for var in plot_vars:
        if var not in df.columns:
            continue
        unit = resolve_unit(var, overrides=overrides)
        assert_unit_in_classes(
            unit, var,
            accepted_classes={"power", "temperature", "flow_rate", "state",
                              "ratio", "volume_per_interval"},
            plot_type="plot_daily_profile_mean_combined", axis="y",
        )
    converted_units: dict = {}

    fig, ax = plt.subplots(figsize=(16, 12))
    colors = [plt.cm.viridis_r(i / (len(plot_vars))) for i in range(len(plot_vars))]  # Assign distinct colors

    # if no distinct names for plot labels are supplied
    # we use the original var names as label-name
    if plot_var_names is None:
         plot_var_names = plot_vars

    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id]

    for var, plot_var_name, color in zip(plot_vars, plot_var_names, colors):
        grouped = df.groupby(["time_of_day"])[var]
        mean_df = grouped.mean().reset_index()

        ax.plot(
            pd.to_datetime(mean_df["time_of_day"], format="%H:%M:%S"),
            mean_df[var],
            linestyle="-",
            linewidth=2,
            label=plot_var_name,
            color=color
        )

    ax.set_xlabel("Tijd")
    # y-label: caller override wins; otherwise look up the first variable's
    # unit (using its converted unit if Diff conversion ran). Mixing
    # variables with different units in one plot is the caller's call --
    # we label by the first one only.
    if y_label is None:
        if not plot_vars:
            raise ValueError("plot_daily_profile_mean_combined: plot_vars is empty.")
        first = plot_vars[0]
        first_overrides = {first: converted_units[first]} if first in converted_units else None
        y_label = axis_label_with_unit(first, overrides=first_overrides)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.legend()

    plt.title(f"{title}")
    plt.grid(True)
    plt.tight_layout()

    # Save or return figure
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_weekly_profile(
        df: pd.DataFrame,
        plot_var: str,
        title: Optional[str] = None,
        plot_var_name: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
        convert_from_energy_per_interval: "pd.Timedelta | str | None" = None,
):
    """
    Plot the weekly profile (weekprofiel) for a variable.

    Shows mean values for each (day-of-week, hour-of-day) combination,
    producing 168 data points across 7 day panels on a single x-axis.
    Min/p25/median/p75/max bands are shown as shading and thin lines.

    Strict-unit handling: ``plot_var`` must resolve to an instantaneous
    unit class (power / temperature / flow_rate / state / ratio /
    volume_per_interval). Pass ``convert_from_energy_per_interval=
    <pd.Timedelta>`` to opt-in to energy-per-interval -> power
    conversion via the shared helper -- see ``plot_daily_profile`` for
    the full contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'ReadingDate' column and the plot_var column.
    plot_var : str
        Column name of the variable to plot.
    title : str, optional
        Plot title. Defaults to "Weekly profile: {plot_var_name}".
    plot_var_name : str, optional
        Y-axis label. Defaults to plot_var.
    project_id : int or str, optional
        When provided, filters df to that project.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.
    convert_from_energy_per_interval : pd.Timedelta or str, optional
        Opt-in energy-per-interval -> mean-power conversion. See
        ``plot_daily_profile`` for the full contract.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    # Strict empty-data guard. The project_id filter has already run --
    # an empty post-filter df means this (var, project) combination has
    # no data and the caller should have skipped the call upstream.
    if df[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_weekly_profile: no non-NA rows for plot_var={plot_var!r}"
            f"{f' and project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    # Optional caller-declared energy-per-interval -> power conversion.
    from etdanalyze.units import (
        assert_unit_in_classes,
        apply_optional_energy_conversion,
    )
    df, plot_var, overrides = apply_optional_energy_conversion(
        df, plot_var, convert_from_energy_per_interval,
        overrides, plot_type="plot_weekly_profile",
    )

    df["day_of_week"] = df["ReadingDate"].dt.dayofweek   # 0=Mon, 6=Sun
    df["hour_of_day"] = df["ReadingDate"].dt.hour
    df["week_slot"] = df["day_of_week"] * 24 + df["hour_of_day"]

    grouped = df.groupby("week_slot")[plot_var]
    summary = grouped.agg(["mean", "min", "max"]).reset_index()
    summary["p01"] = grouped.quantile(0.01).values
    summary["p25"] = grouped.quantile(0.25).values
    summary["p75"] = grouped.quantile(0.75).values
    summary["p99"] = grouped.quantile(0.99).values
    summary["median"] = grouped.median().values

    fig, ax = plt.subplots(figsize=(16, 6))
    # Canonical 4-layer band stack (ADR-002 / Group B):
    # min-max (lightest) > p01-p99 > p25-p75 (darkest), then median +
    # mean lines on top. Same colour, varying alpha so the visual
    # hierarchy reads from "outliers" to "typical" without legend
    # reference.
    ax.fill_between(summary["week_slot"], summary["min"], summary["max"],
                    alpha=0.10, color="steelblue", label="min-max")
    ax.fill_between(summary["week_slot"], summary["p01"], summary["p99"],
                    alpha=0.18, color="steelblue", label="p01-p99")
    ax.fill_between(summary["week_slot"], summary["p25"], summary["p75"],
                    alpha=0.30, color="steelblue", label="p25-p75")
    ax.plot(summary["week_slot"], summary["median"],
            linewidth=0.9, color="steelblue", alpha=0.7, label="median")
    ax.plot(summary["week_slot"], summary["mean"],
            linewidth=2, color="purple", label="mean")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in range(7):
        ax.axvline(day * 24, color="gray", linewidth=0.5, linestyle="--")

    # Layer 3 strict-unit assertion -- same expectations as
    # plot_daily_profile: a weekly profile's y-axis reads as an
    # instantaneous value at each (day, hour) cell.
    _unit = resolve_unit(plot_var, overrides=overrides)
    assert_unit_in_classes(
        _unit, plot_var,
        accepted_classes={"power", "power_per_area", "temperature",
                          "flow_rate", "state", "ratio",
                          "volume_per_interval"},
        plot_type="plot_weekly_profile", axis="y",
    )
    ax.set_xlabel("Day of week / hour")
    ax.set_ylabel(axis_label_with_unit(plot_var, plot_var_name, overrides=overrides))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: day_names[int(x // 24)] if 0 <= int(x // 24) < 7 else ""
    ))
    ax.legend()
    plt.title(title or f"Weekly profile: {plot_var_name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_annual_profile(
        df: pd.DataFrame,
        plot_var: str,
        title: Optional[str] = None,
        plot_var_name: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
        convert_from_energy_per_interval: "pd.Timedelta | str | None" = None,
):
    """
    Plot the annual profile (jaarprofiel) for a variable.

    Shows mean and p25-p75 band for each day-of-year (1-366),
    revealing seasonal patterns across the year.

    Strict-unit handling: ``plot_var`` must resolve to an instantaneous
    unit class. Pass ``convert_from_energy_per_interval=<pd.Timedelta>``
    to opt-in to energy-per-interval -> power conversion via the
    shared helper -- see ``plot_daily_profile`` for the full contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'ReadingDate' column and the plot_var column.
    plot_var : str
        Column name of the variable to plot.
    title : str, optional
        Plot title. Defaults to "Annual profile: {plot_var_name}".
    plot_var_name : str, optional
        Y-axis label. Defaults to plot_var.
    project_id : int or str, optional
        When provided, filters df to that project.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.
    convert_from_energy_per_interval : pd.Timedelta or str, optional
        Opt-in energy-per-interval -> mean-power conversion. See
        ``plot_daily_profile`` for the full contract.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    # Strict empty-data guard -- see plot_weekly_profile for the contract.
    if df[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_annual_profile: no non-NA rows for plot_var={plot_var!r}"
            f"{f' and project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    # Optional caller-declared energy-per-interval -> power conversion.
    from etdanalyze.units import (
        assert_unit_in_classes,
        apply_optional_energy_conversion,
    )
    df, plot_var, overrides = apply_optional_energy_conversion(
        df, plot_var, convert_from_energy_per_interval,
        overrides, plot_type="plot_annual_profile",
    )
    df["day_of_year"] = df["ReadingDate"].dt.dayofyear

    grouped = df.groupby("day_of_year")[plot_var]
    summary = grouped.agg(["mean", "min", "max"]).reset_index()
    summary["p01"] = grouped.quantile(0.01).values
    summary["p25"] = grouped.quantile(0.25).values
    summary["p75"] = grouped.quantile(0.75).values
    summary["p99"] = grouped.quantile(0.99).values
    summary["median"] = grouped.median().values

    fig, ax = plt.subplots(figsize=(16, 6))
    # Canonical 4-layer band stack -- see plot_weekly_profile.
    ax.fill_between(summary["day_of_year"], summary["min"], summary["max"],
                    alpha=0.10, color="steelblue", label="min-max")
    ax.fill_between(summary["day_of_year"], summary["p01"], summary["p99"],
                    alpha=0.18, color="steelblue", label="p01-p99")
    ax.fill_between(summary["day_of_year"], summary["p25"], summary["p75"],
                    alpha=0.30, color="steelblue", label="p25-p75")
    ax.plot(summary["day_of_year"], summary["median"],
            linewidth=0.9, color="steelblue", alpha=0.7, label="median")
    ax.plot(summary["day_of_year"], summary["mean"],
            linewidth=2, color="purple", label="mean")

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for start, name in zip(month_starts, month_names):
        ax.axvline(start, color="gray", linewidth=0.4, linestyle="--")
        ax.text(start + 0.5, ax.get_ylim()[1], name, fontsize=7, color="gray", va="top")

    # Layer 3 strict-unit assertion -- annual profile reads as an
    # instantaneous value at each day-of-year position.
    _unit = resolve_unit(plot_var, overrides=overrides)
    assert_unit_in_classes(
        _unit, plot_var,
        accepted_classes={"power", "power_per_area", "temperature",
                          "flow_rate", "state", "ratio",
                          "volume_per_interval"},
        plot_type="plot_annual_profile", axis="y",
    )
    ax.set_xlabel("Day of year")
    ax.set_ylabel(axis_label_with_unit(plot_var, plot_var_name, overrides=overrides))
    ax.legend()
    plt.title(title or f"Annual profile: {plot_var_name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def _ols_overlay(ax, x: pd.Series, y: pd.Series, label: str = "OLS fit") -> None:
    """Overlay a simple linear OLS fit on an existing scatter axis.

    Uses pandas ``.cov() / .var()`` (per ADR-005 / parent ADR-005 -- no
    numpy import in analytics) and skips silently when there is too
    little data or zero variance to fit.

    The fit reduces visual density on COP scatter plots (MW100): the
    eye reads the slope rather than tracking individual dots.
    """
    if len(x) < 5:
        return
    try:
        var_x = float(x.var())
        if not (var_x > 0):
            return
        slope = float(x.cov(y) / var_x)
        intercept = float(y.mean() - slope * x.mean())
    except Exception:
        return
    x_min, x_max = float(x.min()), float(x.max())
    if not (x_max > x_min):
        return
    x_line = pd.Series([x_min, x_max])
    y_line = slope * x_line + intercept
    # R^2 for the legend label
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    ax.plot(
        x_line, y_line,
        color="tomato", linewidth=2,
        label=f"{label}: slope={slope:.3f}, R^2={r2:.2f}",
    )
    ax.legend(loc="best", fontsize=9)


def plot_cop_vs_temp(
        df: pd.DataFrame,
        cop_col: str = "COP",
        temp_col: str = "TemperatuurBuitenWarmtepomp",
        title: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
        alpha: float = 0.4,
        y_clip: Optional[tuple] = None,
):
    """
    Scatter plot of daily average COP against outdoor temperature.

    Each point represents one day. Requires daily-aggregated input
    (compute daily means before calling).

    The ``alpha`` parameter controls per-point opacity (default 0.4,
    matching the original per-project rendering). For dense pooled
    views (whole-dataset, all households x all days = 10k+ points)
    pass a low value (e.g. ``alpha=0.08``) so overplot density
    reveals cloud shape without hiding outliers via aggregation
    (MW100 strict-disaggregation approach).

    ``y_clip=(ymin, ymax)`` sets ``ax.set_ylim`` *only* -- impossible
    COP outliers (divide-by-near-zero artifacts on mostly-standby
    days) are kept in the data and in the OLS-trend computation; they
    are just visually clipped to keep the physically-plausible cloud
    readable. If the trend ends up flat-with-R^2=0, that is a real
    signal that outliers dominate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cop_col and temp_col columns (daily averages).
    cop_col : str, optional
        Column name for COP values. Default is 'COP'.
    temp_col : str, optional
        Column name for temperature values. Default is 'TemperatuurBuitenWarmtepomp'.
    title : str, optional
        Plot title.
    project_id : int or str, optional
        When provided, filters df to that project.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()

    plot_df = df[[temp_col, cop_col]].dropna()
    # Strict empty-data guard. The scatter is useless without paired
    # (temp, cop) observations; caller is responsible for checking
    # coverage upstream (e.g. via classify_coverage) and skipping the
    # call when none exist. Today's silent-empty-scatter behaviour
    # (visible in the project 1 / 8 cop_vs_temp PNGs) is what this
    # contract surfaces loudly.
    if plot_df.empty:
        raise ValueError(
            f"plot_cop_vs_temp: no rows with both temp_col={temp_col!r} "
            f"and cop_col={cop_col!r} non-NA"
            f"{f' for project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df[temp_col], plot_df[cop_col],
               alpha=alpha, s=20, color="steelblue", label="dagelijks")
    # OLS trend: when y_clip is set, compute on data within the
    # visible band so the trend reflects the physically-plausible
    # cloud the reader can see. Without this, impossible-COP outliers
    # (visually clipped above the y-axis) still dominate the OLS and
    # produce a flat line with R^2=0 that is unhelpful for the
    # visible population. Outliers remain in the scatter (just clipped
    # visually); we do not hide them, we just don't let them drag the
    # trend line.
    if y_clip is not None:
        trend_df = plot_df[
            (plot_df[cop_col] >= y_clip[0])
            & (plot_df[cop_col] <= y_clip[1])
        ]
    else:
        trend_df = plot_df
    if len(trend_df) >= 2:
        _ols_overlay(ax, trend_df[temp_col], trend_df[cop_col], label="OLS-trend")
    # Layer 3 strict-unit assertions -- COP is dimensionless (ratio);
    # outdoor temp is a temperature.
    from etdanalyze.units import assert_unit_in_classes
    assert_unit_in_classes(
        resolve_unit(cop_col, overrides=overrides), cop_col,
        accepted_classes={"ratio"},
        plot_type="plot_cop_vs_temp", axis="y",
    )
    assert_unit_in_classes(
        resolve_unit(temp_col, overrides=overrides), temp_col,
        accepted_classes={"temperature"},
        plot_type="plot_cop_vs_temp", axis="x",
    )
    ax.set_xlabel(axis_label_with_unit(temp_col,
                                       base_label=f"Outdoor temperature [{temp_col}]",
                                       overrides=overrides))
    ax.set_ylabel(axis_label_with_unit(cop_col, base_label=f"COP [{cop_col}]",
                                       overrides=overrides))
    if y_clip is not None:
        ax.set_ylim(*y_clip)
    plt.title(title or "COP vs outdoor temperature")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_cop_vs_delta_t(
        df: pd.DataFrame,
        cop_col: str = "COP",
        supply_temp_col: str = "TemperatuurAfgifteAanvoer",
        outdoor_temp_col: str = "TemperatuurBuitenWarmtepomp",
        title: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
):
    """
    Scatter plot of daily average COP against the temperature lift (delta-T).

    Delta-T = TemperatuurAfgifteAanvoer - TemperatuurBuitenWarmtepomp.
    Relevant for air-source heat pumps where efficiency depends on the
    difference between supply and source temperature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cop_col, supply_temp_col, and outdoor_temp_col columns.
    cop_col : str, optional
        Column name for COP values. Default is 'COP'.
    supply_temp_col : str, optional
        Supply (flow) temperature column. Default is 'TemperatuurAfgifteAanvoer'.
    outdoor_temp_col : str, optional
        Outdoor temperature column. Default is 'TemperatuurBuitenWarmtepomp'.
    title : str, optional
        Plot title.
    project_id : int or str, optional
        When provided, filters df to that project.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()

    df = df.copy()
    df["delta_T"] = df[supply_temp_col] - df[outdoor_temp_col]
    plot_df = df[["delta_T", cop_col]].dropna()
    # Strict empty-data guard -- see plot_cop_vs_temp for the contract.
    if plot_df.empty:
        raise ValueError(
            f"plot_cop_vs_delta_t: no rows with non-NA cop_col={cop_col!r} "
            f"and valid delta_T (from {supply_temp_col!r} - {outdoor_temp_col!r})"
            f"{f' for project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df["delta_T"], plot_df[cop_col],
               alpha=0.4, s=20, color="steelblue", label="dagelijks")
    _ols_overlay(ax, plot_df["delta_T"], plot_df[cop_col], label="OLS-trend")
    # Layer 3 strict-unit assertions -- delta_T is a temperature
    # difference (same class as a temperature); COP is dimensionless.
    from etdanalyze.units import assert_unit_in_classes
    assert_unit_in_classes(
        resolve_unit(cop_col, overrides=overrides), cop_col,
        accepted_classes={"ratio"},
        plot_type="plot_cop_vs_delta_t", axis="y",
    )
    assert_unit_in_classes(
        resolve_unit("delta_T", overrides=overrides), "delta_T",
        accepted_classes={"temperature"},
        plot_type="plot_cop_vs_delta_t", axis="x",
    )
    ax.set_xlabel(axis_label_with_unit(
        "delta_T",
        base_label=f"Temperature lift dT [{supply_temp_col} - {outdoor_temp_col}]",
        overrides=overrides,
    ))
    ax.set_ylabel(axis_label_with_unit(
        cop_col, base_label=f"COP [{cop_col}]", overrides=overrides,
    ))
    plt.title(title or "COP vs temperature lift (dT)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_energy_signature(
        df: pd.DataFrame,
        energy_col: str,
        temp_col: str,
        fit: Optional[dict] = None,
        title: Optional[str] = None,
        energy_label: Optional[str] = None,
        project_id: Optional[int | str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
):
    """
    Scatter plot of energy use per m² against outdoor temperature (energy signature).

    Optionally overlays the fitted heating and DHW regression lines from
    fit_heat_signature(), including the stookgrens (heating cut-off temperature).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with energy_col and temp_col columns (typically monthly averages).
    energy_col : str
        Column with energy values (e.g. W/m² or kWh/m²).
    temp_col : str
        Column with outdoor temperature (°C).
    fit : dict, optional
        Output of analysis_helpers.fit_heat_signature(). When provided, regression
        lines for heating and DHW are drawn and the stookgrens is annotated.
    title : str, optional
        Plot title. Defaults to "Energy signature".
    energy_label : str, optional
        Y-axis label. Defaults to energy_col.
    project_id : int or str, optional
        When provided, filters df to that project.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    # Layer 3 strict-unit assertions -- energy signature plots
    # energy- or power-per-area on the y-axis against temperature
    # on the x-axis.
    from etdanalyze.units import assert_unit_in_classes
    assert_unit_in_classes(
        resolve_unit(energy_col, overrides=overrides), energy_col,
        accepted_classes={"power_per_area", "energy_per_area"},
        plot_type="plot_energy_signature", axis="y",
    )
    assert_unit_in_classes(
        resolve_unit(temp_col, overrides=overrides), temp_col,
        accepted_classes={"temperature"},
        plot_type="plot_energy_signature", axis="x",
    )
    if energy_label is None:
        energy_label = axis_label_with_unit(energy_col, overrides=overrides)

    plot_df = df[[temp_col, energy_col]].dropna()
    # Strict empty-data guard. An energy-signature scatter needs paired
    # (temp, energy) observations to be meaningful; raise loudly when
    # the caller's project / variable selection produced none.
    if plot_df.empty:
        raise ValueError(
            f"plot_energy_signature: no rows with both temp_col={temp_col!r} "
            f"and energy_col={energy_col!r} non-NA"
            f"{f' for project_id={project_id!r}' if project_id is not None else ''}. "
            f"Check coverage upstream and skip the call."
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df[temp_col], plot_df[energy_col],
               alpha=0.5, s=25, color="steelblue", label="Monthly avg")

    if fit is not None:
        t_min, t_max = plot_df[temp_col].min(), plot_df[temp_col].max()
        t_line = pd.Series([t_min, t_max])

        if fit.get("heating_slope") is not None:
            y_line = fit["heating_slope"] * t_line + fit["heating_intercept"]
            ax.plot(t_line, y_line, color="tomato", linewidth=2, label="Heating fit")

        if fit.get("dhw_level") is not None:
            ax.axhline(fit["dhw_level"], color="orange", linewidth=1.5,
                       linestyle="--", label=f"DHW level ({fit['dhw_level']:.1f})")

        if fit.get("stookgrens") is not None:
            sg = fit["stookgrens"]
            ax.axvline(sg, color="gray", linewidth=1, linestyle=":")
            # Stookgrens unit follows the x-axis temperature column.
            # Use resolve_unit so derived temp_col gets the same strict
            # treatment as the axis label (no silent empty unit).
            ax.text(sg + 0.2, ax.get_ylim()[0],
                    f"Stookgrens\n{sg:.1f} {resolve_unit(temp_col, overrides=overrides)}",
                    fontsize=8, color="gray", va="bottom")

    ax.set_xlabel(axis_label_with_unit(temp_col,
                                       base_label=f"Outdoor temperature [{temp_col}]",
                                       overrides=overrides))
    ax.set_ylabel(energy_label)
    ax.legend()
    plt.title(title or "Energy signature")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig


def plot_monthly_profile(
        df: pd.DataFrame,
        plot_var: str,
        title: Optional[str] = None,
        plot_var_name: Optional[str] = None,
        overrides: Optional[dict] = None,
        save_fig_path: Optional[str] = None,
        convert_from_energy_per_interval: "pd.Timedelta | str | None" = None,
):
    """
    Plot the monthly profile (maandprofiel) as a heatmap: 12 months x 24 hours = 288 slots.

    Each cell shows the mean value of plot_var for that (month, hour-of-day) combination,
    aggregated over all rows in df. The heatmap is compact and directly comparable across
    variables and domains.

    Strict-unit handling: ``plot_var`` must resolve to an instantaneous
    unit class. Pass ``convert_from_energy_per_interval=<pd.Timedelta>``
    to opt-in to energy-per-interval -> power conversion via the
    shared helper -- see ``plot_daily_profile`` for the full contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'ReadingDate' datetime column and the plot_var column.
    plot_var : str
        Column name of the variable to plot.
    title : str, optional
        Plot title. Defaults to "Monthly profile: {plot_var_name}".
    plot_var_name : str, optional
        Colorbar label and title suffix. Defaults to plot_var.
    save_fig_path : str, optional
        File path to save the figure. If None, the figure is returned.
    convert_from_energy_per_interval : pd.Timedelta or str, optional
        Opt-in energy-per-interval -> mean-power conversion. See
        ``plot_daily_profile`` for the full contract.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    # Strict empty-data guard -- monthly profile has no internal
    # project_id filter, so the check is just the input itself.
    if df[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_monthly_profile: no non-NA rows for plot_var={plot_var!r}. "
            f"Check coverage upstream and skip the call."
        )

    # Optional caller-declared energy-per-interval -> power conversion.
    from etdanalyze.units import (
        assert_unit_in_classes,
        apply_optional_energy_conversion,
    )
    df, plot_var, overrides = apply_optional_energy_conversion(
        df, plot_var, convert_from_energy_per_interval,
        overrides, plot_type="plot_monthly_profile",
    )

    df["month"] = df["ReadingDate"].dt.month
    df["hour_of_day"] = df["ReadingDate"].dt.hour

    grouped = df.groupby(["month", "hour_of_day"])[plot_var].mean().reset_index()
    pivot = grouped.pivot(index="hour_of_day", columns="month", values=plot_var)

    # Ensure all 12 months and 24 hours are represented
    pivot = pivot.reindex(index=range(24), columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="upper",
        cmap="RdYlBu_r",
        interpolation="nearest",
    )
    # Layer 3 strict-unit assertion -- monthly heatmap encodes
    # variable value via colour. Accepts the same instantaneous
    # classes as the other profile plots.
    _unit = resolve_unit(plot_var, overrides=overrides)
    assert_unit_in_classes(
        _unit, plot_var,
        accepted_classes={"power", "power_per_area", "temperature",
                          "flow_rate", "state", "ratio",
                          "volume_per_interval"},
        plot_type="plot_monthly_profile", axis="colorbar",
    )
    plt.colorbar(im, ax=ax,
                 label=axis_label_with_unit(plot_var, plot_var_name, overrides=overrides))

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    plt.title(title or f"Monthly profile: {plot_var_name}")
    plt.tight_layout()

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def _draw_boxes(
        ax,
        plot_df: pd.DataFrame,
        value_col: str,
        x_group: str,
        color_group: str,
        x_cats: list,
        c_cats: list,
        c_colors: dict,
        rng: np.random.Generator,
        outlier_ids: Optional[set],
        id_col: str,
        xtick_prefix: str = "",
) -> None:
    """
    Internal: draw boxes plus per-row scatter on a single axes. Used by
    plot_boxwhisker_comparison for both single-axis and split-axis paths.

    When outlier_ids is given, only rows whose id_col value is in the set are
    overlaid as 'x' markers. Otherwise the whole population is jittered as
    coloured dots (the original behaviour).
    """
    x_pos = {cat: i for i, cat in enumerate(x_cats)}

    box_data = [
        plot_df[plot_df[x_group] == cat][value_col].dropna().values
        for cat in x_cats
    ]
    ax.boxplot(
        box_data,
        positions=range(len(x_cats)),
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        boxprops=dict(facecolor="whitesmoke", alpha=0.7),
        flierprops=dict(marker="none"),
        whiskerprops=dict(linewidth=1.0, color="grey"),
        capprops=dict(linewidth=1.0, color="grey"),
    )

    legend_added = set()
    if outlier_ids is not None and id_col in plot_df.columns:
        # Outlier-only overlay: 'x' markers, no whole-population scatter.
        ids = set(outlier_ids)
        outlier_df = plot_df[plot_df[id_col].isin(ids)]
        for c_cat in c_cats:
            c_df = outlier_df[outlier_df[color_group] == c_cat]
            for x_cat in x_cats:
                vals = c_df[c_df[x_group] == x_cat][value_col].dropna().values
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.18, 0.18, len(vals))
                label = str(c_cat) if c_cat not in legend_added else "_nolegend_"
                if c_cat not in legend_added:
                    legend_added.add(c_cat)
                ax.scatter(
                    x_pos[x_cat] + jitter, vals,
                    color=c_colors[c_cat],
                    marker="x", s=60, linewidths=1.5, alpha=0.9, zorder=5,
                    label=label,
                )
    else:
        for c_cat in c_cats:
            c_df = plot_df[plot_df[color_group] == c_cat]
            for x_cat in x_cats:
                vals = c_df[c_df[x_group] == x_cat][value_col].dropna().values
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.22, 0.22, len(vals))
                label = str(c_cat) if c_cat not in legend_added else "_nolegend_"
                if c_cat not in legend_added:
                    legend_added.add(c_cat)
                ax.scatter(
                    x_pos[x_cat] + jitter, vals,
                    color=c_colors[c_cat],
                    alpha=0.65, s=22, linewidths=0,
                    label=label, zorder=3,
                )

    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(
        [f"{xtick_prefix}{c}" for c in x_cats], rotation=30, ha="right",
    )
    ax.grid(True, axis="y", alpha=0.3)


DEFAULT_BOXWHISKER_TITLE_TPL = (
    "per-huishouden {value_col}: {variable}{unit_paren} per {x_group}"
)
DEFAULT_BOXWHISKER_YLABEL_TPL = (
    "per-huishouden {value_col} -- {variable}{unit_paren}"
)


def plot_boxwhisker_comparison(
        df: pd.DataFrame,
        value_col: str,
        x_group: str,
        color_group: str,
        *,
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        title_params: Optional[dict] = None,
        wrap: bool = True,
        jitter_seed: int = 42,
        save_fig_path: Optional[str] = None,
        column_name: Optional[str] = None,
        overrides: Optional[dict] = None,
        outlier_ids: Optional[set] = None,
        id_col: str = "HuisIdBSV",
        secondary_axis_data: Optional[pd.DataFrame] = None,
        secondary_axis_label: Optional[str] = None,
        secondary_axis_value_col: Optional[str] = None,
        xtick_prefix: str = "",
) -> Optional[plt.Figure]:
    """
    Box-and-whisker plot with jittered scatter overlay for distribution
    comparison. Produces a single figure (one PNG per call).

    Designed for two standard views (ADR-005):
      1. x=ProjectIdBSV, color=WoningType  -> cross-project distribution
      2. x=WoningType, color=ProjectIdBSV  -> housing-type distribution

    Each x-category gets one box (IQR + whiskers, no outlier markers).
    Individual data points are overlaid as jittered dots coloured by
    color_group, or, when ``outlier_ids`` is given, only those ids
    appear as 'x' markers (used when a separate outlier table is
    rendered alongside).

    When ``secondary_axis_data`` is provided, the figure becomes a
    2-row stacked subplot sharing the x-axis: boxplot on top
    (height ratio 3), companion-metric scatter beneath
    (height ratio 1) with its own y-axis label. This is the
    layout used by the mapping report to show validate-fail-rate
    next to the value distribution.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation unit (e.g. per household), with at
        least value_col, x_group, and color_group columns.
    value_col : str
        Numeric column to plot on the y-axis of the main subplot.
    x_group : str
        Categorical column for x-axis positioning (e.g. 'ProjectIdBSV').
    color_group : str
        Categorical column for dot colour coding.
    title : str, optional
        Title TEMPLATE rendered via ``str.format()``. Available
        placeholders:
          - ``{variable}``: ``column_name`` if given, else ``value_col``
          - ``{unit}``: unit string (``""`` if unitless), resolved via
            ``resolve_unit(column_name, overrides)``
          - ``{unit_paren}``: convenience -- ``" (unit)"`` if non-empty,
            ``""`` if unitless. Use this in default-style templates
            so unitless variables don't render dangling ``()``.
          - ``{value_col}``, ``{x_group}``: as named.
          - any keys in ``title_params``.
        Default: ``"per-huishouden {value_col}: {variable}{unit_paren} per {x_group}"``.
        STRICT CONTRACT: when ``column_name`` is given the helper owns
        the unit -- do NOT pre-stamp a unit suffix in the template
        text. Pre-stamping plus ``{unit_paren}`` produces a visibly
        doubled unit, which is the loud failure mode by design.
    y_label : str, optional
        Y-axis label TEMPLATE. Same placeholders as ``title``.
        Default: ``"per-huishouden {value_col} -- {variable}{unit_paren}"``.
    title_params : dict, optional
        Extra named values for the title / y_label templates (e.g.
        ``{"period": "year"}`` to enable ``"... - {period}"`` in your
        template).
    wrap : bool, default True
        Wrap the title to the figure width (``ax.set_title(wrap=True)``)
        and the y-label to a fixed character width via ``textwrap.fill``.
        Disable only when callers need raw control over line breaks.
    column_name : str, optional
        ETD data model column name (or derived variable when paired
        with ``overrides``). Drives the ``{variable}`` / ``{unit}`` /
        ``{unit_paren}`` placeholders.
    overrides : dict, optional
        Per-call ``{column_name: unit}`` mapping forwarded to
        ``resolve_unit``. Required when ``column_name`` is a derived
        variable not in the etdmap data model (e.g. ``COP_system``,
        ``ZonopwekBrutoPer100M2KW``). Use ``""`` to mark unitless.
        The strict ``resolve_unit`` contract is preserved -- absence
        of a column from BOTH the data model AND ``overrides`` raises
        ``KeyError`` so missing-unit drift surfaces loudly.
    outlier_ids : set, optional
        If given, the whole-population scatter is replaced with 'x'
        markers showing only the rows whose id_col value is in this
        set.
    id_col : str, optional
        Column holding the id used by outlier_ids. Default 'HuisIdBSV'.
    secondary_axis_data : pd.DataFrame, optional
        Extra per-row data plotted as grey triangles on a *separate
        stacked subplot* below the main plot (NOT a twinx() overlay).
        Must contain x_group and the column named by
        secondary_axis_value_col.
    secondary_axis_label : str, optional
        Y-axis label for the secondary subplot.
    secondary_axis_value_col : str, optional
        Column in secondary_axis_data holding the secondary-axis values.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    select_cols = list(dict.fromkeys([value_col, x_group, color_group]))
    if outlier_ids is not None and id_col in df.columns and id_col not in select_cols:
        select_cols = select_cols + [id_col]
    plot_df = df[[c for c in select_cols if c in df.columns]].dropna(subset=[value_col])
    # Strict empty-data guard: no silent empty plots. Caller is responsible
    # for upstream coverage filtering; helper fails fast so the missing-data
    # case surfaces loudly and isn't disguised as a silent return.
    if plot_df.empty:
        raise ValueError(
            f"plot_boxwhisker_comparison: no non-NA rows for value_col="
            f"{value_col!r} in the input. Check coverage upstream and skip "
            f"the call."
        )

    # Legend honesty: x_cats / c_cats are built from values that actually
    # survive the dropna above, so a (project, woningtype) combo with no
    # observations never reaches the legend or the x-axis.
    x_cats = sorted(plot_df[x_group].dropna().unique().tolist(), key=str)
    c_cats = sorted(plot_df[color_group].dropna().unique().tolist(), key=str)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    c_colors = {cat: cmap(i % 10) for i, cat in enumerate(c_cats)}
    rng = np.random.default_rng(jitter_seed)

    # Resolve labels via strict template rendering. Boxplot draws ONE
    # POINT PER HOUSEHOLD using value_col (typically a per-household
    # summary statistic like "mean"); reviewers mistakenly read these
    # plots as distributions of raw values, so the default templates
    # name the statistic explicitly in both the title and the y-axis.
    #
    # Strict-NA-style contract: ``resolve_unit`` is called ONCE per call
    # (not once per label), and raises ``KeyError`` if column_name is
    # neither in the data model nor in ``overrides``. There is no
    # silent-skip guard on the title rendering.
    if column_name:
        unit = resolve_unit(column_name, overrides=overrides)
        variable = column_name
    else:
        unit = ""
        variable = value_col
    unit_paren = f" ({unit})" if unit else ""

    fmt_params = {
        "variable":   variable,
        "unit":       unit,
        "unit_paren": unit_paren,
        "value_col":  value_col,
        "x_group":    x_group,
        **(title_params or {}),
    }
    title_tpl   = title   if title   is not None else DEFAULT_BOXWHISKER_TITLE_TPL
    y_label_tpl = y_label if y_label is not None else DEFAULT_BOXWHISKER_YLABEL_TPL
    try:
        effective_title = title_tpl.format(**fmt_params)
        effective_y     = y_label_tpl.format(**fmt_params)
    except KeyError as missing:
        raise ValueError(
            f"plot_boxwhisker_comparison: title/y_label template references "
            f"placeholder {missing!s} which is not in title_params nor in "
            f"the auto-filled placeholders {{variable}}, {{unit}}, "
            f"{{unit_paren}}, {{value_col}}, {{x_group}}. Pass it via "
            f"title_params={{...}}."
        ) from None

    fig_width = max(8.0, len(x_cats) * 1.6 + 2.0)
    has_secondary = (
        secondary_axis_data is not None
        and secondary_axis_value_col is not None
        and not secondary_axis_data.empty
    )

    if has_secondary:
        # 2-row stacked subplot. Top = boxplot. Bottom = companion-metric
        # scatter on its own y-axis. Sharex keeps the project labels
        # aligned across both. height_ratios=[3, 1] keeps the boxplot
        # dominant.
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(fig_width, 7),
            sharex=True, gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(
            figsize=(fig_width, 6), constrained_layout=True,
        )
        ax2 = None

    _draw_boxes(
        ax, plot_df, value_col, x_group, color_group,
        x_cats, c_cats, c_colors, rng, outlier_ids, id_col,
        xtick_prefix=xtick_prefix,
    )
    # wrap=True is the default; matplotlib's ax.set_title(wrap=True) wraps
    # to figure width; y-axis labels need textwrap.fill (set_ylabel's
    # wrap kwarg wraps at the axis boundary which is too narrow for
    # vertical labels).
    if wrap:
        ax.set_ylabel(textwrap.fill(effective_y, width=60, break_long_words=False))
        ax.set_title(effective_title, wrap=True)
    else:
        ax.set_ylabel(effective_y)
        ax.set_title(effective_title)
    ax.legend(
        title=color_group, bbox_to_anchor=(1.01, 1), loc="upper left",
        borderaxespad=0, fontsize=9,
    )

    if ax2 is not None:
        # Companion-metric scatter beneath. Hide redundant tick labels on
        # the boxplot (they show on the scatter row).
        plt.setp(ax.get_xticklabels(), visible=False)
        for x_cat in x_cats:
            grp = secondary_axis_data[
                secondary_axis_data[x_group].astype(str) == str(x_cat)
            ]
            vals = pd.to_numeric(
                grp.get(secondary_axis_value_col, pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            if len(vals) > 0:
                ax2.scatter(
                    [x_cats.index(x_cat)] * len(vals), vals.values,
                    color="grey", marker="^", s=18, alpha=0.6, zorder=2,
                )
        ax2.set_xticks(range(len(x_cats)))
        ax2.set_xticklabels(
            [f"{xtick_prefix}{c}" for c in x_cats], rotation=30, ha="right",
        )
        if secondary_axis_label:
            ax2.set_ylabel(secondary_axis_label, fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, axis="y", alpha=0.3)

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_household_range_strips(
    df: pd.DataFrame,
    *,
    project_col: str = "ProjectIdBSV",
    id_col: str = "HuisIdBSV",
    color_group: Optional[str] = "Dataleverancier",
    column_name: Optional[str] = None,
    overrides: Optional[dict] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    save_fig_path: Optional[str] = None,
    xtick_prefix: str = "Proj ",
) -> Optional[plt.Figure]:
    """
    Dense per-household range strips, grouped by project.

    Each row describes one household's stats for a single variable;
    each household becomes one vertical 1-pixel-wide strip on the
    x-axis. Within each strip three nested intervals overlay:

      - lightest layer: ``min`` to ``max`` (full observed range)
      - mid layer:      ``p01`` to ``p99``
                        (true tail; ignores single-record extremes)
      - darkest layer:  ``p25`` to ``p75`` (IQR)
      - bright dot at:  ``median``

    The intervals are nested so when overlaid the outer band is
    visible only outside the inner one. Households are coloured by
    ``color_group``; the three layers reuse the same hue at
    increasing opacity (0.25 / 0.55 / 0.95) so a denser plot
    remains legible without changing line thickness.

    Implementation note: rendering is vectorised via
    ``matplotlib.collections.LineCollection`` -- one collection
    per band regardless of household count -- so 100s of households
    render in well under a second. Households are ordered
    project-first; project tick labels sit at the centre of each
    project group.

    Missing percentile columns are tolerated; the corresponding
    band is skipped silently.
    """
    from matplotlib.collections import LineCollection

    required = {id_col, project_col, "min", "max", "median"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"plot_household_range_strips: missing columns {missing}")

    plot_df = df.dropna(subset=["min", "max", "median"]).copy()
    # Strict empty-data guard -- no silent empty plots.
    if plot_df.empty:
        raise ValueError(
            "plot_household_range_strips: no rows with non-NA "
            "min/max/median. Check coverage upstream and skip the call."
        )

    has_q1  = "p01"  in plot_df.columns
    has_q25 = "p25" in plot_df.columns
    has_q75 = "p75" in plot_df.columns
    has_q99 = "p99" in plot_df.columns

    # Stable order: project first, then household id within project.
    plot_df = plot_df.sort_values([project_col, id_col], kind="stable").reset_index(drop=True)
    n = len(plot_df)
    xs = np.arange(n, dtype=float)

    # Project tick centres: average x within each project block.
    proj_series = plot_df[project_col].astype(str)
    project_centres: dict = {}
    for proj in proj_series.unique():
        positions = xs[(proj_series == proj).values]
        if positions.size:
            project_centres[proj] = float(positions.mean())

    # Per-row colour from color_group (or steel-blue fallback).
    cmap = matplotlib.colormaps.get_cmap("tab10")
    if color_group and color_group in plot_df.columns:
        c_cats = sorted(plot_df[color_group].dropna().astype(str).unique().tolist())
        color_idx = {c: cmap(i % 10) for i, c in enumerate(c_cats)}
        rgba_per_row = [
            color_idx.get(str(c), (0.27, 0.51, 0.71, 1.0))
            for c in plot_df[color_group]
        ]
    else:
        c_cats = []
        rgba_per_row = [(0.27, 0.51, 0.71, 1.0)] * n

    def _with_alpha(rgba, a):
        return (rgba[0], rgba[1], rgba[2], a)

    mins = pd.to_numeric(plot_df["min"], errors="coerce").to_numpy(dtype=float)
    maxs = pd.to_numeric(plot_df["max"], errors="coerce").to_numpy(dtype=float)
    medians = pd.to_numeric(plot_df["median"], errors="coerce").to_numpy(dtype=float)

    fig_width = max(8.0, n * 0.10 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    # Outer band (min..max) -- lowest opacity.
    segs_outer = [
        [(xs[i], mins[i]), (xs[i], maxs[i])] for i in range(n)
        if not (np.isnan(mins[i]) or np.isnan(maxs[i]))
    ]
    if segs_outer:
        outer_colors = [
            _with_alpha(rgba_per_row[i], 0.25)
            for i in range(n)
            if not (np.isnan(mins[i]) or np.isnan(maxs[i]))
        ]
        ax.add_collection(LineCollection(
            segs_outer, colors=outer_colors, linewidths=1.0, capstyle="butt",
            zorder=2,
        ))

    # p25..p99 band -- medium opacity.
    if has_q1 and has_q99:
        p01s = pd.to_numeric(plot_df["p01"], errors="coerce").to_numpy(dtype=float)
        p99s = pd.to_numeric(plot_df["p99"], errors="coerce").to_numpy(dtype=float)
        segs_q99, q99_colors = [], []
        for i in range(n):
            if not (np.isnan(p01s[i]) or np.isnan(p99s[i])):
                segs_q99.append([(xs[i], p01s[i]), (xs[i], p99s[i])])
                q99_colors.append(_with_alpha(rgba_per_row[i], 0.55))
        if segs_q99:
            ax.add_collection(LineCollection(
                segs_q99, colors=q99_colors, linewidths=1.0, capstyle="butt",
                zorder=3,
            ))

    # q25..q75 band (IQR) -- highest opacity.
    if has_q25 and has_q75:
        p25s = pd.to_numeric(plot_df["p25"], errors="coerce").to_numpy(dtype=float)
        p75s = pd.to_numeric(plot_df["p75"], errors="coerce").to_numpy(dtype=float)
        segs_iqr, iqr_colors = [], []
        for i in range(n):
            if not (np.isnan(p25s[i]) or np.isnan(p75s[i])):
                segs_iqr.append([(xs[i], p25s[i]), (xs[i], p75s[i])])
                iqr_colors.append(_with_alpha(rgba_per_row[i], 0.95))
        if segs_iqr:
            ax.add_collection(LineCollection(
                segs_iqr, colors=iqr_colors, linewidths=1.0, capstyle="butt",
                zorder=4,
            ))

    # Median markers -- single vectorised scatter.
    valid_med = ~np.isnan(medians)
    if valid_med.any():
        ax.scatter(
            xs[valid_med], medians[valid_med],
            c=[rgba_per_row[i] for i in range(n) if valid_med[i]],
            s=10, marker="o", zorder=5,
            edgecolors="black", linewidths=0.25,
        )

    # Project ticks at group centres.
    if project_centres:
        ax.set_xticks(list(project_centres.values()))
        ax.set_xticklabels(
            [f"{xtick_prefix}{p}" for p in project_centres.keys()],
            rotation=30, ha="right",
        )
    ax.set_xlim(-0.5, n - 0.5)
    # LineCollection does not auto-trigger y-autoscaling; do it explicitly.
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    ax.grid(True, axis="y", alpha=0.3)

    base_y = y_label or "value per household"
    if column_name:
        # Strict: raises if column_name is in neither the data model nor
        # `overrides`. No silent fallback, no "(" guard -- callers must
        # not pre-stamp a unit when passing column_name (helper appends).
        base_y = axis_label_with_unit(column_name, base_y, overrides=overrides)
    ax.set_ylabel(base_y)

    base_title = title or (
        f"{column_name} per household"
        if column_name else "Per-household range strips"
    )
    if column_name:
        base_title = axis_label_with_unit(column_name, base_title, overrides=overrides)
    ax.set_title(base_title, wrap=True)

    # Build a legend with one entry per color_group category
    if c_cats:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=color_idx[c], lw=4) for c in c_cats
        ]
        ax.legend(
            handles, c_cats, title=color_group,
            bbox_to_anchor=(1.01, 1), loc="upper left",
            borderaxespad=0, fontsize=9,
        )

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=150)
        plt.close(fig)
        return None
    return fig


def plot_household_range_heatmap(
    df: pd.DataFrame,
    *,
    project_col: str = "ProjectIdBSV",
    id_col: str = "HuisIdBSV",
    column_name: Optional[str] = None,
    overrides: Optional[dict] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    save_fig_path: Optional[str] = None,
    xtick_prefix: str = "Proj ",
    n_y_bins: int = 300,
    exclude_last_percentile: bool = False,
    show_household_ids: bool = False,
    households_to_label: Optional[object] = None,
    household_id_fontsize: int = 7,
) -> Optional[plt.Figure]:
    """
    Discrete-band per-household range strips: side-by-side columns,
    no gaps, percentile bands rendered as solid blocks of colour.

    Each household becomes one column on the x-axis. The y-axis is
    sliced into bins; each cell sits in exactly one of three
    discrete bands, with a flat colour from a friendly palette:

      - **last percentile** (``#cda4a4``): the outermost 1% on
        each side -- ``min..p25`` and ``p99..max``. Muted dusty
        rose so a wide last-percentile band signals the household
        has extreme outliers without visually drowning the rest.
      - **tail** (``#f4a261``): ``p25..q25`` and ``q75..p99``.
        Warm peach -- meaningful but not loud.
      - **IQR** (``#2a6f97``): ``q25..q75``. Solid deep teal-blue
        for the interquartile range where the household spends
        most of its time.

    The median sits as a high-contrast tick (white core with a
    black outline via path_effects) so it stays visible against
    every band colour. Outside ``min..max`` cells are transparent.

    Parameters
    ----------
    exclude_last_percentile : bool, optional
        When True, clip the y-axis to ``p25..p99`` (the inner 98%)
        and skip drawing the last-percentile band entirely. Useful
        when the extreme outliers stretch the y-axis so far that
        the IQR is squashed to a few pixels. Default False.
    n_y_bins : int, optional
        Vertical resolution. Default 300.
    show_household_ids : bool, optional
        When True, draw each household's ``id_col`` value (e.g.
        ``HuisIdBSV``) horizontally above its bar, with a thin leader
        line connecting the label to the bar's max. Labels are
        staggered in two alternating rows so consecutive labels do
        not overlap horizontally. Use ``households_to_label`` to
        restrict labelling to a specific subset (e.g. outliers).
        Default False (off so other callers don't pay the layout
        adjustment).
    households_to_label : iterable, optional
        When provided alongside ``show_household_ids=True``, only
        households whose ``id_col`` value is in this iterable are
        labelled. Other households are drawn unlabelled. Useful when
        the caller has identified a subset of interest (outliers on
        some statistic, sensors flagged for review, etc.) and wants
        to call them out without cluttering the plot with all
        ~400 household ids. Default None (label every household,
        which is only practical for small cohorts).
    household_id_fontsize : int, optional
        Point size for the household-id labels when
        ``show_household_ids=True``. Default 7 -- horizontal labels
        at this size are legible without zooming in DOCX exports.

    Required columns: ``id_col``, ``project_col``, ``min``, ``max``,
    ``median``, ``p01``, ``p25``, ``p75``,
    ``p99``.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.patheffects as path_effects

    required = {
        id_col, project_col,
        "min", "max", "median",
        "p01", "p25", "p75", "p99",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"plot_household_range_heatmap: missing columns {missing}"
        )

    plot_df = df.dropna(subset=[
        "min", "max", "median",
        "p01", "p25", "p75", "p99",
    ]).copy()
    # Strict empty-data guard -- no silent empty heatmaps.
    if plot_df.empty:
        raise ValueError(
            "plot_household_range_heatmap: no rows with non-NA "
            "min/max/median/p01/p25/p75/p99. Check coverage upstream "
            "and skip the call."
        )

    plot_df = plot_df.sort_values([project_col, id_col], kind="stable").reset_index(drop=True)
    n = len(plot_df)

    mins  = plot_df["min"].astype(float).to_numpy()
    p01s   = plot_df["p01"].astype(float).to_numpy()
    p25s  = plot_df["p25"].astype(float).to_numpy()
    meds  = plot_df["median"].astype(float).to_numpy()
    p75s  = plot_df["p75"].astype(float).to_numpy()
    p99s  = plot_df["p99"].astype(float).to_numpy()
    maxs  = plot_df["max"].astype(float).to_numpy()

    if exclude_last_percentile:
        y_lo = float(np.nanmin(p01s))
        y_hi = float(np.nanmax(p99s))
    else:
        y_lo = float(np.nanmin(mins))
        y_hi = float(np.nanmax(maxs))
    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
        return None
    y_edges = np.linspace(y_lo, y_hi, n_y_bins + 1)
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0

    grid = y_centres[:, None]  # (n_y_bins, 1)
    M = mins[None, :]
    P01 = p01s[None, :]
    P25 = p25s[None, :]
    P75 = p75s[None, :]
    P99 = p99s[None, :]
    MAX = maxs[None, :]

    # Discrete band labels:
    #   1 = last percentile  (min..p25  OR  p99..max)  -- only when shown
    #   2 = tail             (p25..q25  OR  q75..p99)
    #   3 = IQR              (q25..q75)
    # 0 / NaN = outside the displayed band range -> transparent
    bands = np.zeros((n_y_bins, n), dtype=float)
    if not exclude_last_percentile:
        bands[(grid >= M)   & (grid <= P01)]  = 1
        bands[(grid >  P99) & (grid <= MAX)] = 1
    bands[(grid >  P01)  & (grid <= P25)] = 2
    bands[(grid >  P75) & (grid <= P99)] = 2
    bands[(grid >  P25) & (grid <= P75)] = 3
    bands[bands == 0] = np.nan

    # Friendly palette: muted last-percentile, warm tail, solid IQR.
    last_color = "#cda4a4"   # dusty rose -- soft, not loud
    tail_color = "#f4a261"   # warm peach
    iqr_color  = "#2a6f97"   # deep teal-blue
    cmap = ListedColormap([last_color, tail_color, iqr_color])
    cmap.set_bad(color="white", alpha=0.0)
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)

    # Compact width regardless of show_household_ids -- labels are
    # placed inside the plot with vertical staggering, so they do not
    # require extra horizontal real estate.
    fig_width = max(3.0, n * 0.010 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)

    x_edges = np.arange(n + 1) - 0.5
    masked = np.ma.masked_invalid(bands)
    ax.pcolormesh(
        x_edges, y_edges, masked,
        cmap=cmap, norm=norm,
        shading="flat", rasterized=True,
    )

    # Median tick: white core with black outline so it stays visible
    # against every band colour (rose, peach, and teal).
    valid_med = ~np.isnan(meds)
    if valid_med.any():
        # Clip to displayed y-range when last percentile is excluded
        # so the median marker doesn't render outside the panel.
        med_y = meds.copy()
        if exclude_last_percentile:
            med_y[(med_y < y_lo) | (med_y > y_hi)] = np.nan
            valid_med = ~np.isnan(med_y)
        if valid_med.any():
            sc = ax.scatter(
                np.arange(n)[valid_med], med_y[valid_med],
                c="white", s=14, marker="_", linewidths=1.4,
                zorder=5,
            )
            sc.set_path_effects([
                path_effects.Stroke(linewidth=2.6, foreground="black"),
                path_effects.Normal(),
            ])

    proj_series = plot_df[project_col].astype(str)
    project_centres = {}
    for proj in proj_series.unique():
        positions = np.arange(n)[(proj_series == proj).values]
        if positions.size:
            project_centres[proj] = float(positions.mean())
    if project_centres:
        ax.set_xticks(list(project_centres.values()))
        ax.set_xticklabels(
            [f"{xtick_prefix}{p}" for p in project_centres.keys()],
            rotation=30, ha="right", fontsize=8,
        )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis="y", labelsize=8)

    # Per-household identifier labels: horizontal, staggered above each
    # household's own bar. Anchored at the household's max value with a
    # thin leader line connecting label to bar so the association stays
    # clear when labels are sparse. Two alternating row heights so
    # consecutive labels (or pairs of labels close in x) do not overlap.
    # When `households_to_label` is given, only those households are
    # labelled -- which is the practical mode for cohorts of hundreds.
    _ids_with_labels: list[int] = []
    if show_household_ids:
        ids = plot_df[id_col].astype(str).tolist()
        if households_to_label is not None:
            # Normalise to string set for membership test; ids in the
            # iterable may be int or str.
            target = {str(x) for x in households_to_label}
            _ids_with_labels = [i for i in range(n) if ids[i] in target]
        else:
            _ids_with_labels = list(range(n))

    if _ids_with_labels:
        y_range_data = y_hi - y_lo
        base_offset = y_range_data * 0.03
        stagger_offset = y_range_data * 0.10
        for k, i in enumerate(_ids_with_labels):
            y_anchor = maxs[i] if not np.isnan(maxs[i]) else y_hi
            # Stagger by labelled-set position k (not absolute i) so
            # consecutive *labelled* households alternate even when
            # the labelled set is sparse and irregular.
            extra = stagger_offset if k % 2 == 1 else 0.0
            y_pos = y_anchor + base_offset + extra
            # Thin leader line from bar top to label base so the
            # mapping bar -> label is unambiguous (especially for the
            # higher-staggered row).
            ax.plot(
                [i, i], [y_anchor, y_pos],
                color="black", linewidth=0.4, alpha=0.5, zorder=9,
            )
            ax.text(
                i, y_pos, ids[i],
                rotation=0, fontsize=household_id_fontsize,
                ha="center", va="bottom",
                zorder=10,
            )
        # Pad top and bottom of the y-axis so the labels are not clipped
        # by the title (top) or by the project xticks at the bottom.
        # Only applied when labels are actually being drawn -- the data
        # band [y_lo, y_hi] is preserved as the area showing the bars,
        # but the axis itself extends a bit further.
        ax.set_ylim(y_lo - y_range_data * 0.05, y_hi + y_range_data * 0.20)

    base_y = y_label or "value per household"
    if column_name:
        # Strict: raises if column_name is in neither the data model nor
        # `overrides`. No "(" guard -- callers must not pre-stamp.
        base_y = axis_label_with_unit(column_name, base_y, overrides=overrides)
    ax.set_ylabel(base_y, fontsize=9)

    base_title = title or (
        f"{column_name} per household"
        if column_name else "Per-household range bands"
    )
    if column_name:
        base_title = axis_label_with_unit(column_name, base_title, overrides=overrides)
    if exclude_last_percentile:
        base_title = f"{base_title}  (p25..p99 only)"
    ax.set_title(base_title, fontsize=10, wrap=True)

    # Discrete colourbar -- precise labels.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if exclude_last_percentile:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01, ticks=[2, 3])
        cbar.set_ticklabels(["tail", "IQR"])
    else:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01, ticks=[1, 2, 3])
        cbar.set_ticklabels(["last percentile", "tail", "IQR"])
    cbar.ax.tick_params(labelsize=8)

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=150)
        plt.close(fig)
        return None
    return fig


def plot_load_duration_curve(
    df: pd.DataFrame,
    diff_column: str,
    interval: str,
    project_id: Optional[int|str]=None,
    save_fig_path: Optional[str]=None,
    filter_upper_lower_bounds: bool=True,
    outlier_marks: Optional[list]=None,
) -> None:
    """
    Plot the load duration curve for a given variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    diff_column : str
        Column representing the difference to plot.
    interval : str
        Time interval for aggregation (e.g., "5min", "hourly").
    project_id : optional, int|str
        Project identifier to filter data to plot.
    save_fig_path : str, optional
        File path to save the plot. If provided, the plot will be saved to this path.
        If not provided, the function returns the figure object.
    filter_upper_lower_bounds : bool, optional
        If True, pre-filter via analysis_helpers.filter_between_upper_lower_bounds.
        Default True.
    outlier_marks : list of int, optional
        X positions (1-based, in the sorted-curve coordinate space) at which
        to draw red dashed vertical reference lines. Use this to highlight
        positions where outliers were filtered out. Default None.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided. Otherwise, saves the plot
        to the specified path and returns `None`.
    """
    # Filter and sort
    if filter_upper_lower_bounds:
        filtered_df = analysis_helpers.filter_between_upper_lower_bounds(
            df,
            diff_column,
            project_id)
    else:
        filtered_df = df

    # Strict empty-data guard -- the sort is unhelpful on zero rows.
    if filtered_df[diff_column].notna().sum() == 0:
        raise ValueError(
            f"plot_load_duration_curve: no non-NA rows for diff_column="
            f"{diff_column!r}"
            f"{f' (project_id={project_id!r})' if project_id is not None else ''}"
            f"{' after upper/lower filter' if filter_upper_lower_bounds else ''}. "
            f"Check coverage upstream and skip the call."
        )

    sorted_values = filtered_df[diff_column].sort_values(ascending=False).reset_index(drop=True)

    # Convert per-row kWh into average power (kW) via the validated helper.
    # The helper checks that diff_column's data-model unit is kWh and
    # returns the display unit so the y-axis label stays in sync.
    from etdanalyze.units import convert_diff_to_power
    y_data, converted_unit = convert_diff_to_power(sorted_values, diff_column, interval)
    x_data = range(1, len(sorted_values) + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

    # Determine label for line
    if project_id is not None:
        label = f"Project {project_id}"
    else:
        label = f"{diff_column}"
    ax.plot(x_data, y_data, marker="none", label=label)

    if outlier_marks:
        for x_pos in outlier_marks:
            ax.axvline(x=x_pos, color="r", linestyle="--", alpha=0.6, linewidth=0.8)

    ax.set_title(f"Load Duration Curve ({interval}): {diff_column.replace('Diff', 'Netto')}")
    ax.set_xlabel("Time (1 year)")
    ax.set_ylabel(axis_label_with_unit(
        diff_column, "Power", overrides={diff_column: converted_unit},
    ))
    ax.legend()
    ax.grid(True)

    # Some statistics
    threshold = 1.5
    percentage_over = (y_data > threshold).mean() * 100
    print(f"{diff_column} stats for {interval} - Project {project_id}: {percentage_over:.2f}% above {threshold} kW")

    # Save or return figure
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300)
        print(f"Saved plot to {save_fig_path}")
        plt.close(fig)
    else:
        return fig

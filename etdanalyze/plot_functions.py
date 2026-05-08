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

    Handles Diff variants (which inherit unit from parent) because
    thresholds.csv lists them explicitly.
    """
    return _UNIT_MAP.get(col, "")


def axis_label_with_unit(col: str, base_label: Optional[str] = None) -> str:
    """
    Build a unit-suffixed axis label for a column.

    Examples
    --------
    >>> axis_label_with_unit("ElektriciteitNetgebruikHoog", "median value")
    'median value (kWh)'
    >>> axis_label_with_unit("UnknownColumn", "median value")
    'median value'
    """
    base = base_label if base_label is not None else col
    u = unit_for_column(col)
    return f"{base} ({u})" if u else base


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
        interval:str='5min',
        title:str='',
        plot_var_name: Optional[str]=None,
        project_id: Optional[str|int]=None
        ):
    """
    Plot var over time, for a specific project if supplied.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the dataset.
    var : str
        The variable to plot (e.g., 'ElektriciteitsgebruikTotaalNetto').
    interval: str
        interval of df data. choice from '5min', '15min', '60min', '6h', '24h'
    title: str
        title for the graph
    plot_var_name : str
        Label for the y-axis representing the energy variable.
    project_id : int
        The ProjectIdBSV to filter on.
    """
    if project_id is not None:
        df = df[df['ProjectIdBSV'] == project_id].copy()

    # If no specific label name is supplied, use variable name.
    if plot_var_name is None:
        plot_var_name=var

    plot_multiplier = etdtransform.calculated_columns.switch_multiplier(interval)
    df[f'{var}KW'] = df[var] * plot_multiplier
    df = df.sort_values('ReadingDate')

    fig, ax1 = plt.subplots(figsize=(16, 12))
    ax2 = ax1.twinx()

    ax1.plot(df['ReadingDate'], df[f'{var}KW'], linestyle='-', linewidth=0.75, label=f'Project {project_id}')
    ax1.set_ylabel(plot_var_name)

    ax2.plot(df['ReadingDate'], df['Temperatuur'], color='orange', linewidth=0.75, label='Temperatuur (°C)')
    ax2.set_ylabel('Temperatuur (°C)', color='orange')

    ax1.set_xlabel('Datum')

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate(rotation=45)

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_daily_profile(df, plot_var, title=None, plot_var_name=None, project_id=None, save_fig_path=None):
    """
    Plot seasonal data for a specific project, showing statistical ranges and mean values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing daily profile with at least column "ProjectIdBSV".
    plot_var : str
        The column name representing the variable to plot.
    title : str
        Optional. Title to be used in the plot. If not defined the plot_var_name or plot_var is used.
    plot_var_name : str
        A descriptive name for the plot variable (for axis labeling).
    project_id: int|str:
        Optional: when provided the df is filtered to only contain data of that project.
    save_fig_path : str, optional
        File path to save the plot. If None, the figure is returned.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided.
    """

    # add time_of_day column that sets the times correctly.
    df_normalized_dt = etdtransform.calculated_columns.add_normalized_datetime(df)

    grouped = df_normalized_dt.groupby(["ProjectIdBSV", "time_of_day"])[plot_var]
    stats = ["mean", "median", "min", "max"]
    summary_df = grouped.agg(stats).reset_index()

    # Add quartiles (25th and 75th percentiles)
    quartiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quartiles.columns = ["ProjectIdBSV", "time_of_day", "q1", "q3"]
    summary_df = summary_df.merge(quartiles, on=["ProjectIdBSV", "time_of_day"], how="left")

    # Filter for the specific project
    if project_id is not None:
        summary_df = summary_df[summary_df["ProjectIdBSV"] == project_id]

    # Set up plot colors for different statistics
    cmap = plt.colormaps.get_cmap("autumn")
    stats_to_plot = ["min", "q1", "median", "q3", "max"]
    stat_colors = [cmap(i / (len(stats_to_plot) - 1) / 2) for i in range(len(stats_to_plot))]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot statistical ranges (min, q1, median, q3, max)
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

    # When no specific var-name for the axis is defined, we use the 
    # original column name (plot_var)
    if plot_var_name is None:
        plot_var_name=plot_var

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


def plot_daily_profile_mean_combined(
        df: pd.DataFrame,
        plot_vars: list[str],
        title: str="Variabelen per 100m2 (kW)",
        plot_var_names: Optional[list[str]]=None,
        project_id: Optional[str|int]=None,
        save_fig_path: Optional[str]=None,

        ):
    """
    Plot the mean values of multiple variables over time for a specific project (if provided).

    This function visualizes the average time-of-day patterns for multiple variables within a dataset.
    If a `project_id` is provided, it filters the data to only plot that specific project.
    The plot can either be displayed or saved to a specified file path.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data with columns including "ProjectIdBSV",
        and the variables to be plotted.
    plot_vars : list of str
        List of column names in `df` representing the variables to plot.
    title : str, optional
        Title of the plot. Default is "Variabelen per 100m2 (kW)".
    plot_var_names : list of str, optional
        Custom display names for the variables in the legend. If not provided,
        the original column names from `plot_vars` will be used.
    project_id : int or str, optional
        Identifier of the project to filter the data. If `None`, all data is plotted.
    save_fig_path : str, optional
        File path to save the plot. If provided, the plot will be saved to this path.
        If not provided, the function returns the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided. Otherwise, saves the plot
        to the specified path and returns `None`.

    Notes
    -----
    - Ensure `plot_vars` contains valid columns in `df`.
    - Uses `pd.to_datetime` to parse "time_of_day" for proper time-based plotting.
    - If `plot_var_names` is not provided, original column names are used for the legend.

    Examples
    --------
    >>> plot_daily_profile_combined(
    ...     df=data,
    ...     plot_vars=["ElectricityUsage", "SolarOutput"],
    ...     title="Daily Profile",
    ...     plot_var_names=["Electricity (kW)", "Solar Power (kW)"],
    ...     project_id=101,
    ...     save_fig_path="daily_profile.png"
    ... )
    Saved plot to daily_profile.png
    """
    # add time_of_day column that sets the times correctly.
    df = etdtransform.calculated_columns.add_normalized_datetime(df)

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
    ax.set_ylabel("Variabelen per 100m2 (kW)")
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
        save_fig_path: Optional[str] = None,
):
    """
    Plot the weekly profile (weekprofiel) for a variable.

    Shows mean values for each (day-of-week, hour-of-day) combination,
    producing 168 data points across 7 day panels on a single x-axis.
    Min/q1/median/q3/max bands are shown as shading and thin lines.

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

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    df["day_of_week"] = df["ReadingDate"].dt.dayofweek   # 0=Mon, 6=Sun
    df["hour_of_day"] = df["ReadingDate"].dt.hour
    df["week_slot"] = df["day_of_week"] * 24 + df["hour_of_day"]

    grouped = df.groupby("week_slot")[plot_var]
    summary = grouped.agg(["mean", "min", "max"]).reset_index()
    summary["q1"] = grouped.quantile(0.25).values
    summary["q3"] = grouped.quantile(0.75).values
    summary["median"] = grouped.median().values

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(summary["week_slot"], summary["min"], summary["max"],
                    alpha=0.12, color="steelblue", label="min-max")
    ax.fill_between(summary["week_slot"], summary["q1"], summary["q3"],
                    alpha=0.25, color="steelblue", label="q1-q3")
    ax.plot(summary["week_slot"], summary["median"],
            linewidth=0.9, color="steelblue", alpha=0.7, label="median")
    ax.plot(summary["week_slot"], summary["mean"],
            linewidth=2, color="purple", label="mean")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in range(7):
        ax.axvline(day * 24, color="gray", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Day of week / hour")
    ax.set_ylabel(plot_var_name)
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
        save_fig_path: Optional[str] = None,
):
    """
    Plot the annual profile (jaarprofiel) for a variable.

    Shows mean and q1–q3 band for each day-of-year (1–366),
    revealing seasonal patterns across the year.

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

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id].copy()
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
    df["day_of_year"] = df["ReadingDate"].dt.dayofyear

    grouped = df.groupby("day_of_year")[plot_var]
    summary = grouped.mean().reset_index(name="mean")
    summary["q1"] = grouped.quantile(0.25).values
    summary["q3"] = grouped.quantile(0.75).values

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(summary["day_of_year"], summary["q1"], summary["q3"],
                    alpha=0.25, color="steelblue", label="q1-q3")
    ax.plot(summary["day_of_year"], summary["mean"],
            linewidth=1.5, color="purple", label="mean")

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for start, name in zip(month_starts, month_names):
        ax.axvline(start, color="gray", linewidth=0.4, linestyle="--")
        ax.text(start + 0.5, ax.get_ylim()[1], name, fontsize=7, color="gray", va="top")

    ax.set_xlabel("Day of year")
    ax.set_ylabel(plot_var_name)
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


def plot_cop_vs_temp(
        df: pd.DataFrame,
        cop_col: str = "COP",
        temp_col: str = "TemperatuurBuitenWarmtepomp",
        title: Optional[str] = None,
        project_id: Optional[int | str] = None,
        save_fig_path: Optional[str] = None,
):
    """
    Scatter plot of daily average COP against outdoor temperature.

    Each point represents one day. Requires daily-aggregated input
    (compute daily means before calling).

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

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df[temp_col], plot_df[cop_col],
               alpha=0.4, s=20, color="steelblue")
    ax.set_xlabel(f"Outdoor temperature (°C) [{temp_col}]")
    ax.set_ylabel(f"COP [{cop_col}]")
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

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df["delta_T"], plot_df[cop_col],
               alpha=0.4, s=20, color="steelblue")
    ax.set_xlabel(f"Temperature lift dT (degC)  [{supply_temp_col} - {outdoor_temp_col}]")
    ax.set_ylabel(f"COP [{cop_col}]")
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
    if energy_label is None:
        energy_label = energy_col

    plot_df = df[[temp_col, energy_col]].dropna()

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
            ax.text(sg + 0.2, ax.get_ylim()[0],
                    f"Stookgrens\n{sg:.1f}degC",
                    fontsize=8, color="gray", va="bottom")

    ax.set_xlabel(f"Outdoor temperature (°C) [{temp_col}]")
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
        save_fig_path: Optional[str] = None,
):
    """
    Plot the monthly profile (maandprofiel) as a heatmap: 12 months x 24 hours = 288 slots.

    Each cell shows the mean value of plot_var for that (month, hour-of-day) combination,
    aggregated over all rows in df. The heatmap is compact and directly comparable across
    variables and domains.

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

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if plot_var_name is None:
        plot_var_name = plot_var

    df = df.copy()
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
    plt.colorbar(im, ax=ax, label=plot_var_name)

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


def plot_boxwhisker_comparison(
        df: pd.DataFrame,
        value_col: str,
        x_group: str,
        color_group: str,
        *,
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        jitter_seed: int = 42,
        save_fig_path: Optional[str] = None,
        # Optional extensions: unit-aware labels, outlier overlay,
        # and a stacked secondary axis. All default-None / default-empty
        # so existing callers see no behaviour change.
        column_name: Optional[str] = None,
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
    title, y_label, jitter_seed, save_fig_path :
        As before.
    column_name : str, optional
        ETD model column name. When given, the y_label and title are
        suffixed with the variable's unit looked up from etdmap
        thresholds (e.g. ' (kWh)'). Skipped silently if no unit is
        registered.
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
    if plot_df.empty:
        return None

    x_cats = sorted(plot_df[x_group].dropna().unique().tolist(), key=str)
    c_cats = sorted(plot_df[color_group].dropna().unique().tolist(), key=str)

    cmap = matplotlib.colormaps.get_cmap("tab10")
    c_colors = {cat: cmap(i % 10) for i, cat in enumerate(c_cats)}
    rng = np.random.default_rng(jitter_seed)

    # Resolve labels with optional unit lookup
    effective_y = y_label or value_col
    effective_title = title or f"{value_col} by {x_group}"
    if column_name:
        u = unit_for_column(column_name)
        if u:
            effective_y = f"{effective_y} ({u})"
            effective_title = f"{effective_title} ({u})"

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
    ax.set_ylabel(effective_y)
    ax.legend(
        title=color_group, bbox_to_anchor=(1.01, 1), loc="upper left",
        borderaxespad=0, fontsize=9,
    )
    ax.set_title(effective_title)

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
    multiplier = etdtransform.calculated_columns.switch_multiplier(interval)

    if filter_upper_lower_bounds:
        filtered_df = analysis_helpers.filter_between_upper_lower_bounds(
            df,
            diff_column,
            project_id)
    else:
        filtered_df = df

    sorted_values = filtered_df[diff_column].sort_values(ascending=False).reset_index(drop=True)

    x_data = range(1, len(sorted_values) + 1)
    y_data = sorted_values * multiplier

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
    ax.set_ylabel(axis_label_with_unit(diff_column, "Power"))
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

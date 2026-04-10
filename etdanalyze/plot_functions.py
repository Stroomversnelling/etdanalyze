from typing import Optional

import etdtransform
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from . import analysis_helpers

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
                    alpha=0.12, color="steelblue", label="min–max")
    ax.fill_between(summary["week_slot"], summary["q1"], summary["q3"],
                    alpha=0.25, color="steelblue", label="q1–q3")
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
                    alpha=0.25, color="steelblue", label="q1–q3")
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
    ax.set_xlabel(f"Temperature lift ΔT (°C)  [{supply_temp_col} − {outdoor_temp_col}]")
    ax.set_ylabel(f"COP [{cop_col}]")
    plt.title(title or "COP vs temperature lift (ΔT)")
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
                    f"Stookgrens\n{sg:.1f}°C",
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


def plot_load_duration_curve(
    df: pd.DataFrame,
    diff_column: str,
    interval: str,
    project_id: Optional[int|str]=None,
    save_fig_path: Optional[str]=None,
    filter_upper_lower_bounds: bool=True
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
    fig, ax = plt.subplots(figsize=(16, 12))

    # Determine label for line
    if project_id is not None:
        label = f"Project {project_id}"
    else:
        label = f"{diff_column}"
    ax.plot(x_data, y_data, marker="none", label=label)

    plt.title(f"Load Duration Curve ({interval}): {diff_column.replace('Diff', 'Netto')}")
    ax.set_xlabel("Time (1 year)")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()

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

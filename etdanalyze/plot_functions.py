from typing import Optional

import etdtransform
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

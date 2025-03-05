from typing import Optional

import etdtransform
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_profile(df, plot_var, title=None, plot_var_name=None, project_id=None, save_fig_path=None):
    """
    Plot seasonal data for a specific project, showing statistical ranges and mean values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing daily profile with at least column "ProjectIdBSV".
    title : str
        Title to be used in the plot.
    plot_var : str
        The column name representing the variable to plot.
    plot_var_name : str
        A descriptive name for the plot variable (for axis labeling).
    plot_interval : str
        The time interval (e.g., "15min", "1hour").
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


def plot_daily_profile_combined(
        df: pd.DataFrame,
        plot_vars: list[str],
        title: str="Variabelen per 100m2 (kW)",
        plot_var_names: Optional[list[str]]=None,
        project_id: Optional[str|int]=None,
        save_fig_path: Optional[str]=None
        ):
    """
    Plot the mean values of multiple variables for a specific project over time.

    This function visualizes the average time-of-day patterns for a given project's variables
    during a specified season. Multiple variables are plotted together on the same graph
    with distinct colors. The plot can be either displayed or saved to a file.

    Parameters
    ----------
    season_df : pandas.DataFrame
        DataFrame containing the seasonal data with columns including "ProjectIdBSV",
        "time_of_day", and the variables to be plotted.
    season_name : str
        Name of the season (e.g., "Summer" or "Winter") to display in the plot title.
    project_id : int or str
        Identifier of the project for which the variables will be plotted.
    plot_vars : list of str
        List of column names in `season_df` representing the variables to plot.
    plot_interval : str, optional
        Time interval for the plot (e.g., "5min", "15min"). Default is "5min".
    plot_var_names : list of str, optional
        Custom display names for the variables in the legend. If not provided,
        the original column names from `plot_vars` will be used.
    save_fig_path : str, optional
        File path to save the plot. If provided, the plot will be saved and not displayed.
        If not provided, the figure object will be returned.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided. Otherwise, saves the plot
        to the specified path and returns `None`.

    Notes
    -----
    - Ensure `plot_vars` contains valid columns in `season_df`.
    - Uses `pd.to_datetime` to parse "time_of_day" for proper time-based plotting.
    - Distinct colors are assigned to each variable for better visualization.

    Examples
    --------
    >>> plot_average_graphs_together(
    ...     season_df=data,
    ...     season_name="Summer",
    ...     project_id=101,
    ...     plot_vars=["ElectricityUsage", "SolarOutput"],
    ...     plot_var_names=["Electricity (kW)", "Solar Power (kW)"],
    ...     save_fig_path="summer_project_101.png"
    ... )
    Saved plot to summer_project_101.png
    """
    # add time_of_day column that sets the times correctly.
    df = etdtransform.calculated_columns.add_normalized_datetime(df)

    fig, ax = plt.subplots(figsize=(16, 12))
    colors = [plt.cm.viridis_r(i / (len(plot_vars) - 1)) for i in range(len(plot_vars))]  # Assign distinct colors

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

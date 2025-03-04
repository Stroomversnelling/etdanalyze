import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_season_for_project(season_table, season_name, project_id, plot_var, plot_var_name, plot_interval, save_fig_path=None):
    """
    Plot seasonal data for a specific project, showing statistical ranges and mean values.

    Parameters
    ----------
    season_table : ibis.expr.types.Table
        Ibis table containing seasonal data with "ProjectIdBSV" and "time_of_day".
    season_name : str
        Name of the season (e.g., "Summer", "Winter").
    project_id : int or str
        The ProjectIdBSV to plot.
    plot_var : str
        The column name representing the variable to plot.
    plot_var_name : str
        A descriptive name for the plot variable (for axis labeling).
    plot_interval : str
        The time interval (e.g., "15min", "1hour").
    save_fig_path : str, optional
        File path to save the plot. If None, the figure is returned.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if `save_fig_path` is not provided.
    """
    print(f"Plotting {season_name} - Project {project_id} - {plot_var_name} - {plot_interval}")

    # Convert to DataFrame and sort by project and time
    season_df = season_table.order_by("ProjectIdBSV", "time_of_day").execute()

    # Calculate summary statistics for each project and time
    grouped = season_df.groupby(["ProjectIdBSV", "time_of_day"])[plot_var]
    stats = ["mean", "median", "min", "max"]
    summary_df = grouped.agg(stats).reset_index()

    # Add quartiles (25th and 75th percentiles)
    quartiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quartiles.columns = ["ProjectIdBSV", "time_of_day", "q1", "q3"]
    summary_df = summary_df.merge(quartiles, on=["ProjectIdBSV", "time_of_day"], how="left")

    # Filter for the specific project
    project_df = summary_df[summary_df["ProjectIdBSV"] == project_id]

    # Set up plot colors for different statistics
    cmap = plt.colormaps.get_cmap("autumn")
    stats_to_plot = ["min", "q1", "median", "q3", "max"]
    stat_colors = [cmap(i / (len(stats) - 1) / 2) for i in range(len(stats_to_plot))]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot statistical ranges (min, q1, median, q3, max)
    for i, stat in enumerate(stats_to_plot):
        ax.plot(
            pd.to_datetime(project_df["time_of_day"], format="%H:%M:%S"),
            project_df[stat],
            linestyle="-",
            linewidth=0.8,
            color=stat_colors[i],
            alpha=0.5,
            label=stat,
        )

    # Highlight the mean value
    ax.plot(
        pd.to_datetime(project_df["time_of_day"], format="%H:%M:%S"),
        project_df["mean"],
        linestyle="-",
        linewidth=2,
        color="purple",
        label="mean",
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
    plt.title(f"{season_name} - Project {project_id} - {plot_var_name} - {plot_interval}")
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

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_season(season_table, season_name, project, plot_var):

    season_df = season_table.order_by("ProjectIdBSV", "time_of_day").execute()

    # Add lighter lines for each project stats
    grouped = season_df.groupby(["ProjectIdBSV", "time_of_day"])[f"{plot_var}"]
    thin_plot_df = grouped.agg(["mean", "median", "min", "max"]).reset_index()
    quartiles = grouped.quantile([0.25, 0.75]).unstack(level=-1).reset_index()
    quartiles.columns = ["ProjectIdBSV", "time_of_day", "q1", "q3"]
    thin_plot_df = thin_plot_df.merge(
        quartiles, on=["ProjectIdBSV", "time_of_day"], how="left"
    )

    stats = ["min", "q1", "median", "q3", "max"]

    colors = matplotlib.colormaps.get_cmap("autumn")
    selected_colors = [colors(i / (len(stats) - 1) / 2) for i in range(len(stats))]

    plot_df = thin_plot_df[["ProjectIdBSV", "time_of_day", "mean"]]

    thin_plot_df = thin_plot_df[["ProjectIdBSV", "time_of_day", *stats]]

    for project in projects:
        print(
            f"Plotting {season_name} - project {project} - {plot_var_name} - {plot_interval}"
        )

        fig, ax1 = plt.subplots(figsize=(16, 12))
        # ax2 = ax1.twinx() #MJW: I think this is where it creates a second y axis, right?

        thin_project_df = thin_plot_df[thin_plot_df["ProjectIdBSV"] == project]

        for i, stat in enumerate(stats):
            line_color = selected_colors[i]
            ax1.plot(
                pd.to_datetime(thin_project_df["time_of_day"], format="%H:%M:%S"),
                thin_project_df[stat],
                marker="none",
                linestyle="-",
                linewidth=0.8,
                color=line_color,
                alpha=0.5,
                label=stat,
            )

        project_df = plot_df[plot_df["ProjectIdBSV"] == project]

        ax1.plot(
            pd.to_datetime(project_df["time_of_day"], format="%H:%M:%S"),
            project_df["mean"],
            marker="none",
            linestyle="-",
            linewidth=2,
            label=f"mean",
            color="purple",
        )

        ax1.set_ylabel(f"{plot_var_name}", color="purple")
        ax1.tick_params(axis="y", labelcolor="purple")
        # Create a second y-axis for temperature

        ax1.set_xlabel("Tijd")

        # Ensure all unique dates are displayed on the x-axis
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.autoscale(enable=True, axis="x", tight=True)

        fig.autofmt_xdate(rotation=45)
        ax1.legend()

        plt.title(
            f"{season_name} - project {project} - {plot_var_name} - {plot_interval}"
        )

        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()

def plot_seasonal_profiles(
        project_table,
        original_var="TerugleveringTotaalNettoPer100M2",
        plot_var_name="Teruglevering woning per 100m2 (kW)",
    ):
    # plot_var = f"{original_var}KW"
    plot_interval = "5min"
    plot_var = f"{original_var}KW"

    summer_project_table = project_table.filter(
    (_["ReadingDate"].month() == 6)
    | (_["ReadingDate"].month() == 7)
    | (_["ReadingDate"].month() == 8)
)

    winter_project_table = project_table.filter(
        (_["ReadingDate"].month() == 12)
        | (_["ReadingDate"].month() == 1)
        | (_["ReadingDate"].month() == 2)
    )

    print(f"Plotting {plot_var_name} - {plot_interval}")

    print(f"Plotting {"Zomer"} - {plot_var_name} - {plot_interval}")
    plot_season(summer_project_table, "Zomer", projects, plot_var)
    print(f"Plotting {"Winter"} - {plot_var_name} - {plot_interval}")
    plot_season(winter_project_table, "Winter")

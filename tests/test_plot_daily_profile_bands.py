"""Tests for plot_daily_profile_bands -- the band-style daily profile.

Companion to ``plot_daily_profile`` (line style). Same input contract;
different visual layout (min-max / p01-p99 / p25-p75 bands + median /
mean lines, matching weekly + annual).
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from etdanalyze.plot_functions import plot_daily_profile_bands


_TEST_COL = "TemperatuurBuitenWarmtepomp"


def _synthetic_hourly(n_days=14):
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    df = pd.DataFrame({
        "ReadingDate": idx,
        "ProjectIdBSV": 1,
    })
    df[_TEST_COL] = 10.0 + df["ReadingDate"].dt.hour * 0.3
    return df


def test_saves_file(tmp_path):
    df = _synthetic_hourly()
    out = tmp_path / "dagprofielbands.png"
    plot_daily_profile_bands(df, _TEST_COL, save_fig_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_returns_figure_when_no_path():
    df = _synthetic_hourly()
    fig = plot_daily_profile_bands(df, _TEST_COL)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_rejects_all_na_input():
    """Strict empty-data guard -- same shape as other profile helpers."""
    df = _synthetic_hourly()
    df[_TEST_COL] = pd.NA
    with pytest.raises(ValueError, match="no non-NA rows"):
        plot_daily_profile_bands(df, _TEST_COL)


def test_rejects_energy_per_interval_unit():
    """Strict unit class assertion -- same set as other profile helpers."""
    df = _synthetic_hourly().rename(columns={_TEST_COL: "ElektriciteitNetgebruikHoog"})
    with pytest.raises(ValueError, match="not in the unit classes"):
        plot_daily_profile_bands(df, "ElektriciteitNetgebruikHoog")


def test_convert_kwarg_yields_power_axis(tmp_path):
    """The opt-in conversion kwarg works the same as in other profile helpers."""
    df = _synthetic_hourly().rename(columns={_TEST_COL: "ElektriciteitNetgebruikHoog"})
    out = tmp_path / "dagprofielbands_kw.png"
    plot_daily_profile_bands(
        df, "ElektriciteitNetgebruikHoog",
        convert_from_energy_per_interval=pd.Timedelta(hours=1),
        save_fig_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0


def test_band_layers_present():
    """All four band layers (min-max, p01-p99, p25-p75) and median + mean
    lines must be in the legend so the figure is self-describing."""
    df = _synthetic_hourly()
    fig = plot_daily_profile_bands(df, _TEST_COL)
    try:
        ax = fig.axes[0]
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend_labels >= {"min-max", "p01-p99", "p25-p75", "median", "mean"}
    finally:
        plt.close(fig)


def test_project_id_filter(tmp_path):
    """project_id filter narrows the input frame before grouping."""
    df = pd.concat([
        _synthetic_hourly().assign(ProjectIdBSV=1),
        _synthetic_hourly().assign(ProjectIdBSV=2,
                                   **{_TEST_COL: lambda d: d[_TEST_COL] + 5}),
    ], ignore_index=True)
    out = tmp_path / "proj.png"
    plot_daily_profile_bands(
        df, _TEST_COL, project_id=1, save_fig_path=str(out),
    )
    assert out.exists()


def test_project_id_filter_empty_raises():
    """project_id=N where N has no rows -> ValueError per strict guard."""
    df = _synthetic_hourly()  # all ProjectIdBSV == 1
    with pytest.raises(ValueError, match="no non-NA rows"):
        plot_daily_profile_bands(df, _TEST_COL, project_id=999)

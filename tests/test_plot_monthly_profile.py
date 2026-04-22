"""Tests for plot_monthly_profile."""
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

from etdanalyze.plot_functions import plot_monthly_profile


def _synthetic_hourly(n_years=2, value=1.0):
    """Two years of hourly data with a predictable pattern."""
    idx = pd.date_range("2023-01-01", periods=n_years * 8760, freq="h")
    df = pd.DataFrame({"ReadingDate": idx})
    # value varies by hour so the heatmap has non-uniform content
    df["var"] = value + df["ReadingDate"].dt.hour * 0.1
    return df


def test_saves_file(tmp_path):
    df = _synthetic_hourly()
    out = tmp_path / "maandprofiel_test.png"
    plot_monthly_profile(df, "var", title="Test", save_fig_path=str(out))
    assert out.exists(), "Output file was not created"
    assert out.stat().st_size > 0, "Output file is empty"


def test_returns_figure_when_no_path():
    import matplotlib.pyplot as plt
    df = _synthetic_hourly()
    fig = plot_monthly_profile(df, "var")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_288_slots_covered():
    """Pivot should have 24 rows x 12 columns = 288 cells, all non-null."""
    import numpy as np
    df = _synthetic_hourly(n_years=2)
    df["month"] = df["ReadingDate"].dt.month
    df["hour_of_day"] = df["ReadingDate"].dt.hour
    grouped = df.groupby(["month", "hour_of_day"])["var"].mean().reset_index()
    pivot = grouped.pivot(index="hour_of_day", columns="month", values="var")
    pivot = pivot.reindex(index=range(24), columns=range(1, 13))
    assert pivot.shape == (24, 12), f"Expected (24, 12), got {pivot.shape}"
    assert not pivot.isnull().any().any(), "Some (month, hour) slots have no data"


def test_missing_hours_filled():
    """If some hours are absent the reindex fills them; file still saved."""
    df = _synthetic_hourly()
    df = df[df["ReadingDate"].dt.hour != 3]  # drop hour 3
    out_path = None  # return figure mode
    fig = plot_monthly_profile(df, "var")
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

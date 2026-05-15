"""Tests for plot_monthly_profile."""
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

from etdanalyze.plot_functions import plot_monthly_profile


# Use a real data-model column whose unit is in plot_monthly_profile's
# accepted unit classes (Layer 3 strict assertion: this plot reads an
# instantaneous value at each (month, hour) cell, so energy-per-interval
# units like kWh are intentionally NOT accepted). A temperature column
# (unit "graden C", class "temperature") satisfies both the strict label
# contract (Layer 2) and the strict unit-class contract (Layer 3). The
# tests don't care about the column's meaning; they exercise plot mechanics.
_TEST_COL = "TemperatuurBuitenWarmtepomp"


def _synthetic_hourly(n_years=2, value=1.0):
    """Two years of hourly data with a predictable pattern."""
    idx = pd.date_range("2023-01-01", periods=n_years * 8760, freq="h")
    df = pd.DataFrame({"ReadingDate": idx})
    # value varies by hour so the heatmap has non-uniform content
    df[_TEST_COL] = value + df["ReadingDate"].dt.hour * 0.1
    return df


def test_saves_file(tmp_path):
    df = _synthetic_hourly()
    out = tmp_path / "maandprofiel_test.png"
    plot_monthly_profile(df, _TEST_COL, title="Test", save_fig_path=str(out))
    assert out.exists(), "Output file was not created"
    assert out.stat().st_size > 0, "Output file is empty"


def test_returns_figure_when_no_path():
    import matplotlib.pyplot as plt
    df = _synthetic_hourly()
    fig = plot_monthly_profile(df, _TEST_COL)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_288_slots_covered():
    """Pivot should have 24 rows x 12 columns = 288 cells, all non-null."""
    df = _synthetic_hourly(n_years=2)
    df["month"] = df["ReadingDate"].dt.month
    df["hour_of_day"] = df["ReadingDate"].dt.hour
    grouped = df.groupby(["month", "hour_of_day"])[_TEST_COL].mean().reset_index()
    pivot = grouped.pivot(index="hour_of_day", columns="month", values=_TEST_COL)
    pivot = pivot.reindex(index=range(24), columns=range(1, 13))
    assert pivot.shape == (24, 12), f"Expected (24, 12), got {pivot.shape}"
    assert not pivot.isnull().any().any(), "Some (month, hour) slots have no data"


def test_missing_hours_filled():
    """If some hours are absent the reindex fills them; file still saved."""
    df = _synthetic_hourly()
    df = df[df["ReadingDate"].dt.hour != 3]  # drop hour 3
    fig = plot_monthly_profile(df, _TEST_COL)
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_rejects_energy_per_interval_unit():
    """Layer 3 strict assertion: a monthly profile reads as an
    instantaneous value at each (month, hour) cell; energy-per-interval
    units (kWh / Wh / GJ) are NOT accepted -- callers wanting a power
    view of an energy column must convert upstream into a new column
    declared with a power unit. The helper raises ValueError describing
    accepted unit classes."""
    df = _synthetic_hourly()
    df = df.rename(columns={_TEST_COL: "ElektriciteitNetgebruikHoog"})
    with pytest.raises(ValueError, match="not in the unit classes"):
        plot_monthly_profile(df, "ElektriciteitNetgebruikHoog")


def test_accepts_overridden_power_unit():
    """A caller-derived kW column with overrides={col: 'kW'} passes the
    Layer 3 assertion (power is an accepted class)."""
    import matplotlib.pyplot as plt
    df = _synthetic_hourly().rename(columns={_TEST_COL: "my_derived_kW"})
    fig = plot_monthly_profile(
        df, "my_derived_kW",
        overrides={"my_derived_kW": "kW"},
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_convert_from_energy_per_interval_accepts_kwh():
    """Caller opts in to energy-per-interval -> power conversion via the
    new kwarg. The helper L3-asserts the source is energy_per_interval,
    delegates to the shared conversion in etdanalyze.units, and the
    plot succeeds with the derived power column. The input df is not
    mutated (the new derived column appears only on the internal copy
    used by the helper)."""
    import matplotlib.pyplot as plt
    df = _synthetic_hourly().rename(columns={_TEST_COL: "ElektriciteitNetgebruikHoog"})
    cols_before = list(df.columns)
    fig = plot_monthly_profile(
        df, "ElektriciteitNetgebruikHoog",
        convert_from_energy_per_interval=pd.Timedelta(hours=1),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    assert list(df.columns) == cols_before, "Input df must not be mutated"


def test_convert_from_energy_per_interval_rejects_non_energy_source():
    """The opt-in kwarg requires the source column to resolve to an
    energy-per-interval unit. Passing it for a temperature column is
    a contract violation -- the helper raises ValueError describing
    that the source must be in the energy_per_interval class."""
    df = _synthetic_hourly()  # _TEST_COL is graden C (temperature)
    with pytest.raises(ValueError, match="not in the unit classes"):
        plot_monthly_profile(
            df, _TEST_COL,
            convert_from_energy_per_interval=pd.Timedelta(hours=1),
        )


def test_rejects_all_na_input():
    """Strict empty-data contract: no silent empty plots. When the
    plot_var column is entirely NA the helper raises ValueError
    asking the caller to check coverage and skip upstream."""
    df = _synthetic_hourly()
    df[_TEST_COL] = pd.NA
    with pytest.raises(ValueError, match="no non-NA rows"):
        plot_monthly_profile(df, _TEST_COL)

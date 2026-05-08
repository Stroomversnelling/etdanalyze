"""
Unit-conversion helpers for analysis.

Centralises the math that converts between per-row energy (kWh per N-min
interval), average power (kW), per-area normalisation (W/m2 etc.), and
period totals (kWh per year). Future analyses should route through these
helpers instead of inventing multipliers in each script.

The underlying multiplier math is owned by
``etdtransform.calculated_columns.switch_multiplier``. This module
wraps that with explicit input/output unit naming so callers stop
hand-applying multipliers per script.

Supported intervals: 5min, 15min, 60min, 6h, 24h.
"""

import pandas as pd

from etdtransform.calculated_columns import switch_multiplier


_INTERVAL_HOURS = {
    "5min":  5 / 60,
    "15min": 15 / 60,
    "60min": 1.0,
    "6h":    6.0,
    "24h":   24.0,
}

_HOURS_PER_DAY = 24
_DAYS_PER_YEAR = 365


def interval_hours(interval: str) -> float:
    """
    Return hours per row for an interval string.

    Parameters
    ----------
    interval : str
        One of ``"5min"``, ``"15min"``, ``"60min"``, ``"6h"``, ``"24h"``.

    Returns
    -------
    float
        Hours per row (e.g. 5/60 for "5min", 1.0 for "60min").

    Raises
    ------
    ValueError
        If ``interval`` is not a supported value.
    """
    if interval not in _INTERVAL_HOURS:
        raise ValueError(
            f"Unknown interval {interval!r}; supported: "
            f"{sorted(_INTERVAL_HOURS)}"
        )
    return _INTERVAL_HOURS[interval]


def infer_interval_from_readingdate(
    df: pd.DataFrame, col: str = "ReadingDate"
) -> str:
    """
    Infer the interval label ("5min", "60min", ...) from row spacing.

    Looks at the modal time delta between consecutive rows of ``df[col]``
    after sorting, then matches against the supported interval set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a datetime column.
    col : str, default "ReadingDate"
        Name of the datetime column.

    Returns
    -------
    str
        Interval label ("5min", "15min", ...).

    Raises
    ------
    ValueError
        If the column is missing, has fewer than 2 rows, or the inferred
        spacing does not match a supported interval.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not in df; cannot infer interval")
    sorted_times = df[col].dropna().sort_values()
    if len(sorted_times) < 2:
        raise ValueError(
            f"Need at least 2 rows in {col} to infer interval; "
            f"got {len(sorted_times)}"
        )
    delta = sorted_times.diff().dropna().mode().iloc[0]
    minutes = delta.total_seconds() / 60.0
    for label, hours in _INTERVAL_HOURS.items():
        if abs(minutes - hours * 60) < 0.01:
            return label
    raise ValueError(
        f"Inferred {minutes:.3f} minutes between rows; not a supported "
        f"interval. Supported: {sorted(_INTERVAL_HOURS)}"
    )


def convert_diff_to_power(series: pd.Series, interval: str) -> pd.Series:
    """
    Convert per-row energy (kWh per ``interval`` row) to average power (kW).

    Thin wrapper over
    ``etdtransform.calculated_columns.switch_multiplier``. Use when a
    plot or downstream calculation wants kW (an instantaneous-rate-like
    quantity) rather than the per-row energy total.

    Parameters
    ----------
    series : pd.Series
        Energy values, one per row, in kWh-per-``interval``.
    interval : str
        Row interval label ("5min", "15min", ...).

    Returns
    -------
    pd.Series
        Average power in kW (= kWh per hour).
    """
    return series * switch_multiplier(interval)


def mean_per_row_to_period_total(
    mean_value: float,
    interval: str,
    *,
    period_days: int = _DAYS_PER_YEAR,
) -> float:
    """
    Scale a mean-per-row value to a period total.

    Use when the underlying variable is cumulative (sum-resampled in
    etdmodel.csv) and the report wants a per-household per-period figure
    (e.g. kWh per year, liters per year) instead of the raw per-row
    mean.

    Parameters
    ----------
    mean_value : float
        Mean of the per-row values (e.g. mean kWh per 5-min row).
    interval : str
        Row interval label ("5min", "15min", ...).
    period_days : int, default 365
        Number of days in the target period (365 for "per year",
        365 * 7/12 for "verwarmingsperiode" Oct-Apr ~213 days, etc.).

    Returns
    -------
    float
        Period total in the variable's natural unit (kWh, L, ...).
    """
    rows_per_period = period_days * _HOURS_PER_DAY / interval_hours(interval)
    return mean_value * rows_per_period


def to_per_m2(
    series: pd.Series,
    area: "pd.Series | float",
    *,
    scale: float = 1.0,
) -> pd.Series:
    """
    Divide a series by household floor area, with optional kW->W rescaling.

    Parameters
    ----------
    series : pd.Series
        Source values (kW, kWh, ...).
    area : pd.Series or float
        Floor area in m**2. A Series is matched element-wise to ``series``;
        a scalar applies to every row. Must be > 0 and not NA.
    scale : float, default 1.0
        Multiplier applied after division. Pass 1000 to convert kW -> W
        (so the output is W/m**2 instead of kW/m**2).

    Returns
    -------
    pd.Series
        ``series * scale / area``.

    Raises
    ------
    ValueError
        If any area value is NA or not strictly positive.
    """
    if isinstance(area, (int, float)):
        if pd.isna(area) or area <= 0:
            raise ValueError(f"Area must be > 0; got {area!r}")
        return series * scale / area
    if area.isna().any():
        raise ValueError(
            f"Area series contains {int(area.isna().sum())} NA value(s); "
            "per-household area must be set"
        )
    if (area <= 0).any():
        raise ValueError(
            f"Area series contains {int((area <= 0).sum())} non-positive "
            "value(s); per-household area must be > 0"
        )
    return series * scale / area

import re
from typing import Optional

import etdtransform
import numpy as np
import pandas as pd
from ibis import _
from ibis.expr.types import Table


def sort_load(series: pd.Series) -> pd.Series:
    """
    Sort a Series in descending order and reset the index.

    Used by load-duration-curve plots: the result is the y-axis values when
    plotted against ascending integer positions on the x-axis.

    Parameters
    ----------
    series : pd.Series
        Numeric series to sort.

    Returns
    -------
    pd.Series
        Descending-sorted values with a fresh RangeIndex.
    """
    return series.sort_values(ascending=False).reset_index(drop=True)


def _outlier_mask_tukey(series: pd.Series, k: float = 1.5, z_thresh: float = 2.5) -> pd.Series:
    """
    Boolean mask of outliers via Tukey fence (Q1 - k*IQR, Q3 + k*IQR), with
    a z-score fallback for tiny samples (n < 4) or zero IQR. Index-aligned
    with the input.
    """
    s = pd.to_numeric(series, errors="coerce")
    n = s.notna().sum()
    if n < 1:
        return pd.Series(False, index=series.index)
    if n < 4:
        # z-score fallback: too few points for IQR to be meaningful
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index)
        z = (s - s.mean()).abs() / std
        return (z > z_thresh).fillna(False)
    p25, p75 = s.quantile(0.25), s.quantile(0.75)
    iqr = p75 - p25
    if iqr == 0:
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index)
        z = (s - s.mean()).abs() / std
        return (z > z_thresh).fillna(False)
    lo, hi = p25 - k * iqr, p75 + k * iqr
    return ((s < lo) | (s > hi)).fillna(False)


def _percentile_bounds(series: pd.Series) -> tuple:
    """
    Compute (low_bound, high_bound) using the p01*10 / p99*10 rule from
    `filter_between_upper_lower_bounds`. The low bound is 0 unless the
    1st-percentile value is itself negative (in which case it is p01*10).
    Captures legitimate large tails while clipping extreme values that
    dwarf the bulk of the data.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    high_bound = float(s.quantile(0.99) * 10)
    if float(s.min()) < 0 and float(s.quantile(0.01)) < 0:
        low_bound = float(s.quantile(0.01) * 10)
    else:
        low_bound = 0.0
    return low_bound, high_bound


def build_extreme_value_table(
        df: pd.DataFrame,
        value_col: str,
        *,
        group_cols: tuple = ("Dataleverancier", "ProjectIdBSV"),
        id_col: str = "HuisIdBSV",
        top_n: int = 10,
) -> pd.DataFrame:
    """
    Long-form table of rows whose `value_col` falls outside the
    percentile-bounds (p01*10 / p99*10) of its (group_cols) cohort.

    Sibling of `build_outlier_table`. The two answer different questions:

    - `build_outlier_table` (Tukey + z-score): "is this value unusual
      relative to its cohort distribution?" Sensitive to relative
      position. Best for spotting per-project anomalies where the
      cohort has a reasonably symmetric spread.
    - `build_extreme_value_table` (this function, percentile-bounds):
      "is this value so extreme that it dwarfs the bulk of the data?"
      Robust to skewed distributions (energy variables typically are).
      Insensitive to non-normality.

    The two are not interchangeable. Pick the one that matches the
    question you are asking.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation; must include value_col, id_col, and
        every column in group_cols.
    value_col : str
        Numeric column to scan.
    group_cols : tuple of str, optional
        Cohort definition. Default ('Dataleverancier', 'ProjectIdBSV').
    id_col : str, optional
        Identifier column for each observation. Default 'HuisIdBSV'.
    top_n : int, optional
        Maximum number of rows in the result (sorted by absolute
        distance from the violated bound, descending).

    Returns
    -------
    pd.DataFrame
        Columns: id_col, *group_cols, value, bound_low, bound_high,
        direction ('above' or 'below'). Empty when no row is extreme.
    """
    needed = list(dict.fromkeys([id_col, *group_cols, value_col]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"build_extreme_value_table: missing columns {missing}")
    work = df[needed].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(
            columns=[id_col, *group_cols, "value", "bound_low", "bound_high", "direction"]
        )

    rows = []
    for keys, sub in work.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        low_bound, high_bound = _percentile_bounds(sub[value_col])
        if low_bound is None:
            continue
        s = sub[value_col]
        above = sub.index[s > high_bound]
        below = sub.index[s < low_bound]
        for idx in above:
            row = {id_col: sub.loc[idx, id_col]}
            for gcol, gval in zip(group_cols, keys):
                row[gcol] = gval
            row["value"] = float(s.loc[idx])
            row["bound_low"] = float(low_bound)
            row["bound_high"] = float(high_bound)
            row["direction"] = "above"
            row["_distance"] = float(s.loc[idx]) - float(high_bound)
            rows.append(row)
        for idx in below:
            row = {id_col: sub.loc[idx, id_col]}
            for gcol, gval in zip(group_cols, keys):
                row[gcol] = gval
            row["value"] = float(s.loc[idx])
            row["bound_low"] = float(low_bound)
            row["bound_high"] = float(high_bound)
            row["direction"] = "below"
            row["_distance"] = float(low_bound) - float(s.loc[idx])
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[id_col, *group_cols, "value", "bound_low", "bound_high", "direction"]
        )
    out = pd.DataFrame(rows).sort_values("_distance", ascending=False).drop(columns="_distance")
    return out.head(top_n).reset_index(drop=True)


def build_outlier_table(
        df: pd.DataFrame,
        value_col: str,
        *,
        group_cols: tuple = ("Dataleverancier", "ProjectIdBSV"),
        id_col: str = "HuisIdBSV",
        top_n: int = 10,
) -> pd.DataFrame:
    """
    Long-form table of rows whose value_col is an outlier within its
    (group_cols) cohort.

    Outlier detection: Tukey fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR) with z-score
    fallback for tiny groups and zero-IQR cases. Columns in the result:
    id_col, *group_cols, value, group_median, z_score. Sorted by
    abs(z_score) descending. Capped at top_n rows; the caller is
    responsible for noting truncation.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation; must include value_col, id_col, and
        every column in group_cols (extras are ignored).
    value_col : str
        Numeric column to scan for outliers.
    group_cols : tuple of str, optional
        Cohort definition. Default ('Dataleverancier', 'ProjectIdBSV').
    id_col : str, optional
        Identifier column for each observation. Default 'HuisIdBSV'.
    top_n : int, optional
        Maximum number of rows in the result. Default 10.

    Returns
    -------
    pd.DataFrame
        Empty if no outliers are found.
    """
    needed = list(dict.fromkeys([id_col, *group_cols, value_col]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"build_outlier_table: missing columns {missing}")
    work = df[needed].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=[id_col, *group_cols, "value", "group_median", "z_score"])

    rows = []
    for keys, sub in work.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = sub[value_col]
        mask = _outlier_mask_tukey(s)
        if not mask.any():
            continue
        med = s.median()
        std = s.std(ddof=0)
        for idx in sub.index[mask]:
            v = float(s.loc[idx])
            z = float(abs(v - med) / std) if std and not pd.isna(std) else float("inf")
            row = {id_col: sub.loc[idx, id_col]}
            for gcol, gval in zip(group_cols, keys):
                row[gcol] = gval
            row["value"] = v
            row["group_median"] = float(med) if not pd.isna(med) else None
            row["z_score"] = z
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[id_col, *group_cols, "value", "group_median", "z_score"])
    out = pd.DataFrame(rows)
    out = out.sort_values("z_score", ascending=False, key=lambda s: s.abs())
    return out.head(top_n).reset_index(drop=True)


def aggregate_project_metrics(hh_table, project_table, col_list):
    """
    Computes aggregated mean metrics per project and reading date.

    This function performs a semi-join between `hh_table` and `project_table`
    on the "ReadingDate" and "ProjectIdBSV" columns. It then groups the data by
    "ProjectIdBSV" and "ReadingDate", calculating the mean for the specified columns.

    Parameters
    ----------
    hh_table : ibis.expr.types.Table
        The household data table containing the metrics to be aggregated.
    project_table : ibis.expr.types.Table
        The project reference table used for filtering the household data.
    col_list : list of str
        List of column names to be aggregated using the mean function.

    Returns
    -------
    ibis.expr.types.Table
        An Ibis table with the following columns:
        - ProjectIdBSV : Project identifier.
        - ReadingDate : Date of the reading.
        - Mean of each column in `col_list`.

    Examples
    --------
    >>> col_list = [
    ...     "ZelfgebruikPer100M2KW",
    ...     "ZonopwekBrutoPer100M2KW",
    ...     "ElektriciteitsgebruikTotaalBrutoPer100M2KW",
    ...     "TerugleveringTotaalNettoPer100M2KW",
    ...     "Warmtebehoefte"
    ... ]
    >>> result = aggregate_project_metrics(hh_table, project_table, col_list)
    >>> result.limit(5).execute()
    """
    # Perform the semi-join on the specified keys
    x = hh_table.semi_join(project_table, ["ReadingDate", "ProjectIdBSV"])

    # Dynamically create aggregation arguments
    agg_kwargs = {col: x[col].mean() for col in col_list}

    # Perform the group_by and aggregation
    hh_agg_per_project = x.group_by("ProjectIdBSV", "ReadingDate").aggregate(**agg_kwargs)

    return hh_agg_per_project


def multiplier_to_convert_to_KWH(interval):
    """
    Calculate multiplier to convert data to KWH.

    Parameters
    ----------
    interval : str
        The interval between timestamps (e.g. "5min", "6h")

    Returns
    -------
    float
        The number with which the data needs to be multiplied
        in order to normalize to hourly data.
    """
    # split string in numeric & string part (e.g. (5, "min"))
    match = re.match(r"(\d+)([a-zA-Z]+)", interval)
    if match:
        number, word = match.groups()
    if word == 'h':
        muliplier = 1/int(number)
    elif word == 'min':
        muliplier = 60/int(number)
    else:
        raise ValueError(f"expected interval to be either in unit 'min' or 'h', got: {word}")
    return muliplier


def normalize_100m2(hh_table: Table, col_names: list, interval: str) -> Table:
    """
    Normalize all supplied columns to 100m2.

    Parameters
    ----------
    hh_table : ibis table
        the household table that has minimal the columns supplied by col_names
    col_names: list
        The columns that will be normalized.
    interval: str
        The interval in which the data is presented (e.g. "5min")
    Returns
    -------
    ibis table
        The normalized household table
    """
    if not isinstance(hh_table, Table):
        raise TypeError("Expected an Ibis table")
    if "Oppervlakte" not in hh_table.columns:
        raise ValueError("Missing column 'Oppervlakte' in ibis table.")

    multiplier = multiplier_to_convert_to_KWH(interval)

    kwargs={}
    for col in col_names:
        kwargs[f"{col}Per100M2KW"] = multiplier * 100 * (
            hh_table[col]
            / hh_table["Oppervlakte"]
        )
    hh_table = hh_table.mutate(**kwargs)
    return hh_table


def get_summer_winter_table(ibis_table: Table) -> Table:
    """
    Filter an Ibis table into summer and winter months.

    Parameters
    ----------
    ibis_table : Table
        An Ibis table containing a "ReadingDate" column.

    Returns
    -------
    tuple[Table, Table]
        A tuple containing two Ibis tables:
        - The first table is filtered to summer months (June, July, August).
        - The second table is filtered to winter months (December, January, February).

    Examples
    --------
    >>> summer, winter = get_summer_winter_table(my_table)
    """
    summer_table = ibis_table.filter(
        (_["ReadingDate"].month() == 6)
        | (_["ReadingDate"].month() == 7)
        | (_["ReadingDate"].month() == 8)
    )

    winter_table = ibis_table.filter(
        (_["ReadingDate"].month() == 12)
        | (_["ReadingDate"].month() == 1)
        | (_["ReadingDate"].month() == 2)
    )

    return summer_table, winter_table


def filter_between_upper_lower_bounds(
        df: pd.DataFrame,
        diff_column: str,
        project_id: Optional[int|str]=None,
        ) -> pd.DataFrame:
    """
    Filter the dataset to stay within upper and lower bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    diff_column : str
        Column name to filter based on quantiles and bounds.
    project_id : optional, int
        Project identifier to filter by.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame for the given column (and project if supplied).
    """
    if project_id is not None:
        df = df[df["ProjectIdBSV"] == project_id]

    upper_bound = df[diff_column].quantile(0.99) * 10

    lower_bound = df[diff_column].min()
    if lower_bound < 0 and df[diff_column].quantile(0.01) < 0:
        lower_bound = df[diff_column].quantile(0.01) * 10
    else:
        lower_bound = 0

    return df[(df[diff_column] <= upper_bound) & (df[diff_column] >= lower_bound)]


def extract_coldest_weeks(
        df: pd.DataFrame,
        var: str="TemperatuurRA",
        project: Optional[str|int]=None
        ):
    """
    Extracts data for the coldest two weeks for each project and interval.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data of which the coldest weeks need to be determined
    var : str, optional
        Column name representing energy consumption (default is "TemperatuurRA").
    project: optional, str, int
        The project id to filter data.

    Returns
    -------
    dict
        Dictionary of DataFrames for the coldest two weeks for each project and interval.
    """
    # If project is provided, filter df based on corresponding project id.
    if project is not None:
        df = df[df["ProjectIdBSV"] == project]
    df["Koudste2WkTemperatuur"] = etdtransform.calculated_columns.mark_coldest_two_weeks(df, avg_var=var)
    df_coldest_weeks = df[df["Koudste2WkTemperatuur"]]

    return df_coldest_weeks


def extract_highest_peak_weeks(df, var='ElektriciteitsgebruikTotaalNetto', days=6, project=None):
    """
    Identify and extract data surrounding the highest energy consumption peak.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing energy and temperature data.
   var : str, optional
        Column name representing energy consumption (default is 'ElektriciteitsgebruikTotaalNetto').
    days : int, optional
        Number of days around the peak to extract (default is 6).
    project: optional, str, int
        The project id to filter data.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe containing the data around the peak.
    """
    if project is not None:
        df = df[df["ProjectIdBSV"] == project]
    df['HoogstePiekWeek'] = etdtransform.calculated_columns.mark_highest_peak(
        df, var=var, days=days)
    return df[df['HoogstePiekWeek']]


def aggregate_to_monthly(
        table: Table,
        cols: list,
        group_by: Optional[list] = None,
) -> Table:
    """
    Aggregate data to monthly means.

    Adds 'year' and 'month' columns derived from 'ReadingDate' and groups by
    those together with any additional group_by columns. Produces one row per
    (group, year, month) combination. Useful as input to heat signature analysis.

    Parameters
    ----------
    table : Table
        Ibis table with a 'ReadingDate' column and the columns in cols.
    cols : list of str
        Columns to aggregate to monthly means.
    group_by : list of str, optional
        Additional grouping columns (e.g. ['HuisIdBSV']). Default is ['HuisIdBSV'].

    Returns
    -------
    Table
        Ibis table grouped to monthly means with columns:
        group_by cols + ['year', 'month'] + mean of each col in cols.
    """
    if group_by is None:
        group_by = ["HuisIdBSV"]

    table = table.mutate(
        year=_["ReadingDate"].year(),
        month=_["ReadingDate"].month(),
    )
    agg_kwargs = {col: table[col].mean() for col in cols}
    return table.group_by(group_by + ["year", "month"]).aggregate(**agg_kwargs)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

GJ_TO_KWH = 277.778


def convert_gj_to_kwh(table: Table, cols: list) -> Table:
    """
    Convert columns from GJ to kWh by multiplying by 277.778.

    Use this for heat production columns (WarmteproductieWarmtepomp,
    WarmteproductieWarmTapwater, WarmteproductieRuimteverwarming) which are
    stored in GJ in the ETD model.

    Parameters
    ----------
    table : Table
        Ibis table containing the columns to convert.
    cols : list of str
        Column names to convert. Each column is replaced with its kWh equivalent.
        The original column is overwritten; rename before calling if you need both.

    Returns
    -------
    Table
        Ibis table with the specified columns multiplied by 277.778.
    """
    kwargs = {col: table[col] * GJ_TO_KWH for col in cols}
    return table.mutate(**kwargs)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def calculate_ventilatievoud(
        table: Table,
        flow_col: str = "DebietVentilatieAanvoer",
) -> Table:
    """
    Compute the ventilatievoud (air changes per hour) for each row.

    Ventilatievoud = DebietVentilatieAanvoer [m³/h] / house volume [m³]
    where house volume = Oppervlakte [m²] × Compactheid [m].

    Requires 'Oppervlakte' and 'Compactheid' to be present in the table
    (typically joined from the index table).

    Parameters
    ----------
    table : Table
        Ibis table with columns: flow_col, 'Oppervlakte', 'Compactheid'.
    flow_col : str, optional
        Column name for ventilation airflow in m³/h. Default is 'DebietVentilatieAanvoer'.

    Returns
    -------
    Table
        Ibis table with an added 'Ventilatievoud' column (air changes per hour).
    """
    for required in [flow_col, "Oppervlakte", "Compactheid"]:
        if required not in table.columns:
            raise ValueError(f"Missing required column '{required}' in table.")
    return table.mutate(
        Ventilatievoud=table[flow_col] / (table["Oppervlakte"] * table["Compactheid"])
    )


# ---------------------------------------------------------------------------
# COP calculation
# ---------------------------------------------------------------------------

_COP_HEAT_COLS = {
    "system":  "WarmteproductieWarmtepompDiff",
    "hp_only": "WarmteproductieWarmtepompDiff",
    "heating": "WarmteproductieRuimteverwarmingDiff",
    "dhw":     "WarmteproductieWarmTapwaterDiff",
}
# Electricity columns: ElektriciteitsgebruikTotaalWarmtepomp has
# resample_method=sum so its non-Diff name in the 60min parquet is
# already per-hour delta in kWh (no Diff suffix in the catalog).
# ElektriciteitsgebruikBoilervat (no suffix) is a cumulative meter --
# use the *Diff variant for the per-hour delta.
_COP_ELEC_COLS = {
    "system":  ("ElektriciteitsgebruikTotaalWarmtepomp", None),
    "hp_only": ("ElektriciteitsgebruikTotaalWarmtepomp", "ElektriciteitsgebruikBoilervatDiff"),
    "heating": ("ElektriciteitsgebruikTotaalWarmtepomp", "ElektriciteitsgebruikBoilervatDiff"),
    "dhw":     ("ElektriciteitsgebruikTotaalWarmtepomp", None),
}
_COP_MODE_FILTER = {
    "heating": [2, 5],
    "dhw":     [1, 6],
}
_COP_PRODUCTION_GUARD = {
    "heating": "WarmteproductieRuimteverwarmingDiff",
    "dhw":     "WarmteproductieWarmTapwaterDiff",
}


def calculate_cop(
        table: Table,
        variant: str = "system",
        mode_col: str = "Mode",
) -> Table:
    """
    Add a COP (coefficient of performance) column to the table.

    Heat production columns (WarmteproductieWarmtepomp, WarmteproductieRuimteverwarming,
    WarmteproductieWarmTapwater) are expected to be in GJ per interval and are converted
    to kWh internally. Electricity columns are expected to be in kWh per interval.

    COP is set to pd.NA where the denominator is zero or where mode / production
    conditions are not met (for 'heating' and 'dhw' variants).

    Parameters
    ----------
    table : Table
        Ibis table with the relevant energy columns and optionally a Mode column.
    variant : str, optional
        Which COP definition to use:
        - 'system'  : WarmteproductieWarmtepomp / ElektriciteitsgebruikTotaalWarmtepomp
        - 'hp_only' : WarmteproductieWarmtepomp / (ElektriciteitsgebruikTotaalWarmtepomp
                      - ElektriciteitsgebruikBoilervat)
        - 'heating' : WarmteproductieRuimteverwarming / (ElektriciteitsgebruikTotaalWarmtepomp
                      - ElektriciteitsgebruikBoilervat), only when Mode in [2, 5]
                      OR WarmteproductieRuimteverwarming > 0
        - 'dhw'     : WarmteproductieWarmTapwater / ElektriciteitsgebruikTotaalWarmtepomp,
                      only when Mode in [1, 6] OR WarmteproductieWarmTapwater > 0
        Default is 'system'.
    mode_col : str, optional
        Column name for the heat pump operating mode. Default is 'Mode'.

    Returns
    -------
    Table
        Ibis table with an added 'COP' column (float, nullable).

    Notes
    -----
    Apply filter_by_mode or filter_by_woningtype before calling if you want
    pre-filtered subsets. This function sets COP to null for irrelevant rows
    rather than dropping them, so the original row count is preserved.
    """
    if variant not in _COP_HEAT_COLS:
        raise ValueError(f"variant must be one of {list(_COP_HEAT_COLS)}; got '{variant}'")

    heat_col = _COP_HEAT_COLS[variant]
    elec_col, subtract_col = _COP_ELEC_COLS[variant]

    # Numerator: heat production in kWh (GJ × 277.778)
    numerator = table[heat_col] * GJ_TO_KWH

    # Denominator: electricity in kWh, optionally minus boilervat
    denominator = table[elec_col]
    if subtract_col is not None:
        denominator = denominator - table[subtract_col]

    # Build the COP expression: null where denominator <= 0
    import ibis
    cop_expr = (
        ibis.case()
        .when(denominator > 0, numerator / denominator)
        .else_(ibis.null())
        .end()
    )

    # For mode-filtered variants, also null where the mode / production condition is not met
    if variant in _COP_MODE_FILTER:
        modes = _COP_MODE_FILTER[variant]
        prod_col = _COP_PRODUCTION_GUARD[variant]
        condition = (
            _[mode_col].isin(modes) | (table[prod_col] * GJ_TO_KWH > 0)
        )
        cop_expr = (
            ibis.case()
            .when(condition & (denominator > 0), numerator / denominator)
            .else_(ibis.null())
            .end()
        )

    return table.mutate(COP=cop_expr)


# ---------------------------------------------------------------------------
# Heat signature fitting
# ---------------------------------------------------------------------------

def classify_coverage(
    df: pd.DataFrame,
    value_cols: list[str],
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Per-(variable, group) coverage rows for the strict empty-data contract.

    The Innax + general analysis pipeline asks: "for variable V, which
    groups have actual data, and which are empty / constant?" The
    answer drives the plot-helper empty-data guards
    (plots that would receive zero rows are skipped upstream) and the
    report-builder annotation (sections call out partial-coverage vars
    in prose).

    The data model + supplier mapping decide whether a variable *should*
    exist for a project; this function reports what *does* exist. No
    judgement is made about gaps -- callers consume the table to decide
    plot/skip and what prose to render.

    Parameters
    ----------
    df : pd.DataFrame
        Already-loaded analysis frame. Must contain ``value_cols`` and
        all ``group_cols``.
    value_cols : list[str]
        Variable column names to classify.
    group_cols : list[str], optional
        Columns to group by. Defaults to
        ``["ProjectIdBSV", "WoningType"]``. Pass a different list when
        your report uses a different grain (e.g. add ``"period"`` if
        the frame already has a season filter applied per row, or
        ``["Dataleverancier"]`` for a supplier-level coverage view).

    Returns
    -------
    pd.DataFrame
        One row per (variable, group). Columns:

          ``variable`` : str
              From ``value_cols``.
          group cols  : as supplied
              The group key values.
          ``n_rows`` : int
              Row count (zero if the (variable, group) combo has no
              rows in ``df`` at all).
          ``n_nonnull`` : int
              Count of non-NA observations of ``variable`` in the
              group.
          ``is_empty`` : bool
              True when ``n_nonnull == 0``.
          ``is_constant`` : bool
              True when ``n_nonnull > 0`` and the variable's
              standard deviation in the group is zero (numeric / bool
              columns only; non-numeric columns return ``False``).

    Notes
    -----
    Mirrors the spirit of ``etdworkflow.analyze_data_report.
    _classify_column_activity`` (per-HH x col) but works at an
    arbitrary group grain so other reports / sub-tables can reuse it.
    """
    if group_cols is None:
        group_cols = ["ProjectIdBSV", "WoningType"]

    missing = [c for c in (*group_cols, *value_cols) if c not in df.columns]
    if missing:
        raise KeyError(
            f"classify_coverage: column(s) {missing!r} not in df. "
            f"df.columns={list(df.columns)!r}"
        )

    # Build one (group, variable) frame by stacking per-variable views.
    parts: list[pd.DataFrame] = []
    base = df[group_cols + value_cols].copy()
    grouped = base.groupby(group_cols, observed=True, dropna=False)
    n_rows = grouped.size().rename("n_rows").reset_index()

    for var in value_cols:
        ser = base[var]
        n_nonnull = (
            base.assign(_nn=ser.notna())
                .groupby(group_cols, observed=True, dropna=False)["_nn"]
                .sum()
                .rename("n_nonnull")
                .reset_index()
        )
        if pd.api.types.is_bool_dtype(ser) or pd.api.types.is_numeric_dtype(ser):
            stds = grouped[var].std(ddof=0).rename("_std").reset_index()
        else:
            stds = n_nonnull.copy()
            stds["_std"] = pd.NA

        sub = n_rows.merge(n_nonnull, on=group_cols, how="outer")
        sub = sub.merge(stds, on=group_cols, how="left")
        sub["variable"] = var
        sub["is_empty"] = sub["n_nonnull"].fillna(0) == 0
        sub["is_constant"] = (
            (sub["n_nonnull"].fillna(0) > 0)
            & (sub["_std"].fillna(1.0) == 0)
        )
        sub = sub.drop(columns=["_std"])
        cols = ["variable", *group_cols, "n_rows", "n_nonnull",
                "is_empty", "is_constant"]
        parts.append(sub[cols])

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["variable", *group_cols, "n_rows", "n_nonnull",
                 "is_empty", "is_constant"]
    )


def compute_stats_table(
        df: pd.DataFrame,
        group_cols: list,
        value_cols: list,
        *,
        interval: str = "60min",
        period_days: int = 365,
        aggregation_methods: dict | None = None,
) -> pd.DataFrame:
    """
    Descriptive stats for each value column grouped by group_cols.

    Returns long-format DataFrame: one row per (group keys x variable)
    with columns: variable, *group_cols, mean, median, std, min, max,
    p25 (25th percentile), p75 (75th percentile), p01 (1st percentile),
    p99 (99th percentile), count.

    Note: ``p25`` and ``p75`` are the inter-quartile boundaries (25/75);
    ``p01`` and ``p99`` are the extreme-tail percentiles. Names match
    the standardised set produced by
    ``etdmap.mapping_helpers.compute_numeric_column_stats`` so consumers
    of either function can use the same column references.

    For variables whose etdmodel ``ResamplingMethode`` is ``"sum"`` (i.e.
    cumulative usage like kWh / liters), an extra ``period_total`` column
    is added: the per-row mean scaled to the period's natural total via
    ``etdanalyze.units.mean_per_row_to_period_total``. For ``"mean"`` /
    ``"max"`` resampled variables (rates / states like degC / kW) no
    period_total is added -- the per-row mean is already the right stat.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame, one row per ``interval``.
    group_cols : list
        Columns to group by (e.g. ``["ProjectIdBSV"]``).
    value_cols : list
        Variables to compute stats for. Columns not in ``df`` are silently
        skipped.
    interval : str, default ``"60min"``
        Row interval, used to scale per-row means to period totals for
        sum-resampled variables.
    period_days : int, default 365
        Days in the period being summarised. Pass 213 for
        verwarmingsperiode (Oct-Apr), 152 for koelingsperiode (May-Sep),
        etc. Caller filters df to the period before calling.
    aggregation_methods : dict, optional
        ``{var: resample_method}`` mapping to override the data-model
        defaults. If None, pulled from
        ``etdmap.data_model.get_aggregation_config()``. Variables absent
        from both this map and the config default to ``"mean"`` (no
        period_total is computed).

    Returns
    -------
    pd.DataFrame
        Long-format stats; empty DataFrame if no value_cols match.

    Notes
    -----
    This function operates on an in-memory DataFrame. The whole
    filtered dataset must fit in RAM. Future-proofing: when datasets
    outgrow RAM, the caller shards (typically by project) and concats
    the per-shard stats; the schema (parent ADR-018) is preserved
    across shards. See parent ADR-017 for the full sharding /
    fallback ladder.
    """
    if aggregation_methods is None:
        try:
            from etdmap.data_model import get_aggregation_config
            cfg = get_aggregation_config()
            aggregation_methods = {
                var: cfg.get(var, {}).get("resample_method", "mean")
                for var in value_cols
            }
        except Exception:
            aggregation_methods = {}

    from etdanalyze.units import mean_per_row_to_period_total

    results = []
    for col in value_cols:
        if col not in df.columns:
            continue
        grp = df.dropna(subset=[col]).groupby(group_cols)[col]
        stats = grp.agg(["mean", "median", "std", "min", "max", "size"]).reset_index()
        stats = stats.rename(columns={"size": "count"})
        q = grp.quantile([0.01, 0.25, 0.75, 0.99]).unstack(level=-1).reset_index()
        q.columns = list(group_cols) + ["p01", "p25", "p75", "p99"]
        stats = stats.merge(q, on=group_cols, how="left")
        # iqr is part of NUMERIC_STATS_SCHEMA (parent ADR-018); compute it
        # here so the per-group output matches the per-Series kernel.
        stats["iqr"] = stats["p75"] - stats["p25"]
        stats.insert(0, "variable", col)

        method = aggregation_methods.get(col, "mean")
        if method == "sum":
            stats["period_total"] = stats["mean"].apply(
                lambda m, _col=col: (
                    mean_per_row_to_period_total(
                        float(m), _col, interval, period_days=period_days
                    )[0]
                    if pd.notna(m) else pd.NA
                )
            )

        results.append(stats)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_stats_table_lazy(
        tbl,
        group_cols: list,
        value_cols: list,
        *,
        interval: str = "60min",
        period_days: int = 365,
        aggregation_methods: dict | None = None,
) -> pd.DataFrame:
    """
    Lazy ibis variant of :func:`compute_stats_table`.

    Accepts an ibis Table (with any joins / filters / projections already
    applied lazily) and pushes ``count``, ``mean``, ``std``, ``min``,
    ``max``, ``median``, ``p01``, ``p25``, ``p75``, ``p99`` into a
    DuckDB ``group_by().aggregate(...)``. Only the small per-group
    result is executed back to pandas. ``iqr`` and (where applicable)
    ``period_total`` are computed in pandas on the small result.

    Output schema matches :func:`compute_stats_table` and the canonical
    ``NUMERIC_STATS_SCHEMA`` (parent ADR-018) so the schema-parity test
    holds across both paths.

    Parameters
    ----------
    tbl : ibis.expr.types.Table
        Lazy ibis expression with the data for stats. May include any
        already-applied joins (``join_index_table``, ``join_weather_data``)
        and filters (e.g. period months via predicate pushdown).
    group_cols : list
        Columns to group by.
    value_cols : list
        Variables to compute stats for. Columns absent from ``tbl`` are
        silently skipped.
    interval, period_days, aggregation_methods :
        Same semantics as :func:`compute_stats_table`. ``period_total``
        is added in pandas after the small result is executed.

    Returns
    -------
    pd.DataFrame
        Long-format stats; empty DataFrame if no value_cols match.

    Notes
    -----
    Implementation is **batched**: one ``.execute()`` per call to this
    function, carrying every column in ``value_cols`` in a single
    ``aggregate(**{...})``. The upstream join chain is therefore
    computed once per call, not once per variable, so the function
    scales cleanly even on heavily-joined ibis tables (climate's
    ``join_index_table -> join_weather_data`` chain).

    See ``etdworkflow/docs/performance/lazy-join-strategies-2026-05-09.md`` for
    the per-col vs batched vs cached comparison on the real climate
    workflow (batched is ~1.86x faster than per-col and matches the
    hoist+pandas approach end-to-end).

    Use this function when the caller has an ibis Table (with whatever
    upstream joins / filters / projections already applied lazily) and
    wants per-group stats without materialising the full DataFrame in
    pandas memory. Stay with :func:`compute_stats_table` when the
    caller already has a pandas DataFrame in hand (e.g. because a
    later plot or Excel block needs it materialised regardless).
    """
    from etdanalyze.units import mean_per_row_to_period_total
    if aggregation_methods is None:
        try:
            from etdmap.data_model import get_aggregation_config
            cfg = get_aggregation_config()
            aggregation_methods = {
                var: cfg.get(var, {}).get("resample_method", "mean")
                for var in value_cols
            }
        except Exception:
            aggregation_methods = {}

    available_cols = [c for c in value_cols if c in tbl.columns]
    if not available_cols:
        return pd.DataFrame()

    # Batched aggregation: ALL variables in ONE .execute() so the upstream
    # joins and the parquet scan are computed once per call instead of
    # once per variable. On the climate workflow (3 vars per call, index +
    # weather joins) this is ~1.9x faster than the per-variable version.
    # See etdworkflow/docs/performance/lazy-join-strategies-2026-05-09.md.
    agg_exprs: dict = {}
    for col in available_cols:
        e = tbl[col]
        agg_exprs[f"{col}__count"] = e.count()
        agg_exprs[f"{col}__mean"] = e.mean()
        agg_exprs[f"{col}__std"] = e.std()
        agg_exprs[f"{col}__min"] = e.min()
        agg_exprs[f"{col}__max"] = e.max()
        agg_exprs[f"{col}__median"] = e.quantile(0.5)
        agg_exprs[f"{col}__p01"] = e.quantile(0.01)
        agg_exprs[f"{col}__p25"] = e.quantile(0.25)
        agg_exprs[f"{col}__p75"] = e.quantile(0.75)
        agg_exprs[f"{col}__p99"] = e.quantile(0.99)

    wide = tbl.group_by(group_cols).aggregate(**agg_exprs).execute()

    # Unpivot to long format. Drop groups with zero non-null observations
    # per variable so output matches the per-column .drop_null([col]) path
    # used by compute_stats_table.
    results = []
    for col in available_cols:
        sub = wide[list(group_cols)].copy()
        for stat in ("count", "mean", "std", "min", "max", "median",
                     "p01", "p25", "p75", "p99"):
            sub[stat] = wide[f"{col}__{stat}"]
        sub["iqr"] = sub["p75"] - sub["p25"]
        sub = sub[sub["count"].fillna(0) > 0].copy()
        sub.insert(0, "variable", col)

        method = aggregation_methods.get(col, "mean")
        if method == "sum":
            sub["period_total"] = sub["mean"].apply(
                lambda m, _col=col: (
                    mean_per_row_to_period_total(
                        float(m), _col, interval, period_days=period_days
                    )[0]
                    if pd.notna(m) else pd.NA
                )
            )

        results.append(sub)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def fit_heat_signature(
        df: pd.DataFrame,
        energy_col: str,
        temp_col: str,
        heating_threshold: str | float = "months",
        dmz_low: float = 10.0,
        dmz_high: float = 15.0,
) -> dict:
    """
    Fit linear regression curves for the heating and DHW (domestic hot water) components
    of the energy signature, and find the stookgrens (heating cut-off temperature).

    The energy signature plots energy use per m2 against outdoor temperature. For heat
    pump buildings it shows two regions:
    - Heating region: downward-sloping curve (more energy at lower temperatures)
    - DHW / base region: roughly flat curve (independent of temperature)

    Both regions are fitted with ordinary least squares using pandas operations only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least energy_col and temp_col columns. Typically monthly
        averages per household from aggregate_to_monthly().
    energy_col : str
        Column with energy values (kWh or W/m²).
    temp_col : str
        Column with outdoor temperature values (°C).
    heating_threshold : str or float, optional
        Method to separate heating from DHW/base region:
        - 'months'  : use only rows where 'month' column is in [10, 11, 12, 1, 2, 3, 4]
                      for the heating fit and [5, 6, 7, 8, 9] for the DHW fit.
                      Requires a 'month' column in df.
        - 'dmz'     : use rows where temp_col < dmz_low for heating, temp_col > dmz_high
                      for DHW. Rows in between are excluded from both fits.
        - float     : temperature cut-off; rows below this value used for heating,
                      rows at or above for DHW.
        Default is 'months'.
    dmz_low : float, optional
        Lower DMZ boundary in °C. Only used when heating_threshold='dmz'. Default is 10.0.
    dmz_high : float, optional
        Upper DMZ boundary in °C. Only used when heating_threshold='dmz'. Default is 15.0.

    Returns
    -------
    dict with keys:
        heating_slope : float or None
            Slope of the heating regression line (kWh per °C).
        heating_intercept : float or None
            Intercept of the heating regression line.
        dhw_level : float or None
            Mean energy level of the DHW / base region.
        stookgrens : float or None
            Temperature at which the heating line crosses the DHW level (°C).
            Computed as (dhw_level - heating_intercept) / heating_slope when slope < 0.
        r2_heating : float or None
            R² of the heating regression.

    Notes
    -----
    - Requires at least 2 non-null data points in each region to fit.
    - Returns None for a value when there is insufficient data to compute it.
    - The stookgrens is only meaningful when the heating slope is negative.
    """
    df = df[[energy_col, temp_col] + (["month"] if "month" in df.columns else [])].dropna()

    # Split into heating and DHW subsets
    if heating_threshold == "months":
        if "month" not in df.columns:
            raise ValueError("heating_threshold='months' requires a 'month' column in df.")
        heating_df = df[df["month"].isin([10, 11, 12, 1, 2, 3, 4])]
        dhw_df = df[df["month"].isin([5, 6, 7, 8, 9])]
    elif heating_threshold == "dmz":
        heating_df = df[df[temp_col] < dmz_low]
        dhw_df = df[df[temp_col] > dmz_high]
    else:
        cutoff = float(heating_threshold)
        heating_df = df[df[temp_col] < cutoff]
        dhw_df = df[df[temp_col] >= cutoff]

    def _ols(sub: pd.DataFrame) -> tuple:
        """Return (slope, intercept, r2) from two Series using pandas operations."""
        if len(sub) < 2:
            return None, None, None
        x = sub[temp_col]
        y = sub[energy_col]
        slope = x.cov(y) / x.var()
        intercept = y.mean() - slope * x.mean()
        y_hat = slope * x + intercept
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else pd.NA
        return slope, intercept, r2

    h_slope, h_intercept, h_r2 = _ols(heating_df)
    dhw_level = dhw_df[energy_col].mean() if len(dhw_df) >= 1 else None

    stookgrens = None
    if h_slope is not None and dhw_level is not None and h_slope != 0:
        stookgrens = (dhw_level - h_intercept) / h_slope

    return {
        "heating_slope": h_slope,
        "heating_intercept": h_intercept,
        "dhw_level": dhw_level,
        "stookgrens": stookgrens,
        "r2_heating": h_r2,
    }


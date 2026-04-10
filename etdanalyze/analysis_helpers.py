import re
from typing import Optional

import etdtransform
import pandas as pd
from ibis import _
from ibis.expr.types import Table


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
    "system":  "WarmteproductieWarmtepomp",
    "hp_only": "WarmteproductieWarmtepomp",
    "heating": "WarmteproductieRuimteverwarming",
    "dhw":     "WarmteproductieWarmTapwater",
}
_COP_ELEC_COLS = {
    "system":  ("ElektriciteitsgebruikTotaalWarmtepomp", None),
    "hp_only": ("ElektriciteitsgebruikTotaalWarmtepomp", "ElektriciteitsgebruikBoilervat"),
    "heating": ("ElektriciteitsgebruikTotaalWarmtepomp", "ElektriciteitsgebruikBoilervat"),
    "dhw":     ("ElektriciteitsgebruikTotaalWarmtepomp", None),
}
_COP_MODE_FILTER = {
    "heating": [2, 5],
    "dhw":     [1, 6],
}
_COP_PRODUCTION_GUARD = {
    "heating": "WarmteproductieRuimteverwarming",
    "dhw":     "WarmteproductieWarmTapwater",
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

    The energy signature plots energy use per m² against outdoor temperature. For heat
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


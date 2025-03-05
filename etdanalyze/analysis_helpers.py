import re

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


def convert_to_KWH(interval):
    """
    Convert data to KWH.

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
    """Normalize all supplied columns to 100m2

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

    multiplier = convert_to_KWH(interval)

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

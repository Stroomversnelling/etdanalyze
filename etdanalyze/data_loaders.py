import etdtransform
from ibis.expr.types import Table

from . import analysis_helpers


def get_project_table(hh_table: Table, interval: str, col_list: list) -> Table:
    """
    Retrieves and enriches a project table by joining it with aggregated metrics.

    This function loads project tables based on a given time `interval`, 
    calculates aggregated metrics from `hh_table`, and performs 
    a left join to enrich the project table with these metrics.

    Parameters
    ----------
    hh_table : ibis.expr.types.Table
        The household data table containing calculated metrics.
    interval : str
        The time interval to select the appropriate project table.
        Must match a key in `etdtransform.load_data.get_project_tables()`.
    col_list : list of str
        List of column names from `hh_table` to be aggregated.

    Returns
    -------
    ibis.expr.types.Table
        An enriched Ibis table that combines the original project table with
        the calculated metrics. It includes:
        - All columns from the project table.
        - Aggregated metrics joined on "ProjectIdBSV" and "ReadingDate".

    Notes
    -----
    - The aggregation is performed using the `aggregate_project_metrics` function.
    - The join is a left join to ensure that all records from the project table are retained.

    Examples
    --------
    >>> interval = "15min"
    >>> col_list = [
    ...     "ZelfgebruikPer100M2KW",
    ...     "ZonopwekBrutoPer100M2KW",
    ...     "ElektriciteitsgebruikTotaalBrutoPer100M2KW",
    ...     "TerugleveringTotaalNettoPer100M2KW",
    ...     "Warmtebehoefte"
    ... ]
    >>> result = get_project_table(hh_table, interval, col_list)
    >>> result.limit(5).execute()
    """
    dfs_tables = etdtransform.load_data.get_project_tables()

    project_table = dfs_tables[interval]

    col_list = [
         "ZelfgebruikPer100M2KW",
         "ZonopwekBrutoPer100M2KW",
         "ElektriciteitsgebruikTotaalBrutoPer100M2KW",
         "TerugleveringTotaalNettoPer100M2KW",
         "Warmtebehoefte"
    ]
    aggregated_metrics = analysis_helpers.aggregate_project_metrics(
            project_table,
            hh_table,
            col_list
            )
    project_table = project_table.join(
        aggregated_metrics,
        ["ProjectIdBSV", "ReadingDate"],
        how="left"
    )
    return project_table


def get_projects():
    """
    Retrieve a sorted list of unique project identifiers.

    Returns
    -------
    list of int or list of str
        A sorted list of unique project identifiers (`ProjectIdBSV`)
        extracted from the project table for the specified interval.

    Notes
    -----
    - The function assumes that the "ProjectIdBSV" column exists in the project table.
    - The resulting list is sorted in ascending order.

    Examples
    --------
    >>> get_projects("15min")
    [1, 2, 4, 7]
    """
    dfs_tables = etdtransform.load_data.get_project_tables()
    projects = sorted(
        dfs_tables["5min"]
        .select("ProjectIdBSV")
        .distinct()
        .execute()["ProjectIdBSV"]
        .tolist()
    )
    return projects

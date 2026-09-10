import etdtransform


def get_projects(interval=None):
    """
    Retrieve a sorted list of unique project identifiers.

    Parameters
    ----------
    interval : str, optional
        The resampled interval whose project table is read (e.g. "15min").
        When None, the first interval the pipeline declares (see
        `etdtransform.calculated_columns.intervals`) is used. The set of
        projects is the same for every interval table, so the choice only
        matters when not every interval file is present.

    Returns
    -------
    list of int or list of str
        A sorted list of unique project identifiers (`ProjectIdBSV`)
        extracted from the project table for the specified interval.

    Raises
    ------
    ValueError
        If the requested interval (or, when None, any declared interval)
        has no project table.

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
    if interval is None:
        declared = etdtransform.calculated_columns.intervals
        available = [k for k in declared if k in dfs_tables]
        if not available:
            raise ValueError(
                f"get_projects: none of the declared intervals {declared} "
                f"have a project table; available keys: {sorted(dfs_tables)}"
            )
        interval = available[0]
    elif interval not in dfs_tables:
        raise ValueError(
            f"get_projects: no project table for interval {interval!r}; "
            f"available keys: {sorted(dfs_tables)}"
        )
    projects = sorted(
        dfs_tables[interval]
        .select("ProjectIdBSV")
        .distinct()
        .execute()["ProjectIdBSV"]
        .tolist()
    )
    return projects

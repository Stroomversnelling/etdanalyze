import etdtransform


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

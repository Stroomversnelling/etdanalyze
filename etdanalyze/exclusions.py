"""Data-quality exclusions with re-validation against current data.

Analyses routinely drop a handful of households or projects whose values
are physically impossible or otherwise untrustworthy. Those exclusions are
static constants in the project code, but the underlying data drifts: it is
re-mapped, cleaned, and re-aggregated regularly. An exclusion reason that
held when it was written ("net-power peak of 286 kW on a single-phase
connection") may no longer hold after the data is cleaned -- in which case
the household is being dropped for a reason that is no longer true, and a
human should review it.

This module gives exclusions a *checkable* shape and a re-validation pass
that recomputes each exclusion's metric from the current data and reports
whether the reason still holds. It does not decide anything: a drifted
exclusion is surfaced as a WARNING and an annotation row, never silently
removed (the value-judgement of whether to keep excluding stays with a
person). Run it whenever the cache is regenerated so drift surfaces every
run.

Exclusion schema -- each exclusion is a dict with these keys:

    huis_id      int    -- HuisIdBSV of the excluded household
    project      int    -- ProjectIdBSV (informational)
    reason       str    -- human-readable reason
    metric       str    -- column reduced per household to test the reason
    reducer      str    -- one of "max" / "min" / "mean" / "sum"
    comparator   str    -- one of ">" / ">=" / "<" / "<=" / "==" / "!="
    threshold    float  -- the reason holds when reduce(metric) <comparator> threshold
    observed     float  -- the value observed when the exclusion was created
    observed_utc str    -- timestamp of that observation (informational)

A project's ``EXCLUDED_HOUSEHOLD_DETAIL``-style dict maps onto this directly:
a recorded ``peak_kw`` becomes ``observed`` with ``metric`` = the net-power
column, ``reducer="max"``, ``comparator=">"``, ``threshold`` = the
physically-impossible limit.
"""
from __future__ import annotations

import logging
import operator
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Required keys on every exclusion dict. ``project`` / ``observed`` /
# ``observed_utc`` are informational and optional.
REQUIRED_FIELDS: tuple[str, ...] = (
    "huis_id",
    "reason",
    "metric",
    "reducer",
    "comparator",
    "threshold",
)

STATUS_STILL_HOLDS = "still_holds"
STATUS_NO_LONGER_HOLDS = "no_longer_holds"
STATUS_HOUSEHOLD_ABSENT = "household_absent"

_COMPARATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_REDUCERS = {"max", "min", "mean", "sum"}

# Column order of the annotation table returned by revalidate_exclusions.
_RESULT_COLUMNS: tuple[str, ...] = (
    "huis_id",
    "project",
    "reason",
    "metric",
    "reducer",
    "comparator",
    "threshold",
    "observed",
    "current",
    "status",
    "note",
)


def _empty_result() -> pd.DataFrame:
    """An empty annotation table with the canonical columns and dtypes."""
    return pd.DataFrame(
        {
            "huis_id": pd.array([], dtype="Int64"),
            "project": pd.array([], dtype="Int64"),
            "reason": pd.array([], dtype="string"),
            "metric": pd.array([], dtype="string"),
            "reducer": pd.array([], dtype="string"),
            "comparator": pd.array([], dtype="string"),
            "threshold": pd.array([], dtype="Float64"),
            "observed": pd.array([], dtype="Float64"),
            "current": pd.array([], dtype="Float64"),
            "status": pd.array([], dtype="string"),
            "note": pd.array([], dtype="string"),
        }
    )


def _validate(exclusion: dict) -> None:
    """Raise ValueError on a malformed exclusion (fail fast)."""
    missing = [k for k in REQUIRED_FIELDS if k not in exclusion]
    if missing:
        raise ValueError(
            f"exclusion {exclusion!r} is missing required field(s) {missing}; "
            f"required: {list(REQUIRED_FIELDS)}"
        )
    if exclusion["reducer"] not in _REDUCERS:
        raise ValueError(
            f"exclusion for huis_id {exclusion['huis_id']!r} has reducer "
            f"{exclusion['reducer']!r}; expected one of {sorted(_REDUCERS)}"
        )
    if exclusion["comparator"] not in _COMPARATORS:
        raise ValueError(
            f"exclusion for huis_id {exclusion['huis_id']!r} has comparator "
            f"{exclusion['comparator']!r}; expected one of {list(_COMPARATORS)}"
        )


def _reduce(series: pd.Series, reducer: str):
    """Reduce a per-household metric Series to a scalar.

    Uses ``skipna=True``: an exclusion metric (e.g. an observed peak) is
    tested against whatever real observations exist; missing intervals are
    "no observation", not zero. An all-NA Series reduces to ``pd.NA`` and is
    handled by the caller as an unconfirmable reason.
    """
    if reducer == "max":
        return series.max(skipna=True)
    if reducer == "min":
        return series.min(skipna=True)
    if reducer == "mean":
        return series.mean(skipna=True)
    # reducer == "sum"
    return series.sum(skipna=True)


def revalidate_exclusions(
    table,
    exclusions: Sequence[dict],
    *,
    id_col: str = "HuisIdBSV",
) -> pd.DataFrame:
    """Recompute each exclusion's metric from current data and report
    whether the reason still holds.

    ``table`` is an ibis Table or a pandas DataFrame that **still contains
    the excluded households** and the metric columns (load it without the
    exclusion filter, or load only the excluded households). For each
    exclusion the metric is reduced per household and compared to the
    threshold:

      - ``still_holds`` -- the reason is reproduced; keep excluding.
      - ``no_longer_holds`` -- the reason is no longer reproduced (data
        cleaned/drifted, or the metric is now all-NA) -> review. A WARNING
        is logged.
      - ``household_absent`` -- the household is no longer in the data ->
        review. A WARNING is logged.

    Returns an annotation table (one row per exclusion, canonical columns
    and nullable dtypes) suitable for caching and for a report to surface.
    An empty ``exclusions`` yields an empty table; nothing is raised.
    """
    if not exclusions:
        return _empty_result()

    for exclusion in exclusions:
        _validate(exclusion)

    ids = [int(e["huis_id"]) for e in exclusions]
    metrics = sorted({str(e["metric"]) for e in exclusions})

    # Materialise just the excluded households and the needed metric columns.
    if isinstance(table, pd.DataFrame):
        present = [c for c in [id_col, *metrics] if c in table.columns]
        df = table.loc[table[id_col].isin(ids), present].copy()
    else:
        # ibis Table: filter + select + execute (small: a handful of HHs).
        present = [c for c in [id_col, *metrics] if c in table.columns]
        df = table.filter(table[id_col].isin(ids)).select(present).execute()

    rows: list[dict] = []
    for e in exclusions:
        huis_id = int(e["huis_id"])
        metric = str(e["metric"])
        reducer = str(e["reducer"])
        comparator = str(e["comparator"])
        threshold = float(e["threshold"])

        hh_rows = df[df[id_col] == huis_id] if id_col in df.columns else df.iloc[0:0]

        current = pd.NA
        note = ""
        if hh_rows.empty:
            status = STATUS_HOUSEHOLD_ABSENT
            note = "household not present in current data"
        elif metric not in hh_rows.columns:
            status = STATUS_NO_LONGER_HOLDS
            note = f"metric {metric!r} absent from current data"
        else:
            reduced = _reduce(hh_rows[metric], reducer)
            if pd.isna(reduced):
                status = STATUS_NO_LONGER_HOLDS
                note = f"metric {metric!r} is all-NA; reason cannot be confirmed"
            else:
                current = float(reduced)
                holds = _COMPARATORS[comparator](current, threshold)
                status = STATUS_STILL_HOLDS if holds else STATUS_NO_LONGER_HOLDS
                if not holds:
                    note = (
                        f"current {reducer}({metric})={current:g} no longer "
                        f"{comparator} {threshold:g}"
                    )

        if status != STATUS_STILL_HOLDS:
            logger.warning(
                "Exclusion review needed: huis_id %s (%s) -- %s. Reason: %s",
                huis_id,
                status,
                note or "see annotation",
                e["reason"],
            )

        rows.append(
            {
                "huis_id": huis_id,
                "project": e.get("project"),
                "reason": e["reason"],
                "metric": metric,
                "reducer": reducer,
                "comparator": comparator,
                "threshold": threshold,
                "observed": e.get("observed"),
                "current": current,
                "status": status,
                "note": note,
            }
        )

    result = pd.DataFrame(rows, columns=list(_RESULT_COLUMNS))
    return result.astype(
        {
            "huis_id": "Int64",
            "project": "Int64",
            "reason": "string",
            "metric": "string",
            "reducer": "string",
            "comparator": "string",
            "threshold": "Float64",
            "observed": "Float64",
            "current": "Float64",
            "status": "string",
            "note": "string",
        }
    )

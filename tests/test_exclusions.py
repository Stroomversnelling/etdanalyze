"""Tests for etdanalyze.exclusions.revalidate_exclusions.

Covers the three outcome classes (still_holds / no_longer_holds /
household_absent), the all-NA-metric and missing-metric edge cases, the
empty-exclusions short-circuit, schema validation, and that a WARNING is
logged for every non-holding row.
"""
import logging

import pandas as pd
import pytest

from etdanalyze.exclusions import (
    STATUS_HOUSEHOLD_ABSENT,
    STATUS_NO_LONGER_HOLDS,
    STATUS_STILL_HOLDS,
    revalidate_exclusions,
)


def _data():
    """Households 65 and 32 present; 65 still has an impossible 286 kW
    peak, 32's peak has been cleaned down to 12 kW. Household 99 is absent.
    """
    return pd.DataFrame(
        {
            "HuisIdBSV": pd.array([65, 65, 32, 32], dtype="Int64"),
            "net_kw": pd.array([286.0, 4.1, 12.0, 3.0], dtype="Float64"),
        }
    )


def _exclusion(huis_id, threshold=20.0, **over):
    base = {
        "huis_id": huis_id,
        "project": 1,
        "reason": "physically-impossible single-phase peak",
        "metric": "net_kw",
        "reducer": "max",
        "comparator": ">",
        "threshold": threshold,
        "observed": 286.0,
        "observed_utc": "2019-01-24 04:25:00",
    }
    base.update(over)
    return base


class TestRevalidateExclusions:
    def test_empty_returns_typed_empty_frame(self):
        out = revalidate_exclusions(_data(), [])
        assert out.empty
        assert "status" in out.columns
        assert out["huis_id"].dtype == "Int64"

    def test_still_holds(self):
        out = revalidate_exclusions(_data(), [_exclusion(65)])
        row = out.iloc[0]
        assert row["status"] == STATUS_STILL_HOLDS
        assert float(row["current"]) == 286.0

    def test_no_longer_holds_after_cleaning(self):
        out = revalidate_exclusions(_data(), [_exclusion(32)])
        row = out.iloc[0]
        assert row["status"] == STATUS_NO_LONGER_HOLDS
        assert float(row["current"]) == 12.0

    def test_household_absent(self):
        out = revalidate_exclusions(_data(), [_exclusion(99)])
        assert out.iloc[0]["status"] == STATUS_HOUSEHOLD_ABSENT
        assert pd.isna(out.iloc[0]["current"])

    def test_all_na_metric_is_unconfirmable(self):
        df = pd.DataFrame(
            {
                "HuisIdBSV": pd.array([65, 65], dtype="Int64"),
                "net_kw": pd.array([pd.NA, pd.NA], dtype="Float64"),
            }
        )
        out = revalidate_exclusions(df, [_exclusion(65)])
        assert out.iloc[0]["status"] == STATUS_NO_LONGER_HOLDS

    def test_warns_on_non_holding(self, caplog):
        with caplog.at_level(logging.WARNING, logger="etdanalyze.exclusions"):
            revalidate_exclusions(_data(), [_exclusion(32), _exclusion(65)])
        # One warning for the cleaned household, none for the still-holding one.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "32" in warnings[0].getMessage()

    def test_bad_reducer_raises(self):
        with pytest.raises(ValueError, match="reducer"):
            revalidate_exclusions(_data(), [_exclusion(65, reducer="median")])

    def test_missing_field_raises(self):
        bad = _exclusion(65)
        del bad["threshold"]
        with pytest.raises(ValueError, match="threshold"):
            revalidate_exclusions(_data(), [bad])

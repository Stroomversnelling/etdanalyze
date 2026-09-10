"""Tests for interval-string handling in analysis_helpers and data_loaders.

The ``interval`` parameters here accept resampled-interval strings
("5min", "60min", "24h"), never pipeline stage names ("default",
"imputed", "calculated"). These tests pin the conversion factors and the
loud, factual rejection of anything that is not an interval -- the
historical failure mode was an ``UnboundLocalError`` when a stage name
reached ``multiplier_to_convert_to_KWH``.
"""
import pytest

import etdtransform
from etdanalyze.analysis_helpers import multiplier_to_convert_to_KWH
from etdanalyze.data_loaders import get_projects


class TestMultiplierToConvertToKWH:
    def test_known_intervals(self):
        assert multiplier_to_convert_to_KWH("5min") == pytest.approx(12.0)
        assert multiplier_to_convert_to_KWH("15min") == pytest.approx(4.0)
        assert multiplier_to_convert_to_KWH("60min") == pytest.approx(1.0)
        assert multiplier_to_convert_to_KWH("6h") == pytest.approx(1 / 6)
        assert multiplier_to_convert_to_KWH("24h") == pytest.approx(1 / 24)

    @pytest.mark.parametrize("stage", ["default", "imputed", "calculated", ""])
    def test_stage_names_raise_value_error(self, stage):
        with pytest.raises(ValueError, match="got:"):
            multiplier_to_convert_to_KWH(stage)

    def test_unknown_unit_raises_value_error(self):
        with pytest.raises(ValueError, match="'min' or 'h'"):
            multiplier_to_convert_to_KWH("3weeks")


class TestGetProjectsIntervalSelection:
    def test_missing_requested_interval_raises(self, monkeypatch):
        monkeypatch.setattr(
            etdtransform.load_data, "get_project_tables", lambda: {"60min": object()}
        )
        with pytest.raises(ValueError, match="no project table for interval '5min'"):
            get_projects("5min")

    def test_no_declared_interval_available_raises(self, monkeypatch):
        monkeypatch.setattr(
            etdtransform.load_data, "get_project_tables", lambda: {"weather": object()}
        )
        with pytest.raises(ValueError, match="none of the declared intervals"):
            get_projects()

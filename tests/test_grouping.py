"""Tests for etdanalyze.grouping registry + label_for_grouping fallback."""

from etdanalyze.grouping import GROUPING_LABELS, label_for_grouping


class TestLabelForGrouping:
    def test_known_column_returns_registry_entry(self):
        result = label_for_grouping("HuisIdBSV")
        assert result == {"singular": "huishouden", "per_phrase": "per huishouden"}

    def test_weerstation_groep_returns_weerstation(self):
        result = label_for_grouping("WeerstationGroep")
        assert result["singular"] == "weerstation"
        assert result["per_phrase"] == "per weerstation"

    def test_unknown_column_falls_back_to_per_col(self):
        result = label_for_grouping("ZomaarEenKolom")
        assert result == {
            "singular": "ZomaarEenKolom",
            "per_phrase": "per ZomaarEenKolom",
        }

    def test_unknown_with_fallback_singular(self):
        result = label_for_grouping("XYZ", fallback_singular="meting")
        assert result == {"singular": "meting", "per_phrase": "per meting"}

    def test_none_column_falls_back_to_rij(self):
        result = label_for_grouping(None)
        assert result == {"singular": "rij", "per_phrase": "per rij"}

    def test_returns_copy_so_callers_cant_mutate_registry(self):
        result = label_for_grouping("HuisIdBSV")
        result["per_phrase"] = "MUTATED"
        # Registry must remain unchanged.
        assert GROUPING_LABELS["HuisIdBSV"]["per_phrase"] == "per huishouden"

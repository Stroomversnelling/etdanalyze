"""Tests for etdanalyze.units conversion helpers.

The conversion functions are the second half of the single-source-of-truth
units regime: every conversion verifies the input column's unit against
the data model (or a per-call override) before applying its multiplier
and returns a ``(converted, display_unit)`` tuple. These tests cover the
happy paths, the negative path (wrong input unit), and the overrides flow
for columns outside the data model.
"""
import pandas as pd
import pytest

from etdanalyze.units import (
    UNIT_CLASS,
    assert_renderable_on_timeline,
    assert_unit_in_classes,
    convert_diff_to_power,
    convert_energy_per_interval_to_power,
    convert_interval_diff,
    infer_interval,
    is_renderable_on_profile,
    power_display_name,
    resolve_unit,
    to_per_m2,
    variable_type,
    _resolve_input_unit,
)


# ---------------------------------------------------------------------------
# _resolve_input_unit  (private but worth covering directly)
# ---------------------------------------------------------------------------


class TestResolveInputUnit:
    def test_data_model_lookup(self):
        assert _resolve_input_unit("ElektriciteitNetgebruikHoog", None) == "kWh"

    def test_diff_inherits_from_parent(self):
        assert _resolve_input_unit("ElektriciteitNetgebruikHoogDiff", None) == "kWh"

    def test_override_takes_precedence(self):
        # Override beats data model -- needed when the caller has applied
        # a display conversion (e.g. kW after convert_diff_to_power).
        assert _resolve_input_unit(
            "ElektriciteitNetgebruikHoog",
            {"ElektriciteitNetgebruikHoog": "kW"},
        ) == "kW"


# ---------------------------------------------------------------------------
# resolve_unit (public)
# ---------------------------------------------------------------------------


class TestResolveUnit:
    def test_data_model_hit(self):
        assert resolve_unit("ElektriciteitNetgebruikHoog") == "kWh"

    def test_override(self):
        assert resolve_unit("X", overrides={"X": "kW"}) == "kW"

    def test_override_empty_string_means_unitless(self):
        # "" is a deliberate, valid value -- the column is explicitly unitless.
        assert resolve_unit("COP_system", overrides={"COP_system": ""}) == ""

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="No unit known"):
            resolve_unit("__nonexistent__")


# ---------------------------------------------------------------------------
# assert_unit_in_classes
# ---------------------------------------------------------------------------


class TestAssertUnitInClasses:
    def test_member_of_one_class_passes(self):
        # kW is in 'power' class -- assertion passes silently.
        assert_unit_in_classes(
            "kW", "any_col", accepted_classes={"power"},
            plot_type="plot_x", axis="y",
        )

    def test_member_of_any_of_multiple_classes_passes(self):
        assert_unit_in_classes(
            "graden C", "any_col",
            accepted_classes={"power", "temperature"},
            plot_type="plot_x", axis="y",
        )

    def test_unitless_passes_ratio_class(self):
        # "" (the explicitly-unitless declaration) is in the 'ratio' class.
        assert_unit_in_classes(
            "", "COP_system", accepted_classes={"ratio"},
            plot_type="plot_x", axis="y",
        )

    def test_wrong_unit_raises_with_accepted_summary(self):
        # Error message describes accepted classes, doesn't prescribe a fix.
        with pytest.raises(ValueError) as exc_info:
            assert_unit_in_classes(
                "kWh", "ElektriciteitNetgebruikHoog",
                accepted_classes={"power", "temperature"},
                plot_type="plot_test", axis="y",
            )
        msg = str(exc_info.value)
        assert "plot_test" in msg
        assert "ElektriciteitNetgebruikHoog" in msg
        assert "'kWh'" in msg
        assert "power" in msg and "temperature" in msg

    def test_unknown_class_name_ignored(self):
        # A class name that's not in UNIT_CLASS contributes no units but
        # doesn't crash; the assertion still raises for the actual unit.
        with pytest.raises(ValueError):
            assert_unit_in_classes(
                "kWh", "any_col",
                accepted_classes={"power", "nonexistent_class"},
                plot_type="plot_x", axis="y",
            )

    def test_unit_class_registry_known_keys(self):
        # A handful of unit-class sanity checks so future drift fails loudly.
        assert "kW" in UNIT_CLASS["power"]
        assert "graden C" in UNIT_CLASS["temperature"]
        assert "kWh" in UNIT_CLASS["energy_per_interval"]
        assert "" in UNIT_CLASS["ratio"]

    def test_power_per_area_includes_per_100m2_variants(self):
        # *Per100M2KW columns in the solar analysis declare their unit as
        # "kW per 100m2"; both the W/m2 standard form and the per-100m2
        # variants live in the same class so profile L3 assertions accept
        # either declaration.
        assert "W/m2" in UNIT_CLASS["power_per_area"]
        assert "kW/m2" in UNIT_CLASS["power_per_area"]
        assert "kW per 100m2" in UNIT_CLASS["power_per_area"]
        assert "W per 100m2" in UNIT_CLASS["power_per_area"]

    def test_power_per_area_accepted_by_profile_class_set(self):
        # Smoke test: a power_per_area unit passes an assertion whose
        # accepted set mirrors what the profile plots accept.
        accepted = {"power", "power_per_area", "temperature",
                    "flow_rate", "state", "ratio", "volume_per_interval"}
        assert_unit_in_classes(
            "kW per 100m2", "ZonopwekBrutoPer100M2KW",
            accepted_classes=accepted,
            plot_type="plot_daily_profile", axis="y",
        )


# ---------------------------------------------------------------------------
# infer_interval
# ---------------------------------------------------------------------------


class TestInferInterval:
    def _single_household(self, freq="5min", n=100):
        idx = pd.date_range("2024-01-01", periods=n, freq=freq)
        return pd.DataFrame({"ReadingDate": idx, "HuisIdBSV": 1, "v": 1.0})

    def _multi_household(self, freq="5min", n=100, n_hh=3):
        frames = []
        for h in range(1, n_hh + 1):
            idx = pd.date_range("2024-01-01", periods=n, freq=freq)
            frames.append(pd.DataFrame({"ReadingDate": idx, "HuisIdBSV": h, "v": 1.0}))
        # Concat preserves per-household contiguity but interleaved global sort
        # collides timestamps across households -- the bug infer_interval
        # solves by computing per-group when group_col is given.
        return pd.concat(frames, ignore_index=True)

    def test_single_household_5min(self):
        df = self._single_household("5min")
        td = infer_interval(df)
        assert td == pd.Timedelta(minutes=5)

    def test_single_household_60min(self):
        df = self._single_household("60min")
        td = infer_interval(df)
        assert td == pd.Timedelta(hours=1)

    def test_multi_household_without_group_col_fails(self):
        # Without group_col, the global sort collides on identical
        # timestamps across households and produces a zero modal delta.
        # Loud failure with a hint to pass group_col.
        df = self._multi_household("5min", n_hh=3)
        with pytest.raises(ValueError, match="zero"):
            infer_interval(df)

    def test_multi_household_with_group_col(self):
        df = self._multi_household("5min", n_hh=3)
        td = infer_interval(df, group_col="HuisIdBSV")
        assert td == pd.Timedelta(minutes=5)

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not in DataFrame"):
            infer_interval(pd.DataFrame({"X": [1, 2]}), col="ReadingDate")


# ---------------------------------------------------------------------------
# convert_energy_per_interval_to_power
# ---------------------------------------------------------------------------


class TestConvertEnergyPerIntervalToPower:
    def test_kwh_5min_to_kw(self):
        # 5 min = 1/12 h. kWh/(1/12 h) = 12 kW. Input 1 kWh -> 12 kW.
        series = pd.Series([1.0, 2.0, 3.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "ElektriciteitNetgebruikHoog",
            pd.Timedelta(minutes=5),
        )
        assert unit == "kW"
        assert list(out.round(6)) == [12.0, 24.0, 36.0]

    def test_kwh_60min_to_kw_identity(self):
        # 60 min = 1 h. kWh/h = kW directly. Series unchanged in value.
        series = pd.Series([1.0, 2.0, 3.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "ElektriciteitNetgebruikHoog", "60min",
        )
        assert unit == "kW"
        assert list(out.round(6)) == [1.0, 2.0, 3.0]

    def test_string_interval_accepted(self):
        series = pd.Series([1.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "ElektriciteitNetgebruikHoog", "15min",
        )
        assert unit == "kW"
        # 15 min = 1/4 h. 1 kWh / (1/4 h) = 4 kW.
        assert out.iloc[0] == pytest.approx(4.0)

    def test_arbitrary_timedelta_supported(self):
        # 3 min: not in the named-interval set. Should still work because
        # the multiplier is derived from the Timedelta, not a lookup.
        series = pd.Series([1.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "ElektriciteitNetgebruikHoog", pd.Timedelta(minutes=3),
        )
        assert unit == "kW"
        # 3 min = 0.05 h. 1 kWh / 0.05 h = 20 kW.
        assert out.iloc[0] == pytest.approx(20.0)

    def test_wh_to_w(self):
        series = pd.Series([1000.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "X", pd.Timedelta(hours=1),
            overrides={"X": "Wh"},
        )
        assert unit == "W"
        assert out.iloc[0] == pytest.approx(1000.0)

    def test_gj_to_kw(self):
        # 1 GJ over 1 h = 1000/3.6 kWh / 1 h = 277.778 kW.
        series = pd.Series([1.0])
        out, unit = convert_energy_per_interval_to_power(
            series, "X", pd.Timedelta(hours=1),
            overrides={"X": "GJ"},
        )
        assert unit == "kW"
        assert out.iloc[0] == pytest.approx(1000 / 3.6)

    def test_input_series_not_mutated(self):
        # CRITICAL contract: the helper returns a new Series; the caller's
        # input is untouched so the original data-model-declared column
        # stays consistent with its declared unit.
        original = pd.Series([1.0, 2.0, 3.0])
        snapshot = original.copy()
        _ = convert_energy_per_interval_to_power(
            original, "ElektriciteitNetgebruikHoog", "60min",
        )
        pd.testing.assert_series_equal(original, snapshot)

    def test_rejects_non_energy_unit(self):
        # Temperature column -- conversion makes no sense.
        with pytest.raises(ValueError, match="Accepted input units"):
            convert_energy_per_interval_to_power(
                pd.Series([20.0]), "TemperatuurBuitenWeerstation",
                pd.Timedelta(minutes=5),
            )

    def test_rejects_zero_interval(self):
        with pytest.raises(ValueError, match="must be > 0"):
            convert_energy_per_interval_to_power(
                pd.Series([1.0]), "ElektriciteitNetgebruikHoog",
                pd.Timedelta(0),
            )

    def test_unknown_column_raises_keyerror(self):
        with pytest.raises(KeyError, match="No unit known"):
            convert_energy_per_interval_to_power(
                pd.Series([1.0]), "__unknown__", "60min",
            )


# ---------------------------------------------------------------------------
# power_display_name
# ---------------------------------------------------------------------------


class TestPowerDisplayName:
    def test_convention_suffix(self):
        assert power_display_name("ElektriciteitNetgebruikHoog") == "ElektriciteitNetgebruikHoog_kW"

    def test_not_enforced(self):
        # The helper is a convention, not a rule -- callers may choose
        # any name. No invariant beyond "returns a string".
        assert isinstance(power_display_name("Anything"), str)

    def test_unitless_override(self):
        # Empty-string override is the explicit unitless marker.
        assert _resolve_input_unit("COP", {"COP": ""}) == ""

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="No unit known"):
            _resolve_input_unit("__not_a_real_column__", None)


# ---------------------------------------------------------------------------
# convert_diff_to_power
# ---------------------------------------------------------------------------


class TestConvertDiffToPower:
    def test_returns_kw_for_kwh_diff(self):
        # 5-min interval -> multiplier is 12 (12 five-min slots per hour).
        # The exact multiplier comes from etdtransform; we just assert the
        # contract that the function returns (series, "kW") and the series
        # has been transformed (not the same reference).
        series = pd.Series([1.0, 2.0, 3.0], dtype="Float64")
        out, unit = convert_diff_to_power(
            series, "ElektriciteitNetgebruikHoogDiff", "5min",
        )
        assert unit == "kW"
        # Sanity: 1 kWh / 5min == 12 kW
        assert out.iloc[0] == pytest.approx(12.0)

    def test_raises_on_non_kwh_input(self):
        # Pass a momentaan kW column -- its data-model unit is kW (not kWh),
        # so the helper must refuse.
        series = pd.Series([1.0, 2.0], dtype="Float64")
        with pytest.raises(ValueError, match="expects input unit 'kWh'"):
            convert_diff_to_power(series, "ElektriciteitVermogen", "5min")

    def test_override_allows_outside_data_model(self):
        series = pd.Series([1.0, 2.0], dtype="Float64")
        out, unit = convert_diff_to_power(
            series, "my_synthetic_kwh_col", "60min",
            overrides={"my_synthetic_kwh_col": "kWh"},
        )
        assert unit == "kW"
        assert out.iloc[0] == pytest.approx(1.0)  # 1 kWh per 60min == 1 kW

    def test_unknown_column_raises_key_error(self):
        series = pd.Series([1.0], dtype="Float64")
        with pytest.raises(KeyError, match="No unit known"):
            convert_diff_to_power(series, "__not_in_data_model__", "5min")


# ---------------------------------------------------------------------------
# to_per_m2
# ---------------------------------------------------------------------------


class TestToPerM2:
    def test_natural_composition_no_target_unit(self):
        # input unit kWh -> output unit "kWh / m2" (no internal rescale).
        # Output spacing matches the etdmodel convention ("liter / uur",
        # "m3 / uur").
        series = pd.Series([100.0, 200.0], dtype="Float64")
        out, unit = to_per_m2(
            series, "ElektriciteitNetgebruikHoog", area=100.0,
        )
        assert unit == "kWh / m2"
        assert list(out) == [1.0, 2.0]

    def test_target_unit_kw_to_w(self):
        # Zon-opwekMomentaan is kW in the data model; convert -> W -> per m2.
        series = pd.Series([1.0, 2.0], dtype="Float64")
        out, unit = to_per_m2(
            series, "Zon-opwekMomentaan", area=100.0, target_unit="W",
        )
        assert unit == "W / m2"
        # 1 kW * 1000 / 100 m2 = 10 W / m2
        assert list(out) == [10.0, 20.0]

    def test_target_unit_same_as_input_is_noop(self):
        series = pd.Series([100.0], dtype="Float64")
        out, unit = to_per_m2(
            series, "ElektriciteitNetgebruikHoog", area=10.0,
            target_unit="kWh",
        )
        assert unit == "kWh / m2"
        assert list(out) == [10.0]

    def test_unsupported_conversion_raises(self):
        series = pd.Series([1.0], dtype="Float64")
        with pytest.raises(ValueError, match="no supported conversion"):
            to_per_m2(
                series, "ElektriciteitNetgebruikHoog", area=10.0,
                target_unit="J",  # kWh -> J not in the supported table
            )

    def test_override_allows_outside_data_model(self):
        series = pd.Series([10.0], dtype="Float64")
        out, unit = to_per_m2(
            series, "my_synthetic_kw_col", area=10.0,
            target_unit="W", overrides={"my_synthetic_kw_col": "kW"},
        )
        assert unit == "W / m2"
        assert list(out) == [1000.0]  # 10 kW * 1000 / 10 m2 = 1000 W / m2

    def test_area_must_be_positive(self):
        series = pd.Series([1.0], dtype="Float64")
        with pytest.raises(ValueError, match="must be > 0"):
            to_per_m2(series, "ElektriciteitNetgebruikHoog", area=0.0)

    def test_area_series_with_na_raises(self):
        series = pd.Series([1.0, 2.0], dtype="Float64")
        area = pd.Series([10.0, pd.NA], dtype="Float64")
        with pytest.raises(ValueError, match="NA value"):
            to_per_m2(series, "ElektriciteitNetgebruikHoog", area=area)

    def test_unknown_column_raises(self):
        series = pd.Series([1.0], dtype="Float64")
        with pytest.raises(KeyError, match="No unit known"):
            to_per_m2(series, "__not_a_real_column__", area=10.0)


# ---------------------------------------------------------------------------
# convert_interval_diff (the generic Diff -> longitudinal-sibling dispatcher)
# ---------------------------------------------------------------------------


class TestConvertIntervalDiff:
    """Generic dispatcher for converting interval-Diff variables to their
    longitudinal-renderable sibling (ADR-003 of etdanalyze). Routes by
    input unit to the converter registered in _INTERVAL_DIFF_CONVERTERS.
    """

    def test_single_column_kwh_returns_three_tuple(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoogDiff": pd.Series(
                [1.0, 2.0], dtype="Float64"
            ),
        })
        out_df, overrides, name_map = convert_interval_diff(
            df, "ElektriciteitNetgebruikHoogDiff", interval="60min",
        )
        assert name_map == {
            "ElektriciteitNetgebruikHoogDiff": "ElektriciteitNetgebruikHoogDiff__kW",
        }
        new_col = name_map["ElektriciteitNetgebruikHoogDiff"]
        assert new_col in out_df.columns
        assert list(out_df[new_col]) == [1.0, 2.0]  # kWh / 1h = kW
        assert overrides == {new_col: {"unit": "kW", "type": "interval_mean"}}

    def test_multi_column_independent_dispatch(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoogDiff": pd.Series([1.0], dtype="Float64"),
            "ElektriciteitNetgebruikLaagDiff": pd.Series([2.0], dtype="Float64"),
        })
        out_df, overrides, name_map = convert_interval_diff(
            df,
            "ElektriciteitNetgebruikHoogDiff",
            "ElektriciteitNetgebruikLaagDiff",
            interval="60min",
        )
        assert set(name_map) == {
            "ElektriciteitNetgebruikHoogDiff",
            "ElektriciteitNetgebruikLaagDiff",
        }
        for new_col in name_map.values():
            assert new_col in out_df.columns
            assert overrides[new_col] == {"unit": "kW", "type": "interval_mean"}

    def test_no_input_columns_raises(self):
        with pytest.raises(ValueError, match="pass one or more"):
            convert_interval_diff(pd.DataFrame(), interval="60min")

    def test_unregistered_unit_raises_with_listing(self):
        df = pd.DataFrame({
            "MyExoticDiff": pd.Series([1.0], dtype="Float64"),
        })
        # Force-declare an input unit that has no registered converter.
        # ("J/cm2" is unitless on the irradiance axis from the data model
        # perspective but no Diff converter exists for it.)
        with pytest.raises(ValueError, match="no converter registered for unit"):
            convert_interval_diff(
                df, "MyExoticDiff", interval="60min",
                overrides={"MyExoticDiff": "J/cm2"},
            )

    def test_input_df_not_mutated(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoogDiff": pd.Series([1.0], dtype="Float64"),
        })
        cols_before = list(df.columns)
        convert_interval_diff(
            df, "ElektriciteitNetgebruikHoogDiff", interval="60min",
        )
        assert list(df.columns) == cols_before

    def test_custom_new_name(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoogDiff": pd.Series([1.0], dtype="Float64"),
        })
        out_df, overrides, name_map = convert_interval_diff(
            df, "ElektriciteitNetgebruikHoogDiff",
            interval="60min",
            new_names={"ElektriciteitNetgebruikHoogDiff": "my_custom_col"},
        )
        assert name_map["ElektriciteitNetgebruikHoogDiff"] == "my_custom_col"
        assert "my_custom_col" in out_df.columns
        assert "my_custom_col" in overrides


# ---------------------------------------------------------------------------
# ADR-003: variable-type override schema and profile-renderability gate
# ---------------------------------------------------------------------------

class TestVariableType:
    """variable_type resolves classification from overrides first, then etdmap."""

    def test_override_type_wins(self):
        ov = {"my_col": {"unit": "kW", "type": "interval_mean"}}
        assert variable_type("my_col", overrides=ov) == "interval_mean"

    def test_etdmap_momentaan_lookup(self):
        # TemperatuurBuitenWarmtepomp is Momentaan=ja in etdmap.
        assert variable_type("TemperatuurBuitenWarmtepomp") == "momentaan"

    def test_etdmap_diff_lookup(self):
        assert variable_type("ElektriciteitNetgebruikHoogDiff") == "diff"

    def test_unknown_column_returns_none(self):
        assert variable_type("totally_made_up_col") is None

    def test_plain_string_override_does_not_classify(self):
        # Plain-string override carries only unit; classification falls
        # through to etdmap, which returns None for unknown columns.
        assert variable_type("my_col", overrides={"my_col": "kW"}) is None


class TestIsRenderableOnProfile:
    """Profile-renderability gate accepts momentaan and interval_mean only."""

    @pytest.mark.parametrize("vtype", ["momentaan", "interval_mean"])
    def test_admits_renderable_types(self, vtype):
        ov = {"c": {"unit": "kW", "type": vtype}}
        assert is_renderable_on_profile("c", overrides=ov) is True

    @pytest.mark.parametrize("vtype", ["diff", "cumulatief"])
    def test_refuses_non_renderable_types(self, vtype):
        ov = {"c": {"unit": "kWh", "type": vtype}}
        assert is_renderable_on_profile("c", overrides=ov) is False

    def test_unclassified_column_refused(self):
        # No override, not in etdmap -> classification is None -> gate refuses.
        assert is_renderable_on_profile("totally_made_up_col") is False


class TestNestedOverrideValidation:
    """Boundary validation: legacy boolean keys raise, type enum is strict."""

    def test_legacy_momentaan_key_rejected_with_hint(self):
        ov = {"c": {"unit": "kW", "momentaan": True}}
        with pytest.raises(ValueError, match="legacy key.*momentaan"):
            resolve_unit("c", overrides=ov)

    def test_legacy_cumulatief_key_rejected(self):
        ov = {"c": {"unit": "kWh", "cumulatief": True}}
        with pytest.raises(ValueError, match="legacy key.*cumulatief"):
            resolve_unit("c", overrides=ov)

    def test_legacy_diff_key_rejected(self):
        ov = {"c": {"unit": "kWh", "diff": True}}
        with pytest.raises(ValueError, match="legacy key.*diff"):
            resolve_unit("c", overrides=ov)

    def test_legacy_aggregatie_methode_key_rejected(self):
        # Old dead-doc key. Reject loud so callers move to the kwarg.
        ov = {"c": {"unit": "kW", "aggregatie_methode": "mean"}}
        with pytest.raises(ValueError, match="legacy key.*aggregatie_methode"):
            resolve_unit("c", overrides=ov)

    def test_unknown_nested_key_rejected(self):
        ov = {"c": {"unit": "kW", "bogus_field": 42}}
        with pytest.raises(ValueError, match="unknown key"):
            resolve_unit("c", overrides=ov)

    def test_invalid_type_enum_value_rejected(self):
        ov = {"c": {"unit": "kW", "type": "not_a_real_type"}}
        with pytest.raises(ValueError, match="allowed values"):
            resolve_unit("c", overrides=ov)

    def test_plain_string_override_still_accepted(self):
        # Backward compat for short-form overrides: unchanged behaviour.
        assert resolve_unit("c", overrides={"c": "kW"}) == "kW"

    def test_unit_only_nested_override(self):
        # Nested override with only "unit" is fine; no type validation triggered.
        assert resolve_unit("c", overrides={"c": {"unit": "kW"}}) == "kW"

    def test_type_only_nested_override(self):
        # Nested override with only "type" is fine; resolve_unit falls
        # through to etdmap for the unit lookup.
        ov = {"TemperatuurBuitenWarmtepomp": {"type": "momentaan"}}
        # etdmap unit for TemperatuurBuitenWarmtepomp is "graden C".
        assert resolve_unit("TemperatuurBuitenWarmtepomp", overrides=ov) == "graden C"


class TestAssertRenderableOnTimeline:
    """The profile-plot gate (ADR-003). Renderable types pass silently;
    every other classification raises ValueError with a tailored hint."""

    def test_momentaan_passes(self):
        # Real etdmap Momentaan=ja column -> renderable, returns None.
        assert assert_renderable_on_timeline("TemperatuurBuitenWarmtepomp") is None

    def test_interval_mean_override_passes(self):
        ov = {"c": {"unit": "kW", "type": "interval_mean"}}
        assert assert_renderable_on_timeline("c", overrides=ov) is None

    def test_cumulative_raises(self):
        ov = {"c": {"unit": "kWh", "type": "cumulatief"}}
        with pytest.raises(ValueError, match="cumulative meter reading"):
            assert_renderable_on_timeline("c", overrides=ov)

    def test_diff_with_registered_unit_points_to_converter(self):
        # kWh-Diff has a registered converter -> hint names convert_interval_diff.
        ov = {"c": {"unit": "kWh", "type": "diff"}}
        with pytest.raises(ValueError, match="convert_interval_diff"):
            assert_renderable_on_timeline("c", overrides=ov)

    def test_diff_with_unregistered_unit_names_registered_units(self):
        # GJ is a valid energy unit but has no registered interval-diff
        # converter -> the error reports that no converter is registered.
        ov = {"c": {"unit": "GJ", "type": "diff"}}
        with pytest.raises(ValueError, match="No converter is currently registered"):
            assert_renderable_on_timeline("c", overrides=ov)

    def test_unclassified_column_raises_with_adr_pointer(self):
        with pytest.raises(ValueError, match="ADR-003"):
            assert_renderable_on_timeline("totally_made_up_col")

    def test_error_message_carries_plot_type_and_axis(self):
        ov = {"c": {"unit": "kWh", "type": "cumulatief"}}
        with pytest.raises(ValueError, match="plot_daily_profile.*colorbar"):
            assert_renderable_on_timeline(
                "c", overrides=ov, plot_type="plot_daily_profile", axis="colorbar")

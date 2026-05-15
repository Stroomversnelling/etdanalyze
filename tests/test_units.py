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
    apply_optional_energy_conversion,
    assert_unit_in_classes,
    convert_diff_to_power,
    convert_energy_per_interval_to_power,
    infer_interval,
    mean_per_row_to_period_total,
    power_display_name,
    resolve_unit,
    to_per_m2,
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
# mean_per_row_to_period_total
# ---------------------------------------------------------------------------


class TestMeanPerRowToPeriodTotal:
    def test_kwh_annual_total(self):
        # mean of 1 kWh per 60-min row over 365 days = 8760 kWh/year.
        total, unit = mean_per_row_to_period_total(
            1.0, "ElektriciteitNetgebruikHoog", "60min",
        )
        assert unit == "kWh"
        assert total == pytest.approx(8760.0)

    def test_unit_passes_through(self):
        # mean 0.5 m3 per row over 365 days at hourly interval = 4380 m3/year.
        total, unit = mean_per_row_to_period_total(
            0.5, "Gasgebruik", "60min",
        )
        assert unit == "m3"
        assert total == pytest.approx(0.5 * 365 * 24)

    def test_override_for_synthetic_column(self):
        total, unit = mean_per_row_to_period_total(
            2.0, "my_synthetic_liters_col", "60min",
            overrides={"my_synthetic_liters_col": "liter"},
        )
        assert unit == "liter"
        assert total == pytest.approx(2.0 * 365 * 24)

    def test_custom_period_days(self):
        # Heating season ~ 213 days (Oct-Apr).
        total, _unit = mean_per_row_to_period_total(
            1.0, "ElektriciteitNetgebruikHoog", "60min", period_days=213,
        )
        assert total == pytest.approx(213 * 24)

    def test_unknown_column_raises(self):
        with pytest.raises(KeyError, match="No unit known"):
            mean_per_row_to_period_total(1.0, "__not_a_real_column__", "60min")


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
# apply_optional_energy_conversion
# ---------------------------------------------------------------------------


class TestApplyOptionalEnergyConversion:
    """The shared hook plot helpers use to opt-in to energy-per-interval
    -> power conversion. The conversion math itself is in
    ``convert_energy_per_interval_to_power``; these tests cover the
    integration semantics: pass-through on None, source-class assertion,
    no input mutation, overrides extended with the converted unit."""

    def test_none_kwarg_is_passthrough(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoog": pd.Series([1.0], dtype="Float64"),
        })
        out_df, out_col, out_overrides = apply_optional_energy_conversion(
            df, "ElektriciteitNetgebruikHoog", None, None,
            plot_type="test_plot",
        )
        # df / col / overrides unchanged
        assert out_df is df
        assert out_col == "ElektriciteitNetgebruikHoog"
        assert out_overrides is None

    def test_kwh_input_yields_kw_column(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoog": pd.Series(
                [1.0, 2.0], dtype="Float64"
            ),
        })
        out_df, out_col, out_overrides = apply_optional_energy_conversion(
            df, "ElektriciteitNetgebruikHoog",
            pd.Timedelta(hours=1), None,
            plot_type="test_plot",
        )
        assert out_col == "ElektriciteitNetgebruikHoog_kW"
        assert out_col in out_df.columns
        # 1 kWh / 1h = 1 kW
        assert list(out_df[out_col]) == [1.0, 2.0]
        assert out_overrides == {"ElektriciteitNetgebruikHoog_kW": "kW"}

    def test_input_df_not_mutated(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoog": pd.Series([1.0], dtype="Float64"),
        })
        cols_before = list(df.columns)
        apply_optional_energy_conversion(
            df, "ElektriciteitNetgebruikHoog",
            pd.Timedelta(hours=1), None,
            plot_type="test_plot",
        )
        assert list(df.columns) == cols_before

    def test_rejects_non_energy_source(self):
        df = pd.DataFrame({
            "TemperatuurBuitenWeerstation": pd.Series([1.0], dtype="Float64"),
        })
        with pytest.raises(ValueError, match="not in the unit classes"):
            apply_optional_energy_conversion(
                df, "TemperatuurBuitenWeerstation",
                pd.Timedelta(hours=1), None,
                plot_type="test_plot",
            )

    def test_existing_overrides_preserved_and_extended(self):
        df = pd.DataFrame({
            "my_energy_var": pd.Series([1.0], dtype="Float64"),
        })
        overrides = {"my_energy_var": "kWh", "other_col": "graden C"}
        _, _, out_overrides = apply_optional_energy_conversion(
            df, "my_energy_var", pd.Timedelta(hours=1), overrides,
            plot_type="test_plot",
        )
        assert out_overrides["my_energy_var"] == "kWh"
        assert out_overrides["other_col"] == "graden C"
        assert out_overrides["my_energy_var_kW"] == "kW"
        # Caller's dict not mutated
        assert "my_energy_var_kW" not in overrides

    def test_accepts_timedelta_string(self):
        df = pd.DataFrame({
            "ElektriciteitNetgebruikHoog": pd.Series([2.0], dtype="Float64"),
        })
        out_df, out_col, _ = apply_optional_energy_conversion(
            df, "ElektriciteitNetgebruikHoog", "1h", None,
            plot_type="test_plot",
        )
        assert list(out_df[out_col]) == [2.0]  # 2 kWh / 1h = 2 kW

"""
Unit-conversion helpers for analysis.

Centralises the math that converts between per-row energy (kWh per N-min
interval), average power (kW), per-area normalisation (W/m2 etc.), and
period totals (kWh per year). Future analyses should route through these
helpers instead of inventing multipliers in each script.

Single source of truth for unit metadata is
``etdmap.data_model.load_unit_map`` (built from etdmodel.csv). Every
conversion in this module looks up the input column's unit there (or via
a per-call ``overrides=`` dict) and validates it against what the
conversion expects. The pre-data-model habit of trusting the caller to
get input units right is no longer allowed -- callers who omit the
override and pass a column not in the data model get a loud ``KeyError``;
callers who pass a column whose data-model unit doesn't match the
conversion's expected input unit get a loud ``ValueError``.

Each conversion returns a ``(converted, display_unit)`` tuple. Hand the
``display_unit`` to ``etdanalyze.plot_functions.axis_label_with_unit`` via
``overrides`` so plot labels stay in sync with the converted values.

Supported intervals: 5min, 15min, 60min, 6h, 24h.
"""

from typing import Optional

import pandas as pd

from etdmap.data_model import (
    cumulative_columns,
    diff_columns,
    load_unit_map,
    momentaan_columns,
)
from etdtransform.calculated_columns import switch_multiplier


_INTERVAL_HOURS = {
    "5min":  5 / 60,
    "15min": 15 / 60,
    "60min": 1.0,
    "6h":    6.0,
    "24h":   24.0,
}

_HOURS_PER_DAY = 24
_DAYS_PER_YEAR = 365

# Loaded once at module import. Mirror of the dict used by
# etdanalyze.plot_functions.unit_for_column; loaded here too so this
# module has no import-time dependency on plot_functions (which itself
# imports from this module via lazy in-function imports).
_UNIT_MAP = load_unit_map()


# ---------------------------------------------------------------------------
# Metadata (Layer 3): per-variable classification.
#
# Profile plots gate on Momentaan / Diff / Cumulatief (see ADR-003).
# The classification is exposed by etdmap as module-level list
# constants (cumulative_columns, momentaan_columns, diff_columns) --
# the predicates below are thin membership checks over those lists,
# with per-call override support for synthetic columns.
# ---------------------------------------------------------------------------


def _override_field(overrides: "Optional[dict]", col_name: str, field: str):
    """Return overrides[col_name][field] when overrides uses the nested
    shape ``{col: {"unit"|"momentaan"|"cumulatief"|"aggregatie_methode": ...}}``.

    Returns ``None`` when the override is absent or uses the plain
    ``{col: unit_str}`` shape (which only carries the unit, no metadata).
    """
    if overrides is None or col_name not in overrides:
        return None
    val = overrides[col_name]
    if isinstance(val, dict):
        return val.get(field)
    return None


def _resolve_input_unit(col_name: str, overrides: Optional[dict]) -> str:
    """
    Look up the unit for ``col_name``, allowing per-call overrides.

    Lookup order: overrides -> data model -> raise. There is no silent
    fallback; a column with no known unit is a bug the caller must fix
    by either passing an override or registering the column in the data
    model.

    Override shapes both supported:
    - plain ``{col: "kW"}`` -- unit only.
    - nested ``{col: {"unit": "kW", "momentaan": True}}`` -- unit plus
      timeline-renderability metadata (see ADR-003).
    """
    if overrides is not None and col_name in overrides:
        val = overrides[col_name]
        if isinstance(val, dict):
            if "unit" in val:
                return val["unit"]
            # Nested dict without a "unit" key falls through to data model.
        else:
            return val
    unit = _UNIT_MAP.get(col_name, "")
    if not unit:
        raise KeyError(
            f"No unit known for column {col_name!r}. "
            f"Pass overrides={{{col_name!r}: '<unit>'}} or add the column "
            f"to etdmodel.csv (Grist source of truth)."
        )
    return unit


def resolve_unit(col_name: str, overrides: Optional[dict] = None) -> str:
    """Public alias for the strict unit resolver.

    Looks up ``col_name``'s unit via ``overrides`` first, then the data
    model. Raises ``KeyError`` if neither has it. Returns ``""`` for
    explicitly-unitless declarations (``overrides={col: ""}``).
    """
    return _resolve_input_unit(col_name, overrides)


# ---------------------------------------------------------------------------
# Unit-class registry (Layer 3 of the unit-handling architecture).
#
# Plot helpers assert that the values they're about to render belong to a
# unit class that makes sense for the plot type. Pure labelling (Layer 2)
# is handled by ``resolve_unit`` / ``axis_label_with_unit`` and doesn't
# care which class a unit is in; Layer 3 is what prevents a caller from
# (e.g.) passing a kWh column to a daily-profile plot, where the y-axis
# only reads as power.
# ---------------------------------------------------------------------------

UNIT_CLASS: dict[str, set[str]] = {
    "power":               {"kW", "W"},
    "temperature":         {"graden C"},
    "flow_rate":           {"m3 / uur", "liter / uur", "1 / uur"},
    "state":               {"ppm", "%", "graden C", "kPa", "Pa"},
    "ratio":               {""},                          # explicitly unitless
    "energy_per_interval": {"kWh", "Wh", "GJ"},
    "volume_per_interval": {"liter", "m3"},
    "power_per_area":      {"W/m2", "kW/m2", "W / m2", "kW / m2",
                             "W per 100m2", "kW per 100m2"},
    "energy_per_area":     {"kWh/m2", "Wh/m2", "kWh / m2", "Wh / m2"},
}


def assert_unit_in_classes(
    unit: str,
    col_name: str,
    accepted_classes: set[str],
    *,
    plot_type: str = "<plot>",
    axis: str = "value",
) -> None:
    """Raise ``ValueError`` if ``unit`` is not a member of any accepted class.

    Used by plot helpers (Layer 3) to enforce that the values handed in
    belong to a unit class that fits the plot type. The error message
    describes what the plot **accepts**; it does not prescribe a fix
    (callers are free to convert upstream, declare an alternative
    override, or pick a different plot).

    Parameters
    ----------
    unit : str
        The resolved unit string (from ``resolve_unit``). ``""`` is a
        valid value -- it's a member of the ``"ratio"`` class.
    col_name : str
        The column name the unit was resolved for; used in the error
        message.
    accepted_classes : set[str]
        Names of unit classes (keys of ``UNIT_CLASS``) the caller's
        plot type accepts.
    plot_type : str
        Name of the plot helper invoking the assertion; used in the
        error message.
    axis : str
        Which axis / role the column plays in the plot (e.g. ``"y"``,
        ``"x"``); used in the error message.

    Raises
    ------
    ValueError
        ``unit`` is not in any class listed in ``accepted_classes``.
    """
    accepted_units: set[str] = set()
    for cls in accepted_classes:
        accepted_units |= UNIT_CLASS.get(cls, set())

    if unit in accepted_units:
        return

    accepted_summary = "; ".join(
        f"{cls} ({sorted(UNIT_CLASS[cls])})"
        for cls in sorted(accepted_classes)
        if cls in UNIT_CLASS
    )
    raise ValueError(
        f"{plot_type}: {axis} column {col_name!r} has unit {unit!r}, "
        f"which is not in the unit classes this plot accepts. "
        f"Accepted: {accepted_summary}."
    )


# ---------------------------------------------------------------------------
# Metadata predicates and timeline-renderability gate.
#
# These are the Layer 3 check for *profile* plots (anything where the
# x-axis is time-of-day, day-of-week, day-of-year, or a heatmap of those
# combinations). The check is metadata-driven: it asks the data model
# whether each variable is an instantaneous reading. The unit class is
# irrelevant -- a Momentaan=ja variable in any unit is renderable on a
# timeline, whether it's degrees, ppm, J/cm2, or anything else.
#
# Physics-specific plots (plot_energy_signature, plot_cop_vs_temp,
# plot_cop_vs_delta_t) keep the UNIT_CLASS check because they require a
# specific physical quantity on each axis, not just instantaneous-ness.
# ---------------------------------------------------------------------------


def _bool_truthy(value) -> bool:
    """Coerce an override choice value to a Python bool, strictly.

    Accepts exactly:
    - ``True`` / ``False`` (Python bool)
    - ``"ja"`` / ``"nee"`` (strict -- matches the Grist choice field
      values, no lowercasing of arbitrary input).

    Anything else raises ``ValueError``. Override values come from
    application code, but the choice contract is the same as the data
    model's (ADR-003): sloppy free-text would defeat the gate.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "ja":
            return True
        if value == "nee":
            return False
    raise ValueError(
        f"choice override must be True / False / 'ja' / 'nee', got "
        f"{type(value).__name__}: {value!r}"
    )


def is_momentaan(col_name: str, overrides: Optional[dict] = None) -> bool:
    """True when the column is declared instantaneous (``Momentaan='ja'``).

    Lookup order: ``overrides[col]["momentaan"]`` -> membership in
    ``etdmap.data_model.momentaan_columns`` -> ``False``. An override
    value can be a Python bool or one of the model's ``"ja"``/``"nee"``
    strings; anything else raises (see ``_bool_truthy``).
    """
    ov = _override_field(overrides, col_name, "momentaan")
    if ov is not None:
        return _bool_truthy(ov)
    return col_name in momentaan_columns


def is_cumulatief(col_name: str, overrides: Optional[dict] = None) -> bool:
    """True when the column is a running-total meter reading (``Cumulatief='ja'``).

    Cumulative readings are never plottable on a profile -- their values
    monotonically increase regardless of the underlying phenomenon. Plot
    the corresponding Diff variant instead, which represents per-row
    delta. Membership-checked against
    ``etdmap.data_model.cumulative_columns``.
    """
    ov = _override_field(overrides, col_name, "cumulatief")
    if ov is not None:
        return _bool_truthy(ov)
    return col_name in cumulative_columns


def is_diff_variable(col_name: str, overrides: Optional[dict] = None) -> bool:
    """True when the column is a dataset-cadence delta (``Diff='ja'``).

    Diff variables are interval deltas of cumulative meter readings --
    e.g. ``ElektriciteitNetgebruikLaagDiff`` is the per-5-min kWh delta
    of ``ElektriciteitNetgebruikLaag``. Their raw values depend on row
    cadence, so they are not directly renderable on time profiles.
    Conversion to an instantaneous-style sibling (e.g. kWh-per-row ->
    kW) requires a unit-aware converter -- see
    ``assert_renderable_on_timeline`` for the available conversion paths
    and what to do when none exists for a variable's unit.

    The ``Diff`` column in the data model is the authoritative signal --
    not the variable-name suffix. Membership-checked against
    ``etdmap.data_model.diff_columns``. Several Diff variables (e.g.
    ``TerugleveringTotaalNetto``, ``Netuitwisseling``) do not carry the
    suffix in their name but are still deltas; their explicit
    ``Diff="ja"`` row puts them in the list.

    Per-call override: ``overrides={col: {"diff": True}}`` for synthetic
    columns.
    """
    ov = _override_field(overrides, col_name, "diff")
    if ov is not None:
        return _bool_truthy(ov)
    return col_name in diff_columns


# ---------------------------------------------------------------------------
# Interval-Diff conversion registry and dispatcher.
#
# Diff variables (Diff="ja") hold dataset-cadence deltas of cumulative
# meter readings -- kWh per 5-min row, m3 per row, etc. Their raw values
# depend on row cadence; they need converting to an instantaneous sibling
# before any longitudinal output (profile plot, time-series table) can
# render them honestly.
#
# Each unit gets its own converter function. The dispatcher
# ``convert_interval_diff`` routes by input unit; it has no physics
# knowledge of its own. Adding a new convertible unit (m3, liter, J/cm^2)
# is one function + one registry entry; nothing else moves.
# ---------------------------------------------------------------------------


def _unit_slug(unit: str) -> str:
    """Compact identifier-safe representation of a unit string.

    Used to build default column names for converted variables, e.g.
    ``"ZonopwekBrutoDiff__kW"`` or ``"GasgebruikDiff__m3_uur"``. No
    semantic content -- just a deterministic way to embed the unit in
    a Python identifier.
    """
    return (
        unit
        .replace("/", "_")
        .replace(" ", "")
        .replace("²", "2")   # superscript 2 (m^2)
    )


def _converter_kwh_per_interval(
    series: pd.Series,
    col_name: str,
    *,
    interval: str,
    overrides: Optional[dict] = None,
    new_name: Optional[str] = None,
) -> tuple[pd.Series, str, str]:
    """kWh-per-row Diff -> average kW. Wraps ``convert_diff_to_power``."""
    converted, display_unit = convert_diff_to_power(
        series, col_name, interval, overrides=overrides,
    )
    name = new_name if new_name is not None else f"{col_name}__{_unit_slug(display_unit)}"
    return converted, name, display_unit


def _converter_volume_per_interval(
    series: pd.Series,
    col_name: str,
    *,
    interval: str,
    overrides: Optional[dict] = None,
    new_name: Optional[str] = None,
) -> tuple[pd.Series, str, str]:
    """volume-per-row Diff -> average volumetric flow rate.

    Same time-multiplier math as kWh-per-row -> kW (kWh per row * rows
    per hour = kWh per hour = kW). Here: liter per row * rows per hour =
    liter per hour ("liter / uur"); analogous for m3 -> "m3 / uur".
    Display unit preserves the input volume unit with " / uur" appended,
    landing the result in the ``flow_rate`` UNIT_CLASS so the plot
    helpers' Layer-3 assertions accept it without further coercion.
    """
    input_unit = _resolve_input_unit(col_name, overrides)
    if input_unit not in {"liter", "m3"}:
        raise ValueError(
            f"_converter_volume_per_interval expects input unit "
            f"'liter' or 'm3' for {col_name!r}; got {input_unit!r}."
        )
    converted = series * switch_multiplier(interval)
    display_unit = f"{input_unit} / uur"
    name = new_name if new_name is not None else f"{col_name}__{_unit_slug(display_unit)}"
    return converted, name, display_unit


_INTERVAL_DIFF_CONVERTERS: "dict[str, callable]" = {
    "kWh":   _converter_kwh_per_interval,
    "liter": _converter_volume_per_interval,
    "m3":    _converter_volume_per_interval,
    # Future entries land here alongside their converter implementation.
    # Each entry: input_unit -> callable with the same signature shape:
    #     (series, col_name, *, interval, overrides, new_name)
    #     -> (converted_series, new_name, display_unit)
}


def convert_interval_diff(
    df: pd.DataFrame,
    *col_names: str,
    interval: str,
    overrides: Optional[dict] = None,
    new_names: "Optional[dict[str, str]]" = None,
) -> "tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, str]]":
    """Convert one or more interval-Diff columns to their longitudinal sibling.

    Looks up each column's unit (overrides -> data model), dispatches to
    the converter registered for that unit in
    ``_INTERVAL_DIFF_CONVERTERS``, attaches the result to the returned
    DataFrame, and builds an ``overrides`` dict ready to pass to any
    plot helper.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame containing each ``col_name``.
    *col_names : str
        One or more column names to convert. Mixed units in one call
        are fine: each column dispatches independently.
    interval : str
        Row cadence label (``"5min"``, ``"15min"``, ``"60min"``, ...).
        Passed through to each converter; converters may use it or
        ignore it depending on their physics.
    overrides : dict, optional
        Per-call unit / classification overrides used by the unit
        resolver. Same shape as elsewhere in this module.
    new_names : dict[str, str], optional
        Per-column override for the new column name. Default name is
        ``f"{col}__{unit_slug}"``.

    Returns
    -------
    (df_with_new_cols, overrides, name_map)
        - ``df_with_new_cols``: shallow copy of ``df`` with one new
          column per requested conversion attached.
        - ``overrides``: keyed by NEW column name, ready to pass to any
          plot helper:
          ``{"ZonopwekBrutoDiff__kW": {"unit": "kW", "momentaan": True}, ...}``.
        - ``name_map``: ``{original_col: new_col_name}`` -- useful for
          callers that loop over original names (e.g. composing plot
          titles with the original variable name).

    Raises
    ------
    ValueError
        No converter is registered for one of the requested columns'
        units. Error names the registered units.
    """
    if not col_names:
        raise ValueError(
            "convert_interval_diff: pass one or more column names."
        )
    new_names = dict(new_names) if new_names else {}
    # Build the attachments dict, then do ONE df.assign(**attachments)
    # at the end. df.assign returns a new df with the new columns
    # attached but SHARES the original column arrays (no deep copy),
    # so memory + CPU cost is just the new converted Series themselves.
    attachments: dict = {}
    out_overrides: dict = {}
    name_map: "dict[str, str]" = {}
    for col in col_names:
        unit = _resolve_input_unit(col, overrides)
        converter = _INTERVAL_DIFF_CONVERTERS.get(unit)
        if converter is None:
            registered = ", ".join(sorted(_INTERVAL_DIFF_CONVERTERS)) or "(none)"
            raise ValueError(
                f"convert_interval_diff: no converter registered for unit "
                f"{unit!r} (column {col!r}). Registered units: {registered}. "
                f"Add a converter function and register it in "
                f"_INTERVAL_DIFF_CONVERTERS in etdanalyze.units before "
                f"calling this for {unit!r}-Diff variables."
            )
        converted, new_name, display_unit = converter(
            df[col], col,
            interval=interval,
            overrides=overrides,
            new_name=new_names.get(col),
        )
        attachments[new_name] = converted
        out_overrides[new_name] = {"unit": display_unit, "momentaan": True}
        name_map[col] = new_name
    out_df = df.assign(**attachments)
    return out_df, out_overrides, name_map


def assert_renderable_on_timeline(
    col_name: str,
    *,
    overrides: Optional[dict] = None,
    plot_type: str = "<plot>",
    axis: str = "y",
) -> None:
    """Raise ``ValueError`` if ``col_name`` cannot be plotted on a time profile.

    A column is renderable on a timeline when its data-model metadata
    declares it instantaneous (``Momentaan='ja'``), or a per-call
    override declares it so.

    Two specific failure modes surface with tailored hints:

    - ``Cumulatief='ja'``: cumulative meter reading. Plot the
      corresponding Diff variant.
    - ``Diff='ja'``: dataset-cadence delta. Call
      :func:`convert_interval_diff` upstream of the plot to convert it
      to an instantaneous sibling, then pass the new column name plus
      the returned overrides to the plot helper.

    Parameters
    ----------
    col_name : str
        Column name being plotted; looked up in the data model first.
    overrides : dict, optional
        Per-call overrides. Plain ``{col: "kW"}`` covers unit only;
        nested ``{col: {"unit": "kW", "momentaan": True}}`` adds the
        timeline-renderability flag for synthetic columns.
    plot_type : str
        Name of the calling plot helper. Used in error messages.
    axis : str
        Which axis the column drives ("y" / "colorbar" / etc.).

    Raises
    ------
    ValueError
        The column is not renderable: it is cumulative, an unconverted
        Diff, or has no Momentaan flag in the data model and no
        override declaring it instantaneous.
    """
    if is_momentaan(col_name, overrides):
        return
    if is_cumulatief(col_name, overrides):
        raise ValueError(
            f"{plot_type}: {axis} column {col_name!r} is a cumulative meter "
            f"reading (Cumulatief=ja). Cumulative readings are not "
            f"plottable on a time profile -- their values monotonically "
            f"increase regardless of the underlying phenomenon. Plot the "
            f"corresponding Diff variant instead."
        )
    if is_diff_variable(col_name, overrides):
        unit = ""
        try:
            unit = _resolve_input_unit(col_name, overrides)
        except KeyError:
            pass
        registered = sorted(_INTERVAL_DIFF_CONVERTERS)
        if unit in _INTERVAL_DIFF_CONVERTERS:
            raise ValueError(
                f"{plot_type}: {axis} column {col_name!r} is a "
                f"dataset-cadence delta (Diff=ja) in {unit!r}. Call "
                f"etdanalyze.units.convert_interval_diff(df, {col_name!r}, "
                f"interval=<row cadence>) before plotting, then pass the "
                f"returned new-column name and overrides to the plot helper."
            )
        registered_str = ", ".join(registered) or "(none)"
        raise ValueError(
            f"{plot_type}: {axis} column {col_name!r} is a dataset-cadence "
            f"delta (Diff=ja) in {unit!r}. No converter is currently "
            f"registered for {unit!r}-Diff. Registered units: {registered_str}. "
            f"Add a converter in etdanalyze.units (register in "
            f"_INTERVAL_DIFF_CONVERTERS) before plotting."
        )
    raise ValueError(
        f"{plot_type}: {axis} column {col_name!r} has no Momentaan flag in "
        f"the data model and no per-call override declares it instantaneous. "
        f"Set Momentaan=ja in etdmodel.csv (Grist source of truth) if the "
        f"column belongs in the data model, or pass "
        f"overrides={{{col_name!r}: {{'momentaan': True}}}} for a synthetic "
        f"column."
    )


def power_display_name(col_name: str) -> str:
    """Conventional name for the kW-display sibling of an energy column.

    Convenience for callers who want a consistent naming pattern when
    deriving a power column from an energy column. Not enforced
    anywhere: callers may pick any name for the new column. The
    declared unit of the new column is the caller's responsibility
    (typically passed via ``overrides`` to plot helpers).
    """
    return f"{col_name}_kW"


def interval_hours(interval: str) -> float:
    """
    Return hours per row for an interval string.

    Parameters
    ----------
    interval : str
        One of ``"5min"``, ``"15min"``, ``"60min"``, ``"6h"``, ``"24h"``.

    Returns
    -------
    float
        Hours per row (e.g. 5/60 for "5min", 1.0 for "60min").

    Raises
    ------
    ValueError
        If ``interval`` is not a supported value.
    """
    if interval not in _INTERVAL_HOURS:
        raise ValueError(
            f"Unknown interval {interval!r}; supported: "
            f"{sorted(_INTERVAL_HOURS)}"
        )
    return _INTERVAL_HOURS[interval]


def infer_interval(
    df: pd.DataFrame,
    col: str = "ReadingDate",
    group_col: Optional[str] = None,
) -> pd.Timedelta:
    """Return the modal time delta between rows as a ``pd.Timedelta``.

    Arbitrary precision -- not constrained to a named-interval set.
    Callers can derive hours via ``.total_seconds() / 3600`` or compare
    against named intervals if they need to display "5min" etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a datetime ``col`` and (optionally) a
        ``group_col``.
    col : str, default ``"ReadingDate"``
        Datetime column to inspect.
    group_col : str, optional
        If provided, deltas are computed **within each group**'s
        sorted series and the modal across all groups is returned.
        This is the right behaviour for multi-household DataFrames:
        rows from different households interleave at the same
        timestamp when sorted globally, producing a spurious
        zero-second modal delta. Without ``group_col``, the global
        modal is returned -- only valid for a single contiguous
        time series.

    Returns
    -------
    pd.Timedelta
        The modal delta. Use ``.total_seconds() / 3600`` to obtain
        hours.

    Raises
    ------
    ValueError
        Column missing, fewer than two rows (or fewer than two per
        group), or the modal delta is zero (suggests forgetting to
        pass ``group_col`` on a multi-household DataFrame).
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not in DataFrame; cannot infer interval")

    if group_col is not None and group_col in df.columns:
        deltas = (
            df.sort_values([group_col, col])
              .groupby(group_col, observed=True)[col]
              .diff()
              .dropna()
        )
    else:
        deltas = df[col].sort_values().diff().dropna()

    if len(deltas) == 0:
        raise ValueError(
            f"infer_interval: cannot infer from {col!r}: no non-NA deltas. "
            f"Need at least 2 rows"
            f"{' per group' if group_col else ''}."
        )

    modal_series = deltas.mode()
    if len(modal_series) == 0:
        raise ValueError(
            f"infer_interval: no modal delta found in {col!r}."
        )
    modal = modal_series.iloc[0]
    if modal == pd.Timedelta(0):
        hint = (
            " If the DataFrame contains multiple households / time series, "
            "pass group_col=<id column> so deltas are computed within each "
            "series rather than across the global sort."
        ) if group_col is None else ""
        raise ValueError(
            f"infer_interval: modal delta in {col!r} is zero seconds.{hint}"
        )
    return modal


def convert_energy_per_interval_to_power(
    series: pd.Series,
    col_name: str,
    interval: "pd.Timedelta | str",
    *,
    overrides: Optional[dict] = None,
) -> tuple[pd.Series, str]:
    """Convert per-row energy (energy-per-``interval``) to mean power.

    Each row's value is the energy accumulated over ``interval``;
    dividing by the interval in hours yields mean power across the
    same interval. The input ``series`` is **not modified** -- a new
    Series is returned. Callers should assign the result to a new
    column whose unit they declare via ``overrides`` at the next step
    (typically a plot helper). The original data-model-declared column
    stays unchanged so the model remains the single source of truth
    for raw column units.

    Accepted input units: ``"kWh"``, ``"Wh"``, ``"GJ"``. Anything else
    raises ``ValueError`` -- the helper does not silently guess.

    Parameters
    ----------
    series : pd.Series
        Energy values, one per row. **Not modified.**
    col_name : str
        Column name (must resolve to a recognised energy unit via
        ``resolve_unit`` and ``overrides``).
    interval : pd.Timedelta or interval string
        Row spacing. Interval strings (``"5min"``, ``"15min"``,
        ``"60min"``, ``"6h"``, ``"24h"``, or any ``pd.Timedelta``-
        parseable string) are parsed via ``pd.Timedelta``. Arbitrary
        non-named intervals are supported.
    overrides : dict, optional
        Per-call ``{col: unit}`` for columns not in the data model.

    Returns
    -------
    (pd.Series, str)
        A new Series of mean-power values, and the display unit
        (``"kW"`` for kWh / GJ input, ``"W"`` for Wh input).

    Raises
    ------
    KeyError
        ``col_name`` is in neither the data model nor ``overrides``.
    ValueError
        Resolved unit is not an accepted energy unit, or ``interval``
        is unparseable / non-positive.
    """
    unit = resolve_unit(col_name, overrides=overrides)
    if unit not in {"kWh", "Wh", "GJ"}:
        raise ValueError(
            f"convert_energy_per_interval_to_power: column {col_name!r} has "
            f"unit {unit!r}. Accepted input units: 'kWh' (-> 'kW'), "
            f"'Wh' (-> 'W'), 'GJ' (-> 'kW' via kWh)."
        )

    if isinstance(interval, pd.Timedelta):
        td = interval
    elif isinstance(interval, str):
        try:
            td = pd.Timedelta(interval)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"convert_energy_per_interval_to_power: interval {interval!r} "
                f"is not pd.Timedelta-parseable: {exc}"
            ) from None
    else:
        raise ValueError(
            f"convert_energy_per_interval_to_power: interval must be "
            f"pd.Timedelta or interval string; got {type(interval).__name__}"
        )

    hours = td.total_seconds() / 3600.0
    if hours <= 0:
        raise ValueError(
            f"convert_energy_per_interval_to_power: interval must be > 0; "
            f"got {td}"
        )

    if unit == "kWh":
        return series / hours, "kW"
    if unit == "Wh":
        return series / hours, "W"
    if unit == "GJ":
        # GJ -> kWh (* 1000 / 3.6) -> kW (/ hours)
        return series * (1000.0 / 3.6) / hours, "kW"
    raise AssertionError("unreachable")  # pragma: no cover


def infer_interval_from_readingdate(
    df: pd.DataFrame, col: str = "ReadingDate"
) -> str:
    """
    Infer the interval label ("5min", "60min", ...) from row spacing.

    Looks at the modal time delta between consecutive rows of ``df[col]``
    after sorting, then matches against the supported interval set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a datetime column.
    col : str, default "ReadingDate"
        Name of the datetime column.

    Returns
    -------
    str
        Interval label ("5min", "15min", ...).

    Raises
    ------
    ValueError
        If the column is missing, has fewer than 2 rows, or the inferred
        spacing does not match a supported interval.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not in df; cannot infer interval")
    sorted_times = df[col].dropna().sort_values()
    if len(sorted_times) < 2:
        raise ValueError(
            f"Need at least 2 rows in {col} to infer interval; "
            f"got {len(sorted_times)}"
        )
    delta = sorted_times.diff().dropna().mode().iloc[0]
    minutes = delta.total_seconds() / 60.0
    for label, hours in _INTERVAL_HOURS.items():
        if abs(minutes - hours * 60) < 0.01:
            return label
    raise ValueError(
        f"Inferred {minutes:.3f} minutes between rows; not a supported "
        f"interval. Supported: {sorted(_INTERVAL_HOURS)}"
    )


def convert_diff_to_power(
    series: pd.Series,
    col_name: str,
    interval: str,
    *,
    overrides: Optional[dict] = None,
) -> tuple[pd.Series, str]:
    """
    Convert per-row energy (kWh per ``interval`` row) to average power (kW).

    The input column's unit must be ``"kWh"`` -- looked up in the data
    model (or via ``overrides=``). The pre-data-model habit of passing a
    series whose unit the caller assumed to be kWh, without checking, is
    no longer allowed.

    Parameters
    ----------
    series : pd.Series
        Energy values, one per row, in kWh-per-``interval``.
    col_name : str
        Column name in the data model (or in ``overrides``) used to
        verify the input unit.
    interval : str
        Row interval label ("5min", "15min", ...).
    overrides : dict, optional
        Per-call ``{col: unit}`` for columns outside the data model.

    Returns
    -------
    (pd.Series, str)
        Average power in kW (= kWh per hour) and the display unit
        ``"kW"``. Pass the display unit to ``axis_label_with_unit``
        via ``overrides`` so plot labels stay in sync.

    Raises
    ------
    KeyError
        If ``col_name`` has no unit in the data model and no override.
    ValueError
        If the input unit is not ``"kWh"``.
    """
    input_unit = _resolve_input_unit(col_name, overrides)
    if input_unit != "kWh":
        raise ValueError(
            f"convert_diff_to_power expects input unit 'kWh' for "
            f"{col_name!r}; got {input_unit!r}. The series is not "
            f"per-row energy and the multiplier would produce nonsense."
        )
    return series * switch_multiplier(interval), "kW"


def mean_per_row_to_period_total(
    mean_value: float,
    col_name: str,
    interval: str,
    *,
    period_days: int = _DAYS_PER_YEAR,
    overrides: Optional[dict] = None,
) -> tuple[float, str]:
    """
    Scale a mean-per-row value to a period total in the variable's own unit.

    Use when the underlying variable is cumulative (sum-resampled in
    etdmodel.csv) and the report wants a per-household per-period figure
    (e.g. kWh per year, m3 per year) instead of the raw per-row mean.
    The per-row-to-per-period scaling preserves the unit -- only the
    time scope changes -- so the display unit equals the input unit.

    Parameters
    ----------
    mean_value : float
        Mean of the per-row values.
    col_name : str
        Column name in the data model (or in ``overrides``) used to
        record / verify the input unit.
    interval : str
        Row interval label ("5min", "15min", ...).
    period_days : int, default 365
        Number of days in the target period (365 for "per year",
        ~213 for "verwarmingsperiode" Oct-Apr, etc.).
    overrides : dict, optional
        Per-call ``{col: unit}`` for columns outside the data model.

    Returns
    -------
    (float, str)
        Period total in the variable's natural unit, and the unit string
        (e.g. ``("4500.0", "kWh")``).

    Raises
    ------
    KeyError
        If ``col_name`` has no unit in the data model and no override.
    """
    input_unit = _resolve_input_unit(col_name, overrides)
    rows_per_period = period_days * _HOURS_PER_DAY / interval_hours(interval)
    return mean_value * rows_per_period, input_unit


# Supported internal unit conversions inside to_per_m2.
# Extend deliberately -- every entry is a hand-verified relationship
# between two well-known SI prefixes of the same physical quantity.
_TO_PER_M2_RESCALES = {
    ("kW", "W"):   1000.0,
    ("W",  "kW"):  0.001,
    ("kWh", "Wh"): 1000.0,
    ("Wh", "kWh"): 0.001,
}


def to_per_m2(
    series: pd.Series,
    col_name: str,
    area: "pd.Series | float",
    *,
    target_unit: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> tuple[pd.Series, str]:
    """
    Divide a series by floor area, returning a per-m2 series and display unit.

    The input column's unit is looked up via the data model (or
    ``overrides=``); the output display unit is composed as
    ``f"{effective_unit} / m2"`` (with spaces, matching the etdmodel
    convention for composed units such as ``"liter / uur"``,
    ``"m3 / uur"``). When ``target_unit`` is given the helper applies a
    known internal conversion (currently kW <-> W, kWh <-> Wh) before
    dividing by area. The pre-data-model habit of multiplying by a raw
    scale factor whose meaning was known only to the caller is no longer
    allowed.

    Parameters
    ----------
    series : pd.Series
        Source values.
    col_name : str
        Column name in the data model (or in ``overrides``) used to
        verify the input unit.
    area : pd.Series or float
        Floor area in m**2. A Series is matched element-wise to ``series``;
        a scalar applies to every row. Must be > 0 and not NA.
    target_unit : str, optional
        Desired unit for the value before dividing by area. If omitted, the
        input unit is preserved (output = ``<input_unit> / m2``). Must be
        equal to the input unit or one of the supported internal rescales.
    overrides : dict, optional
        Per-call ``{col: unit}`` for columns outside the data model.

    Returns
    -------
    (pd.Series, str)
        Per-m**2 series and display unit (e.g. ``"W / m2"``, ``"kWh / m2"``).

    Raises
    ------
    KeyError
        If ``col_name`` has no unit in the data model and no override.
    ValueError
        If ``target_unit`` is not the input unit or a supported rescale,
        or if any area value is NA / non-positive.
    """
    input_unit = _resolve_input_unit(col_name, overrides)

    if target_unit is None or target_unit == input_unit:
        scaled = series
        effective_unit = input_unit
    else:
        key = (input_unit, target_unit)
        if key not in _TO_PER_M2_RESCALES:
            raise ValueError(
                f"to_per_m2: no supported conversion {input_unit!r} -> "
                f"{target_unit!r} for {col_name!r}. Supported: "
                f"{sorted(_TO_PER_M2_RESCALES)}."
            )
        scaled = series * _TO_PER_M2_RESCALES[key]
        effective_unit = target_unit

    if isinstance(area, (int, float)):
        if pd.isna(area) or area <= 0:
            raise ValueError(f"Area must be > 0; got {area!r}")
    else:
        if area.isna().any():
            raise ValueError(
                f"Area series contains {int(area.isna().sum())} NA value(s); "
                "per-household area must be set"
            )
        if (area <= 0).any():
            raise ValueError(
                f"Area series contains {int((area <= 0).sum())} non-positive "
                "value(s); per-household area must be > 0"
            )

    return scaled / area, f"{effective_unit} / m2"

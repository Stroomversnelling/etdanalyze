# etdanalyze Decision Records

Decisions specific to this repo. Decisions that apply to multiple repos live
in the parent project's `DECISIONS.md`.

---

## ADR-001: Units are first-class citizens in etdanalyze

Every numeric value flowing through etdanalyze -- plot labels, return
values from helper functions, intermediate computed columns -- MUST carry
an explicit unit. Silent unit propagation (caller and function disagreeing
about whether values are kWh or kW) has produced wrong plots and wrong
analysis output in this project's history. Notable past cases:

- A load-duration-curve helper labelled its y-axis "Power (kWh)" by
  looking up the column's data-model unit, while the plotted values had
  already been multiplied to kW. Reader saw kWh, axis values were kW.
- A per-m2 normalisation accepted a `scale=1000` raw multiplier whose
  meaning ("kW -> W") was known only to the caller; passing a series
  that wasn't actually in kW silently produced unphysical W/m2 numbers.
- A Diff-to-power conversion divided by interval without checking that
  the input was per-row energy; calling it on an already-power column
  silently produced numbers ~12x too small.

The single source of truth for units is the etdmap data model
(`Eenheid` column on etdmodel.csv, exposed via
`etdmap.data_model.load_unit_map`). Going forward:

### Rules

1. **Every label is unit-suffixed via the canonical label builder.** All
   axis labels, titles, legend entries, and colorbar labels in plot
   helpers go through `etdanalyze.plot_functions.axis_label_with_unit`.
   The builder looks the unit up in the etdmap data model and raises
   `KeyError` on unknown columns. There is no silent bare-label
   fallback.

2. **Per-call overrides are how the caller adds units the data model
   does not have.** When a label or conversion concerns a derived
   metric, a synthetic column, or a display-unit variant produced by an
   in-helper conversion, the caller passes
   `overrides={col_name: unit}`. The overrides dict beats the data
   model.

3. **An empty-string override is the explicit unitless flag.** Use
   `overrides={col: ""}` to declare a value genuinely has no unit
   (COP, ratios, dimensionless factors). The label builder returns the
   bare base label in that case (no parentheses). An absent override on
   a column the data model also does not know is an error, not
   "probably unitless" -- a missing unit is something the caller has to
   resolve, not something the helper guesses at.

4. **Conversion helpers verify expected input units before doing
   anything else.** Every function that transforms values from one unit
   to another (`convert_diff_to_power`, `to_per_m2`,
   `mean_per_row_to_period_total`, future additions) takes the source
   column name (or accepts a per-call override) and asserts the
   column's unit matches what the conversion expects. A mismatch raises
   `ValueError` at the entry of the helper, before any multiplication.
   Pre-data-model "trust the caller" semantics are forbidden.

5. **A helper that produces a converted value returns the display
   unit alongside it.** Signature pattern:
   `def convert(...) -> tuple[<value>, str]`. The caller hands the
   returned display unit straight to `axis_label_with_unit` via
   `overrides`, keeping the displayed value and its label in sync at
   the point of conversion. The two never drift.

6. **When a helper can accept more than one input unit, generalise the
   helper -- do not split it.** If a per-area conversion can naturally
   accept kW or W, the helper accepts either, checks the input unit at
   the start, and applies the necessary internal rescale (kW -> W or
   identity) before dividing by area. Internal unit conversion inside a
   helper is appropriate; that is the point of a helper. What is
   forbidden is two near-duplicate helpers (`from_kwh`, `from_wh`) that
   each silently assume one input unit. Generalise; do not duplicate.

### Implementation references

- `etdanalyze.plot_functions.axis_label_with_unit(col, base_label=None,
  *, overrides=None)` -- the canonical label builder.
- `etdanalyze.units._resolve_input_unit(col_name, overrides)` -- the
  canonical input-unit resolver. Lookup order: overrides -> data model
  -> raise `KeyError`.
- `etdanalyze.units.convert_diff_to_power`,
  `mean_per_row_to_period_total`, `to_per_m2` -- conversion helpers
  following the policy: they assert input unit, perform internal
  conversion as needed, return `(value, display_unit)`.
- `etdanalyze.units._TO_PER_M2_RESCALES` -- the explicit table of
  supported internal conversions (currently kW <-> W, kWh <-> Wh).
  Extend deliberately; every entry is a hand-verified relationship
  between two well-known SI prefixes of the same physical quantity.

### Scope

Applies to all code in `etdanalyze` and to any analysis script that
imports its helpers (including `etdanalyze-internal` and `etdworkflow`
callers). When migrating older callers, the loud failures surfaced by
these helpers are the intended outcome -- they identify the silent-trust
bugs that the policy exists to prevent. Callers MUST update to the new
signatures; there is no compatibility shim, by design.

---

## ADR-002: Plot helpers raise on empty data; callers own coverage and titles

Plot helpers in `etdanalyze.plot_functions` are producers of analytical
artifacts in the sense of parent ADR-021. They assume the caller has
filtered the input to the data that the plot is intended to render.
When that assumption is violated they raise immediately with a factual
description of what was received. The caller is responsible for
checking coverage upstream (via `classify_coverage` or equivalent),
for skipping plot calls that would receive nothing, for composing the
title string in full (including any "projects with data" annotation),
and for surfacing partial coverage in surrounding prose. The helper
draws what it is given; it does not pretend, infer, or hide absences.

### Rules

1. **Every plot helper that consumes a numeric column or filters
   internally raises `ValueError` on empty input.** Empty means: zero
   non-NA values in the column the plot reads (or, when the helper
   accepts `project_id=`, zero non-NA values within the filtered
   project). Silent `return None`, blank PNG output, and empty axes
   are forbidden. The strict raise applies regardless of whether the
   helper has an internal filter (`project_id=N`) or simply reads the
   passed-in frame.

2. **The error message reports facts only** -- per ADR-021. Required:
   the helper name, the failing column(s), and the filter the caller
   asserted (e.g. `project_id=8`). Forbidden: speculation about why
   the data is missing, prescription of a specific fix. Permitted:
   a short pointer to the contractual remedy ("check coverage upstream
   and skip the call"), without claiming where the missing data should
   have come from.

3. **The helper never enforces a "complete set" of groups.** When a
   plot iterates groups (one box per project, one line per WoningType),
   it renders only the groups present in the post-NA input. Legends,
   x-axis ticks, and colour mappings are built from the groups that
   actually have data. Absent groups are not flagged as errors -- they
   simply do not appear. The helper has no canonical list of
   "expected" projects; that knowledge belongs to the caller.

4. **The caller composes the title string in full.** Plot helpers
   accept `title=` and render it verbatim. They do not append
   "[projects 1, 4]" suffixes, do not look up units to put in titles,
   and do not infer any title content from the input frame. When the
   caller wants to annotate a pooled-mode plot with the list of
   projects that contributed, the caller assembles that suffix from
   the coverage table and concatenates it onto the title string
   before passing it in.

5. **`classify_coverage` is the standard coverage producer.** Located
   in `etdanalyze.analysis_helpers`, it returns one row per (variable,
   group) describing `n_rows`, `n_nonnull`, `is_empty`, `is_constant`.
   It is the upstream pre-check that callers run once per analyse
   step. Analyse scripts write `coverage.parquet` alongside their
   stats parquet so the report layer can read it and annotate sections
   inline ("alleen project 1 heeft data voor X"). Coverage is a
   first-class output, not an implementation detail.

6. **Required vs absent is not a plot-helper concern.** Whether a
   variable *should* exist for a particular project is decided by the
   data model (`etdmodel.csv`) and the supplier mapping. The report's
   job is to show what is there. If a "should-exist but doesn't" check
   is needed for QA, it lives in a separate workflow (e.g.
   `etdworkflow.analyze_data_report._check_required_coverage`), not in
   the plot path.

### What a compliant guard looks like

```python
def plot_daily_profile(df, plot_var, project_id=None, ...):
    ...
    # Strict empty-data guard: no silent empty plots.
    if df[plot_var].notna().sum() == 0:
        raise ValueError(
            f"plot_daily_profile: no non-NA rows for plot_var={plot_var!r} "
            f"in the input. Check coverage upstream and skip the call."
        )
    if project_id is not None:
        proj_mask = df["ProjectIdBSV"] == project_id
        if df.loc[proj_mask, plot_var].notna().sum() == 0:
            raise ValueError(
                f"plot_daily_profile: no non-NA rows for plot_var={plot_var!r} "
                f"and project_id={project_id!r}. Check coverage and skip the "
                f"call for this project."
            )
    ...
```

Caller side (pooled-mode plot, with title composition):

```python
coverage_df = classify_coverage(df_year, profile_vars, group_cols=["ProjectIdBSV"])
coverage_df.to_parquet(cache_dir / "coverage.parquet", index=False)

for var in profile_vars:
    var_cov = coverage_df[coverage_df["variable"] == var]
    if var_cov.empty or var_cov["is_empty"].all():
        continue  # nothing to plot; skip upstream so helper guard never fires

    # Caller composes the full title string, including the optional
    # "projecten met data" suffix for pooled-mode plots where projects
    # are not visually separated.
    title_suffix = projects_with_data_label(coverage_df, var)
    title = f"Dagprofiel  -  {var}{title_suffix}"

    plot_daily_profile(df=df_year, plot_var=var, title=title, ...)
```

### Why

- **Silent empty plots corrupt review.** A blank PNG embedded in a
  markdown report renders as an unlabelled empty frame; readers cannot
  tell whether the variable is missing, the projects had no observations,
  or the pipeline failed. Each cause needs a different response. The
  helper's job is to refuse to draw nothing.
- **Helper-side title manipulation creates cross-repo coupling.** If
  the helper appended Dutch suffixes, English reports would inherit
  them and English callers could not opt out without parameter sprawl.
  Composition in the caller keeps language and convention local.
- **Coverage as a first-class artifact mirrors the data-model dependency
  chain (parent ADR-004).** `coverage.parquet` is computed once per
  analyse step, consumed by report assembly, and can be inspected
  independently. It is the same shape as `per_project.parquet` and
  `coverage_per_project.parquet` -- a flat, queryable artefact the
  report layer reasons about, not a hidden state inside plot helpers.
- **Strict failure surfaces upstream regressions.** An all-NA column
  in the stats cache should never reach a plot helper; if it does, the
  helper raises and points the operator at the coverage check. This
  catches stats-emission bugs the same way ADR-022's strict NA
  propagation catches reduction bugs.

### Scope

Applies to every helper in `etdanalyze.plot_functions` that consumes
data and produces a figure. Callers in `etdanalyze-internal/analysis`,
`etdworkflow`, and any future report path must adopt the
`classify_coverage` + skip-empty + caller-composed-title idiom. Existing
callers that wrap plot calls in `try/except` (e.g.
`etdworkflow/analyze_data_report.py`) continue to work -- the new
`ValueError` is just absorbed into their warning path -- but the
strict-pattern caller idiom is the preferred shape for new code.

---

"""Generic per-project analysis scaffold: output paths, household loading,
cache and Excel writes, and the ADR-005 ``run()`` return-dict shaping.

These are small, composable infrastructure pieces that an analyse script
calls in the order it wants. The runner does not wrap a whole section in a
single mega-function; each script stays a readable narrative of its compute
steps and uses these helpers for the boring parts (output dirs,
household-data load, cache + Excel writes, return-dict shaping).

Everything here is project-agnostic. Project identity (the ``project_id``
string) and any data-quality exclusion lists are passed in by the caller;
this module never hardcodes them. The two-stage analysis architecture and
the output layout (``<output_folder_path>/<project_id>/{cache,excel,plots}``)
are shared conventions across analyses, which is why this lives in the
shared package.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import ibis
import pandas as pd
import etdtransform
from etdtransform.load_data import get_household_table

# Built tables are reused for the rest of the session. Reading an interval is
# not free -- the store keeps one directory per household, so opening it scales
# with the number of households -- and an analysis run asks for the same
# interval from several sections. The tables are lazy and immutable, so callers
# filtering or selecting from a shared one cannot affect each other.
_BASE_TABLE_CACHE: dict = {}


def _base_household_table(interval: str, include_weather: bool) -> ibis.Table:
    """Return the session's table for this interval, reading it once."""
    key = (interval, include_weather, str(etdtransform.options.aggregate_folder_path))
    if key not in _BASE_TABLE_CACHE:
        _BASE_TABLE_CACHE[key] = get_household_table(
            interval, include_weather=include_weather
        )
    return _BASE_TABLE_CACHE[key]


def clear_household_table_cache() -> None:
    """Forget the tables read so far.

    Needed when the data on disk gains or loses households mid-session: the
    cached tables were built against the households present at the time, so
    new ones would not appear until they are rebuilt.
    """
    _BASE_TABLE_CACHE.clear()

# Standard identity columns produced by the index join. A loader that
# restricts to a variable list always keeps these (when present) so the
# downstream grouping/plotting code can find its keys.
DEFAULT_ID_COLUMNS: list[str] = [
    "HuisIdBSV",
    "ProjectIdBSV",
    "ReadingDate",
    "Dataleverancier",
    "WoningType",
]


def project_output_path(options: dict, project_id: str) -> Path:
    """Resolve ``options['output_folder_path'] / project_id``.

    Single source of truth for "where does this project's output live".
    Every artefact a project produces lands under
    ``<output_folder_path>/<project_id>/...`` so the outputs folder stays
    organised when multiple projects ship in parallel.
    """
    return Path(options["output_folder_path"]) / project_id


def project_plots_path(options: dict, project_id: str) -> Path:
    """Plots live under the project folder, not in a parallel parent
    ``plots/`` tree. Keeps one project = one root directory.
    """
    return project_output_path(options, project_id) / "plots"


@dataclass(frozen=True)
class SectionPaths:
    """Resolved output paths for one section. Each analyse script's
    ``run()`` starts by calling :func:`init_section` to get one of these.
    """

    cache_dir: Path
    excel_dir: Path
    plots_dir: Path
    stem: str


def init_section(options: dict, project_id: str, section_stem: str) -> SectionPaths:
    """Resolve cache / excel / plots directories for ``section_stem`` under
    ``project_id``'s outputs subfolder.

    Directories are **resolved, not created**. They are made lazily by
    whatever actually writes into them (:func:`write_cache_parquets`,
    :func:`write_excel_workbook`, and the plot helpers before saving), so a
    section that produces only a cache does not leave an empty ``plots/``
    behind, and a section that produces only plots does not leave an empty
    ``cache/``. Empty directories are not harmless: they read as "this output
    was generated and is empty" rather than "this section has no such output".

    This also does NOT wipe. Wiping is a separate, deliberate step
    (:func:`wipe_section`) the caller performs only when it is about to
    recompute, so a cache that is still valid is not destroyed on every run.
    Pair with :func:`cache_is_fresh` to decide.
    """
    output_path = project_output_path(options, project_id)
    plots_path = project_plots_path(options, project_id)

    cache_dir = output_path / "cache" / section_stem
    excel_dir = output_path / "excel"
    plots_dir = plots_path / section_stem

    return SectionPaths(
        cache_dir=cache_dir, excel_dir=excel_dir, plots_dir=plots_dir, stem=section_stem
    )


def wipe_section(paths: SectionPaths, *, cache: bool = False, plots: bool = False) -> None:
    """Wipe the cache and/or plots directory for a section, recursively.

    Called by an analyse script immediately before it recomputes, so a run
    that reuses a fresh cache (see :func:`cache_is_fresh`) leaves the cache
    untouched. A partial ``--vars`` / ``--periods`` run typically wipes the
    cache (it is about to overwrite it) but not the plots (it would destroy
    plots it will not regenerate).
    """
    if cache:
        _wipe_directory(paths.cache_dir)
    if plots:
        _wipe_directory(paths.plots_dir)


def cache_is_fresh(cache_dir: Path, names, dependencies) -> bool:
    """True iff every ``{name}.parquet`` in ``cache_dir`` exists and is at
    least as new as every existing path in ``dependencies``.

    ``dependencies`` are the inputs whose change should invalidate the cache:
    the analyse script and the etdanalyze modules that implement the
    calculation, plus the source data parquet(s). If any dependency is newer
    than the oldest cache file -- the calculation or the data changed -- the
    cache is stale and the caller should recompute. A missing expected cache
    file also means stale.
    """
    cache_paths = [Path(cache_dir) / f"{n}.parquet" for n in names]
    if not cache_paths or not all(p.exists() for p in cache_paths):
        return False
    dep_mtimes = [
        Path(d).stat().st_mtime for d in dependencies if d and Path(d).exists()
    ]
    if not dep_mtimes:
        return True
    oldest_cache = min(p.stat().st_mtime for p in cache_paths)
    return oldest_cache >= max(dep_mtimes)


def _wipe_directory(d: Path) -> None:
    """Remove every file and subdirectory in ``d``. Used to clear stale
    outputs at the top of a run (recursive so a section that organises its
    plots into subfolders is fully cleaned).
    """
    if not d.exists():
        return
    n = 0
    for entry in d.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        n += 1
    if n:
        print(f"  cleanup: wiped {n} stale entr(y/ies) in {d}")


def load_household_table(
    interval: str,
    *,
    variables: Sequence[str] | None = None,
    extra_id_cols: Sequence[str] | None = None,
    projects: Sequence | None = None,
    excluded_projects: Sequence[int] = (),
    excluded_households: Sequence[int] = (),
    include_weather: bool = False,
) -> ibis.Table:
    """Lazy ibis read of the household data for one ``interval``
    (e.g. ``"5min"`` / ``"15min"`` / ``"60min"``), joined with the index
    table and (optionally) weather. Returns an ibis Table; the caller adds
    filters and materialises when needed.

    Exclusions are applied at the data query so dropped rows never enter any
    cache, plot, or table. The exclusion *values* belong to the caller
    (data-quality decisions are project-specific); only the mechanism is
    generic. ``excluded_projects`` filters on ``ProjectIdBSV``,
    ``excluded_households`` on ``HuisIdBSV``.

    ``variables`` restricts to those columns plus the standard id columns
    (:data:`DEFAULT_ID_COLUMNS`, extended by ``extra_id_cols``). ``None``
    keeps every column the join produces. ``projects`` restricts to the
    given ``ProjectIdBSV`` values (CLI strings are coerced to int).
    """
    hh = _base_household_table(interval, include_weather)

    if excluded_projects:
        hh = hh.filter(~hh["ProjectIdBSV"].isin(list(excluded_projects)))
    if excluded_households:
        hh = hh.filter(~hh["HuisIdBSV"].isin(list(excluded_households)))

    if projects is not None:
        # CLI flags arrive as strings; coerce to int against the int-typed
        # ProjectIdBSV so a future index-dtype migration keeps working.
        coerced: list = []
        for v in projects:
            try:
                coerced.append(int(v))
            except (TypeError, ValueError):
                coerced.append(v)
        hh = hh.filter(hh["ProjectIdBSV"].isin(coerced))

    if variables is not None:
        id_cols = list(DEFAULT_ID_COLUMNS)
        if extra_id_cols:
            id_cols = id_cols + [c for c in extra_id_cols if c not in id_cols]
        keep = [c for c in id_cols if c in hh.columns] + [
            v for v in variables if v in hh.columns
        ]
        hh = hh.select(keep)
    return hh


def write_cache_parquets(
    cache_dir: Path, frames: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    """Write ``{name: df}`` as ``{name}.parquet`` under ``cache_dir``.

    Returns the dict of resolved paths (matches the ADR-005 ``run()``
    contract shape). The parquet cache is the authoritative output: it is
    written first and is the single source of truth that every renderer
    (Excel, plots) reads.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)  # created lazily: see init_section
    paths: dict[str, Path] = {}
    for name, df in frames.items():
        path = cache_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
    print(f"  cache written: {cache_dir}/")
    return paths


def write_excel_workbook(
    excel_dir: Path, stem: str, sheets: dict[str, pd.DataFrame]
) -> Path:
    """Write one workbook ``{stem}.xlsx`` with one sheet per ``(name, df)``
    pair. Sheet names are passed through unchanged.

    Excel (openpyxl) cannot represent timezone-aware datetimes; the
    canonical datetime dtype in this project is ``datetime64[ns, UTC]``. We
    strip the timezone for the Excel render only (the parquet caches keep
    the tz-aware dtype). The stripped values are still UTC clock-time, just
    labelled naive.
    """
    excel_dir.mkdir(parents=True, exist_ok=True)  # created lazily: see init_section
    excel_path = excel_dir / f"{stem}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            _strip_tz_for_excel(df).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  excel written: {excel_path}")
    return excel_path


def _strip_tz_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy of ``df`` with every tz-aware datetime column
    converted to tz-naive (clock-time preserved).

    The parquet caches keep the original tz-aware dtype; the strip is only
    for the Excel presentation layer.
    """
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def finalize(
    cache_paths: dict[str, Path],
    excel_paths: dict[str, Path] | Path | None,
    plot_paths: dict[str, Path] | None,
) -> dict:
    """Return the ``{"cache": ..., "excel": ..., "plots": ...}`` dict that
    ADR-005 declares as the analyse-script ``run()`` contract.
    """
    if isinstance(excel_paths, Path):
        excel_paths = {excel_paths.stem: excel_paths}
    return {
        "cache": cache_paths,
        "excel": excel_paths or {},
        "plots": plot_paths or {},
    }


def cleanup_project_outputs(
    options: dict, project_id: str, current_sections: Iterable[str]
) -> None:
    """Remove anything in the project's output area that does not belong to
    a current section. Fires on full-run paths only -- section runs rely on
    :func:`init_section`'s per-section wipe, which is local.

    Scope (all under ``project_output_path`` / ``project_plots_path``):
      - ``cache/<section>/`` -- subdirs whose name is not a current section
      - ``cache/`` -- stray files directly under cache root
      - ``plots/<section>/`` -- subdirs whose name is not a current section
      - ``plots/`` -- stray files directly under plots root
      - ``excel/{name}.xlsx`` -- files whose stem is not a current section
      - ``sections/{name}.{md,docx}`` -- files whose stem is not a current
        section

    The combined report and any reviewer-notes file at the project root are
    project-scoped, not section-scoped, and are intentionally left alone.

    The deletion list is printed so the operator can audit. No files are
    moved or backed up; full runs are expected to regenerate everything
    that mattered.
    """
    sections = set(current_sections)
    output_root = project_output_path(options, project_id).resolve()
    plots_root = project_plots_path(options, project_id).resolve()
    deleted: list[Path] = []

    def _drop_unknown_in_subdir(parent: Path, allowed_stems: set[str]) -> None:
        if not parent.exists():
            return
        for entry in parent.iterdir():
            if entry.is_dir():
                if entry.name not in allowed_stems:
                    shutil.rmtree(entry)
                    deleted.append(entry)
            elif entry.is_file():
                if entry.stem not in allowed_stems:
                    entry.unlink()
                    deleted.append(entry)

    _drop_unknown_in_subdir(output_root / "cache", sections)
    _drop_unknown_in_subdir(plots_root, sections)
    _drop_unknown_in_subdir(output_root / "excel", sections)
    _drop_unknown_in_subdir(output_root / "sections", sections)

    if deleted:
        print(f"  cleanup: removed {len(deleted)} stale entry/entries:")
        for f in deleted[:20]:
            try:
                rel = f.relative_to(output_root.parent)
            except ValueError:
                rel = f
            print(f"    - {rel}")
        if len(deleted) > 20:
            print(f"    ... and {len(deleted) - 20} more")
    else:
        print(
            "  cleanup: nothing stale in scope "
            "(every entry matched a current section)"
        )

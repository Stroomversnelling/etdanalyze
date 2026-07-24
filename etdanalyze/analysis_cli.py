"""Shared CLI-flag plumbing for analyse scripts.

Each analyse script loads config and then calls ``run(options)``. This
module lets every script accept the same set of iteration-speed flags
without copy-pasting argparse code:

  --vars VAR [VAR ...]      Restrict to these variable names. Default = all.
  --periods PERIOD [PERIOD ...]
                            Restrict to these periods (jaar /
                            verwarmingsperiode / koelingsperiode). Default
                            = all.
  --skip-plots              Do not regenerate PNGs (cache + Excel still
                            built). Useful when only the stats logic changed.
  --skip-excel              Do not (re)write the Excel workbooks. Useful for
                            a fast smoke loop; combine with --skip-plots to
                            refresh only the parquet cache.
  --no-seasonal-profiles    Suppress the verwarmingsperiode / koelings-
                            periode profile-plot variants.
  --projects ID [ID ...]    Restrict per-project plot loops to these
                            project IDs.

Excel, PNG and DOCX are all renderers over the parquet cache (the single
source of truth). Each gets its own ``--skip-<renderer>`` flag; they share
one consistent family rather than a deeper abstraction.

Defaults preserve full-run behaviour: every flag absent -> full run.

Usage in an analyse script::

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="...")
        add_common_flags(parser)
        args = parser.parse_args()
        options = init_config()
        merge_cli_args_into_options(options, args)
        run(options)

And inside ``run(options)``::

    vars_filter     = options.get("vars_filter")      # None or list[str]
    periods_filter  = options.get("periods_filter")   # None or list[str]
    skip_plots      = options.get("skip_plots", False)
    skip_excel      = options.get("skip_excel", False)
    projects_filter = options.get("projects_filter")  # None or list
"""
from __future__ import annotations

import argparse


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Add the shared flags to a per-script parser."""
    parser.add_argument(
        "--vars",
        "--variables",
        dest="vars_filter",
        nargs="+",
        metavar="VAR",
        help="Restrict the analyse step to these variable names (default: all).",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        metavar="PERIOD",
        choices=["jaar", "verwarmingsperiode", "koelingsperiode"],
        help=(
            "Restrict to these periods (default: all). Choices: "
            "jaar, verwarmingsperiode, koelingsperiode."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the cache even if it is fresh (overrides cache_is_fresh).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation entirely (cache + Excel still written).",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel workbook generation (parquet cache still written).",
    )
    parser.add_argument(
        "--no-seasonal-profiles",
        action="store_true",
        help=(
            "Skip the verwarmingsperiode / koelingsperiode profile-plot "
            "variants. Default off; turn on for fast iteration."
        ),
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        dest="projects_filter",
        metavar="ID",
        help="Restrict per-project plot loops to these project IDs.",
    )


def merge_cli_args_into_options(options: dict, args: argparse.Namespace) -> dict:
    """Copy CLI flag values into the options dict consumed by ``run()``.

    Mutates and returns ``options`` for convenience. Uses ``getattr`` with
    defaults so a parser that did not register a given flag still works.
    """
    options["vars_filter"] = getattr(args, "vars_filter", None)
    options["periods_filter"] = getattr(args, "periods", None)
    options["force"] = bool(getattr(args, "force", False))
    options["skip_plots"] = bool(getattr(args, "skip_plots", False))
    options["skip_excel"] = bool(getattr(args, "skip_excel", False))
    options["no_seasonal_profiles"] = bool(getattr(args, "no_seasonal_profiles", False))
    options["projects_filter"] = getattr(args, "projects_filter", None)
    return options


def apply_vars_filter(vars_list: list[str], vars_filter: list[str] | None) -> list[str]:
    """Intersect a script's variable list with the caller's ``--vars`` filter.

    Returns ``vars_list`` unchanged when ``vars_filter`` is None. When a
    filter is supplied the result preserves the original order of
    ``vars_list``. A variable named in ``vars_filter`` but absent from
    ``vars_list`` is silently dropped -- the caller may be running across
    multiple scripts with one CLI invocation; per-script intersection is
    the right shape.
    """
    if vars_filter is None:
        return vars_list
    keep = set(vars_filter)
    return [v for v in vars_list if v in keep]


def apply_periods_filter(periods, periods_filter: list[str] | None):
    """Intersect a script's PERIODS structure with the ``--periods`` filter.

    Accepts both the dict form (``{"jaar": None, "verwarmingsperiode":
    [10, 11, ...], ...}``) and a tuple / list of period names. Returns the
    same shape with non-matching periods removed. Returns the input
    unchanged when filter is None.
    """
    if periods_filter is None:
        return periods
    keep = set(periods_filter)
    if isinstance(periods, dict):
        return {k: v for k, v in periods.items() if k in keep}
    return [p for p in periods if p in keep]


def apply_projects_filter(projects, projects_filter: list[str] | None):
    """Intersect a list of project IDs with the ``--projects`` filter.

    Project IDs in caches are commonly ``int``; CLI strings are coerced here
    so callers do not have to. Filter values that do not match any project
    are silently dropped.
    """
    if projects_filter is None:
        return list(projects)
    keep: set = set()
    for v in projects_filter:
        keep.add(v)
        try:
            keep.add(int(v))
        except (TypeError, ValueError):
            pass
    return [p for p in projects if p in keep]

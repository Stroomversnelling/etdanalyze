"""Registry of common grouping columns -> human-readable Dutch labels.

Plot helpers compose titles and y-axis labels that describe the grain
of the data being plotted. Historically the boxwhisker helper baked
the phrase "per-huishouden" into its default templates, which silently
mislabelled charts grouped by anything other than ``HuisIdBSV``
(weerstation, project, mode, woningtype, ...). The registry below
maps each known grouping column to a Dutch phrase the templates use
via the ``{per_grouping}`` placeholder; unknown columns fall back to
``"per <col>"`` so the label is at least honest about the unit.

Add an entry whenever a new grouping column starts appearing in
analyse scripts -- the helper's default templates then describe the
chart correctly without per-caller overrides.
"""

from typing import Optional


GROUPING_LABELS: dict[str, dict[str, str]] = {
    "HuisIdBSV":        {"singular": "huishouden",      "per_phrase": "per huishouden"},
    "ProjectIdBSV":     {"singular": "project",         "per_phrase": "per project"},
    "WoningType":       {"singular": "woningtype",      "per_phrase": "per woningtype"},
    "WeerstationGroep": {"singular": "weerstation",     "per_phrase": "per weerstation"},
    "Dataleverancier":  {"singular": "dataleverancier", "per_phrase": "per dataleverancier"},
    "Mode":             {"singular": "modus",           "per_phrase": "per modus"},
}


def label_for_grouping(
    col: Optional[str],
    *,
    fallback_singular: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"singular", "per_phrase"}`` for a grouping column.

    Always returns a dict with both keys populated. Unknown columns
    fall back to ``"per <col>"`` (or ``"per <fallback_singular>"``
    when supplied) so callers always have a label to render even when
    the registry doesn't know the column.
    """
    if col and col in GROUPING_LABELS:
        return dict(GROUPING_LABELS[col])
    singular = fallback_singular or (col or "rij")
    return {"singular": singular, "per_phrase": f"per {singular}"}

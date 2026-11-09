"""Shared test configuration for etdanalyze."""


# ---------------------------------------------------------------------------
# No skipped tests (ADR-007). A skip hides a missing prerequisite or an
# unmet condition as "not run", which reads as green when it is not. This
# hook turns ANY skip -- pytest.skip(), skip/skipif markers, importorskip --
# into a hard failure at runtime.
#
# THE RULE: if a test is there, you may not write skipping in code. Make the
# prerequisite available (build the fixture/artifact it needs), or let the
# test fail so the gap is visible. Do not add pytest.skip.
# ---------------------------------------------------------------------------
import pytest as _pytest

# Only these classes of skip may exist. A skip is allowed when its reason
# begins with "<TOKEN>: " and TOKEN is a key below.
#
# The allowlist is EMPTY on purpose. No skip in this code base has yet been
# shown to be legitimate -- every one examined turned out to be a missing
# prerequisite that should fail loudly instead. Empty means every skip is a
# hard failure until someone argues a class onto the list IN REVIEW. Adding a
# key is a deliberate, visible decision; deciding in the moment is not.
ALLOWED_SKIP_CLASSES = {
    # "PLATFORM": "behaviour genuinely differs by OS and is covered elsewhere",
    # "OPTIONAL_DEP": "an optional third-party package is not installed",
}


def _skip_reason(report):
    """
    The reason text of a skip.

    report.longrepr for a skip is the tuple (path, lineno, "Skipped: <reason>")
    for both a skipif marker and an in-body pytest.skip(). Verified against
    pytest directly rather than assumed.
    """
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        text = str(longrepr[2])
    else:
        text = str(longrepr)
    prefix = "Skipped: "
    return text[len(prefix):] if text.startswith(prefix) else text


@_pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return
    reason = _skip_reason(report)
    token = reason.split(":", 1)[0].strip() if ":" in reason else ""
    if token in ALLOWED_SKIP_CLASSES:
        return
    allowed = sorted(ALLOWED_SKIP_CLASSES) or ["(none -- the allowlist is empty)"]
    report.outcome = "failed"
    report.longrepr = (
        "SKIP NOT ALLOWED (ADR-007).\n"
        "  reason given: %r\n"
        "  skip class  : %r\n"
        "  allowed     : %s\n"
        "If a test is there, it must run. Provide the prerequisite it needs, or\n"
        "let it fail so the gap is visible. If a skip is genuinely warranted,\n"
        "add its class to ALLOWED_SKIP_CLASSES in review -- never silently."
        % (reason, token, allowed)
    )

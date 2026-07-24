"""Tests for etdanalyze.analysis_runner pure helpers.

Covers the tz-strip for Excel, write_excel_workbook round-trip, and the
finalize() return-dict shaping. The data-loading helpers (ibis/parquet) are
exercised by the analysis scripts, not unit-tested here.
"""
import pandas as pd

from etdanalyze.analysis_runner import (
    _strip_tz_for_excel,
    finalize,
    write_cache_parquets,
    write_excel_workbook,
)


def _tz_frame():
    return pd.DataFrame(
        {
            "ReadingDate": pd.to_datetime(
                ["2021-06-01 10:00", "2021-06-01 11:00"], utc=True
            ),
            "value": pd.array([1.0, 2.0], dtype="Float64"),
        }
    )


class TestStripTz:
    def test_strips_tz_preserving_clock(self):
        out = _strip_tz_for_excel(_tz_frame())
        assert out["ReadingDate"].dt.tz is None
        assert str(out["ReadingDate"].iloc[0]) == "2021-06-01 10:00:00"

    def test_leaves_non_datetime_untouched(self):
        out = _strip_tz_for_excel(_tz_frame())
        assert out["value"].dtype == "Float64"

    def test_does_not_mutate_input(self):
        df = _tz_frame()
        _strip_tz_for_excel(df)
        assert df["ReadingDate"].dt.tz is not None


class TestWriteExcel:
    def test_workbook_written_and_readable(self, tmp_path):
        path = write_excel_workbook(
            tmp_path, "demo", {"Sheet1": _tz_frame()}
        )
        assert path.exists()
        back = pd.read_excel(path, sheet_name="Sheet1")
        assert list(back.columns) == ["ReadingDate", "value"]
        assert len(back) == 2


class TestCacheAndFinalize:
    def test_write_cache_parquets(self, tmp_path):
        paths = write_cache_parquets(tmp_path, {"t": _tz_frame()})
        assert paths["t"].exists()
        back = pd.read_parquet(paths["t"], dtype_backend="numpy_nullable")
        # tz-aware dtype preserved in the cache (unlike Excel).
        assert back["ReadingDate"].dt.tz is not None

    def test_finalize_shapes_dict(self, tmp_path):
        out = finalize({"c": tmp_path}, None, None)
        assert out == {"cache": {"c": tmp_path}, "excel": {}, "plots": {}}

    def test_finalize_wraps_single_excel_path(self, tmp_path):
        xl = tmp_path / "demo.xlsx"
        out = finalize({}, xl, None)
        assert out["excel"] == {"demo": xl}

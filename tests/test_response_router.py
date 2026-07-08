"""回覆格式決策的單元測試。

測試 fallback 規則引擎的決策邏輯。
"""

import pytest
from agent.nodes.response_router import _fallback_decision, _build_result_summary


class TestFallbackDecision:
    """測試 _fallback_decision 規則引擎。"""

    def test_empty_result(self):
        """空結果應回傳 text 格式。"""
        fmt, chart = _fallback_decision({"intent_type": "aggregate"}, [])
        assert fmt == "text"
        assert chart is None

    def test_single_value_aggregate(self):
        """單一數值的聚合結果應回傳 text。"""
        result = [{"COUNT(*)": 1000}]
        fmt, chart = _fallback_decision({"intent_type": "aggregate"}, result)
        assert fmt == "text"
        assert chart is None

    def test_detail_query(self):
        """明細查詢應回傳 table。"""
        result = [{"id": i, "name": f"item_{i}"} for i in range(10)]
        fmt, chart = _fallback_decision({"intent_type": "detail"}, result)
        assert fmt == "table"
        assert chart is None

    def test_trend_query(self):
        """趨勢查詢應回傳 chart + line。"""
        result = [{"month": f"2019-0{i}", "sales": 1000 * i} for i in range(1, 4)]
        fmt, chart = _fallback_decision({"intent_type": "trend"}, result)
        assert fmt == "chart"
        assert chart == "line"

    def test_comparison_few_items_pie(self):
        """少於等於 6 項的比較應回傳 pie。"""
        result = [{"branch": f"B{i}", "total": 100 * i} for i in range(1, 4)]
        fmt, chart = _fallback_decision({"intent_type": "comparison"}, result)
        assert fmt == "chart"
        assert chart == "pie"

    def test_comparison_many_items_bar(self):
        """超過 6 項的比較應回傳 bar。"""
        result = [{"product": f"P{i}", "total": 100 * i} for i in range(1, 10)]
        fmt, chart = _fallback_decision({"intent_type": "comparison"}, result)
        assert fmt == "chart"
        assert chart == "bar"

    def test_unknown_intent_few_results(self):
        """未知意圖 + 少量結果應回傳 text。"""
        result = [{"x": 1}, {"x": 2}]
        fmt, chart = _fallback_decision({"intent_type": "unknown"}, result)
        assert fmt == "text"
        assert chart is None

    def test_unknown_intent_many_results(self):
        """未知意圖 + 大量結果應回傳 table。"""
        result = [{"x": i} for i in range(10)]
        fmt, chart = _fallback_decision({"intent_type": "unknown"}, result)
        assert fmt == "table"
        assert chart is None


class TestBuildResultSummary:
    """測試 _build_result_summary 摘要生成。"""

    def test_empty_result(self):
        """空結果應回傳無結果文字。"""
        summary = _build_result_summary([])
        assert "0 筆" in summary

    def test_single_row(self):
        """單筆結果摘要應包含筆數和欄位。"""
        result = [{"COUNT(*)": 1000}]
        summary = _build_result_summary(result)
        assert "1" in summary
        assert "COUNT(*)" in summary

    def test_multiple_rows(self):
        """多筆結果摘要應顯示前 3 筆。"""
        result = [{"branch": f"B{i}", "sales": 100 * i} for i in range(5)]
        summary = _build_result_summary(result)
        assert "5" in summary
        assert "branch" in summary
        assert "前 3" in summary

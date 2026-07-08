"""圖表生成模組的單元測試。

測試欄位辨識、標題生成邏輯。
"""

import pytest
from visualization.chart_generator import (
    _identify_columns,
    _generate_title,
)


class TestIdentifyColumns:
    """測試 _identify_columns 欄位辨識邏輯。"""

    def test_string_and_number(self):
        """有字串欄位和數值欄位時應正確辨識。"""
        data = [{"branch": "A", "total_sales": 10000.5}]
        columns = ["branch", "total_sales"]
        label, value = _identify_columns(data, columns)
        assert label == "branch"
        assert value == "total_sales"

    def test_two_numeric_columns(self):
        """兩個數值欄位時，第一個作為 label，第二個作為 value。"""
        data = [{"month": 1, "sales": 5000}]
        columns = ["month", "sales"]
        label, value = _identify_columns(data, columns)
        assert label == "month"
        assert value == "sales"

    def test_single_column(self):
        """只有一個欄位時，label 為 None。"""
        data = [{"count": 1000}]
        columns = ["count"]
        label, value = _identify_columns(data, columns)
        assert label is None
        assert value == "count"


class TestGenerateTitle:
    """測試 _generate_title 標題生成。"""

    def test_with_intent_description(self):
        """有 intent description 時應使用它作為標題。"""
        intent = {"description": "各分店的銷售額比較"}
        title = _generate_title(intent, "branch", "sales")
        assert title == "各分店的銷售額比較"

    def test_without_intent(self):
        """無 intent 時應用欄位名稱組合。"""
        title = _generate_title(None, "branch", "sales")
        assert "sales" in title
        assert "branch" in title

    def test_empty_intent(self):
        """空 intent 時應用欄位名稱組合。"""
        title = _generate_title({}, "product_line", "revenue")
        assert "revenue" in title

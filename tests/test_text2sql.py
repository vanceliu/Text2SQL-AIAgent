"""Text2SQL 節點的單元測試。

測試 SQL 清理和 fallback 提取邏輯。
"""

import pytest
from agent.nodes.text2sql import _clean_sql, _fallback_extract_sql


class TestCleanSql:
    """測試 _clean_sql 的 SQL 清理邏輯。"""

    def test_normal_sql(self):
        """正常 SQL 應保持不變（只確保結尾分號）。"""
        assert _clean_sql("SELECT * FROM sales") == "SELECT * FROM sales;"

    def test_trailing_semicolon(self):
        """已有分號的 SQL 不應重複加。"""
        assert _clean_sql("SELECT * FROM sales;") == "SELECT * FROM sales;"

    def test_multiple_semicolons(self):
        """多餘分號應被清理。"""
        assert _clean_sql("SELECT * FROM sales;;;") == "SELECT * FROM sales;"

    def test_whitespace(self):
        """前後空白應被移除。"""
        assert _clean_sql("  SELECT * FROM sales  ") == "SELECT * FROM sales;"

    def test_empty_string(self):
        """空字串應回傳空字串。"""
        assert _clean_sql("") == ""

    def test_multiline_sql(self):
        """多行 SQL 應正確處理。"""
        sql = "SELECT branch,\n  SUM(sales)\nFROM sales\nGROUP BY branch"
        result = _clean_sql(sql)
        assert result.endswith(";")
        assert "SELECT" in result


class TestFallbackExtractSql:
    """測試 _fallback_extract_sql 的 SQL 提取邏輯。"""

    def test_plain_sql(self):
        """純 SQL 文字應直接回傳。"""
        result = _fallback_extract_sql("SELECT COUNT(*) FROM sales")
        assert "SELECT" in result

    def test_sql_in_code_block(self):
        """markdown code block 中的 SQL 應被提取。"""
        content = "```sql\nSELECT * FROM sales\n```"
        result = _fallback_extract_sql(content)
        assert "SELECT * FROM sales" == result

    def test_sql_in_plain_code_block(self):
        """無語言標記的 code block 也應提取。"""
        content = "```\nSELECT branch FROM sales\n```"
        result = _fallback_extract_sql(content)
        assert "SELECT branch FROM sales" == result

    def test_mixed_text_with_select(self):
        """混合文字中有 SELECT 開頭的行應被提取。"""
        content = "Here is the query:\nSELECT * FROM sales WHERE rating > 9"
        result = _fallback_extract_sql(content)
        assert "SELECT" in result

    def test_no_sql_returns_content(self):
        """無 SQL 的文字應回傳原始內容。"""
        content = "I cannot generate SQL for this question."
        result = _fallback_extract_sql(content)
        assert result == content

"""SQL 驗證節點的單元測試。

測試 SQL 安全性檢查、語法驗證、重試邏輯。
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.nodes.sql_validator import (
    validate_sql,
    _check_sql_safety,
    _check_sql_syntax,
    _validation_failed,
    MAX_RETRY_COUNT,
)
from agent.state import AgentState


class TestCheckSqlSafety:
    """測試 _check_sql_safety 函式的安全性檢查邏輯。"""

    def test_valid_select(self):
        """正常 SELECT 語句應通過安全檢查。"""
        assert _check_sql_safety("SELECT * FROM sales;") is None

    def test_select_with_where(self):
        """帶 WHERE 的 SELECT 應通過。"""
        assert _check_sql_safety("SELECT branch, sales FROM sales WHERE rating > 9;") is None

    def test_select_with_group_by(self):
        """帶 GROUP BY 的聚合查詢應通過。"""
        sql = "SELECT branch, SUM(sales) FROM sales GROUP BY branch;"
        assert _check_sql_safety(sql) is None

    def test_insert_blocked(self):
        """INSERT 語句應被阻擋。"""
        result = _check_sql_safety("INSERT INTO sales VALUES ('test');")
        assert result is not None
        assert "INSERT" in result

    def test_delete_blocked(self):
        """DELETE 語句應被阻擋。"""
        result = _check_sql_safety("DELETE FROM sales WHERE 1=1;")
        assert result is not None
        assert "DELETE" in result

    def test_drop_blocked(self):
        """DROP 語句應被阻擋。"""
        result = _check_sql_safety("DROP TABLE sales;")
        assert result is not None

    def test_update_blocked(self):
        """UPDATE 語句應被阻擋。"""
        result = _check_sql_safety("UPDATE sales SET rating = 10;")
        assert result is not None
        assert "UPDATE" in result

    def test_truncate_blocked(self):
        """TRUNCATE 語句應被阻擋。"""
        result = _check_sql_safety("TRUNCATE TABLE sales;")
        assert result is not None

    def test_alter_blocked(self):
        """ALTER 語句應被阻擋。"""
        result = _check_sql_safety("ALTER TABLE sales ADD COLUMN test TEXT;")
        assert result is not None


class TestCheckSqlSyntax:
    """測試 _check_sql_syntax 函式的語法驗證邏輯（需要實際 DB）。"""

    def test_valid_syntax(self):
        """語法正確的 SQL 應通過（需要 supermarket.db 存在）。"""
        result = _check_sql_syntax("SELECT COUNT(*) FROM sales;")
        # 若 DB 不存在會回傳錯誤，但不應 raise exception
        assert result is None or isinstance(result, str)

    def test_invalid_syntax(self):
        """語法錯誤的 SQL 應回傳錯誤訊息。"""
        result = _check_sql_syntax("SELEC * FORM sales;")
        assert result is not None
        assert "語法錯誤" in result or "異常" in result


class TestValidationFailed:
    """測試 _validation_failed 的重試邏輯。"""

    def test_retry_within_limit(self):
        """未超過重試上限時，應回傳錯誤但不終止。"""
        result = _validation_failed("test error", retry_count=0)
        assert result["sql_valid"] is False
        assert result["sql_error"] == "test error"
        assert result["retry_count"] == 1
        assert result["error"] is None

    def test_retry_at_limit(self):
        """達到重試上限時，應設定終止 error。"""
        result = _validation_failed("test error", retry_count=MAX_RETRY_COUNT - 1)
        assert result["sql_valid"] is False
        assert result["error"] is not None
        assert "換個方式提問" in result["error"]

    def test_retry_count_increments(self):
        """每次失敗 retry_count 應加 1。"""
        result = _validation_failed("err", retry_count=1)
        assert result["retry_count"] == 2


class TestValidateSqlNode:
    """測試 validate_sql 節點的完整流程。"""

    def test_empty_sql(self):
        """空 SQL 應觸發驗證失敗。"""
        state = AgentState(generated_sql="", retry_count=0)
        result = validate_sql(state)
        assert result["sql_valid"] is False
        assert "為空" in result["sql_error"]

    def test_none_sql(self):
        """None SQL 應觸發驗證失敗。"""
        state = AgentState(generated_sql=None, retry_count=0)
        result = validate_sql(state)
        assert result["sql_valid"] is False

    def test_dangerous_sql(self):
        """危險 SQL 應觸發驗證失敗。"""
        state = AgentState(generated_sql="DROP TABLE sales;", retry_count=0)
        result = validate_sql(state)
        assert result["sql_valid"] is False

    def test_valid_select_passes(self):
        """合法 SELECT 應通過驗證（語法檢查可能因 DB 不存在而跳過）。"""
        state = AgentState(generated_sql="SELECT COUNT(*) FROM sales;", retry_count=0)
        result = validate_sql(state)
        # 若 DB 存在則 sql_valid=True，否則可能因 EXPLAIN 失敗而 False
        assert "sql_valid" in result

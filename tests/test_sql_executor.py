"""SQL 執行節點的單元測試。

測試 execute_sql 節點的查詢執行功能。
使用臨時 SQLite 資料庫進行測試。
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.state import AgentState
from agent.nodes.sql_executor import execute_sql


@pytest.fixture
def temp_db_with_data():
    """建立包含測試資料的臨時 SQLite 資料庫。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE sales (
            invoice_id TEXT,
            branch TEXT,
            city TEXT,
            unit_price REAL,
            quantity INTEGER,
            sales REAL,
            rating REAL
        )
    """)
    test_data = [
        ("INV-001", "A", "Yangon", 74.69, 7, 548.97, 9.1),
        ("INV-002", "B", "Mandalay", 15.28, 5, 80.22, 9.6),
        ("INV-003", "A", "Yangon", 46.33, 3, 145.94, 7.2),
    ]
    conn.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?)", test_data
    )
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class TestExecuteSql:
    """測試 execute_sql 節點。"""

    def test_simple_count(self, temp_db_with_data):
        """簡單 COUNT 查詢應回傳正確結果。"""
        state = AgentState(generated_sql="SELECT COUNT(*) as cnt FROM sales;")
        with patch("agent.nodes.sql_executor.DEFAULT_DATABASE_PATH", temp_db_with_data):
            result = execute_sql(state)
        assert result["query_result"] == [{"cnt": 3}]

    def test_select_with_where(self, temp_db_with_data):
        """帶 WHERE 條件的查詢應正確過濾。"""
        state = AgentState(generated_sql="SELECT branch FROM sales WHERE rating > 9;")
        with patch("agent.nodes.sql_executor.DEFAULT_DATABASE_PATH", temp_db_with_data):
            result = execute_sql(state)
        assert len(result["query_result"]) == 2

    def test_aggregate_query(self, temp_db_with_data):
        """聚合查詢應回傳正確結果。"""
        state = AgentState(generated_sql="SELECT branch, SUM(sales) as total FROM sales GROUP BY branch;")
        with patch("agent.nodes.sql_executor.DEFAULT_DATABASE_PATH", temp_db_with_data):
            result = execute_sql(state)
        assert len(result["query_result"]) == 2
        branches = [r["branch"] for r in result["query_result"]]
        assert "A" in branches
        assert "B" in branches

    def test_invalid_sql_returns_error(self, temp_db_with_data):
        """無效 SQL 應回傳 error。"""
        state = AgentState(generated_sql="SELEC * FORM sales;")
        with patch("agent.nodes.sql_executor.DEFAULT_DATABASE_PATH", temp_db_with_data):
            result = execute_sql(state)
        assert result["query_result"] is None
        assert "error" in result

    def test_result_format_is_list_of_dict(self, temp_db_with_data):
        """結果格式應為 list of dict。"""
        state = AgentState(generated_sql="SELECT * FROM sales LIMIT 2;")
        with patch("agent.nodes.sql_executor.DEFAULT_DATABASE_PATH", temp_db_with_data):
            result = execute_sql(state)
        assert isinstance(result["query_result"], list)
        assert all(isinstance(row, dict) for row in result["query_result"])
        assert "invoice_id" in result["query_result"][0]

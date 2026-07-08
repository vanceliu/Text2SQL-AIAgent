"""Schema 讀取模組的單元測試。

測試 get_table_schema 和 get_column_names 功能。
需要 supermarket.db 存在才能完整執行。
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from db.schema import get_table_schema, get_column_names


@pytest.fixture
def temp_db():
    """建立臨時測試資料庫，包含簡單的 sales 資料表。"""
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
            date TEXT,
            rating REAL
        )
    """)
    conn.execute("""
        INSERT INTO sales VALUES
        ('INV-001', 'A', 'Yangon', 74.69, 7, 548.97, '2019-01-05', 9.1)
    """)
    conn.execute("""
        INSERT INTO sales VALUES
        ('INV-002', 'B', 'Mandalay', 15.28, 5, 80.22, '2019-03-08', 9.6)
    """)
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class TestGetTableSchema:
    """測試 get_table_schema 的 Schema 讀取功能。"""

    def test_returns_string(self, temp_db):
        """應回傳格式化的字串。"""
        result = get_table_schema(db_path=temp_db, table_name="sales")
        assert isinstance(result, str)

    def test_contains_table_name(self, temp_db):
        """回傳結果應包含 table 名稱。"""
        result = get_table_schema(db_path=temp_db, table_name="sales")
        assert "sales" in result

    def test_contains_row_count(self, temp_db):
        """回傳結果應包含資料筆數。"""
        result = get_table_schema(db_path=temp_db, table_name="sales")
        assert "2" in result

    def test_contains_column_names(self, temp_db):
        """回傳結果應包含欄位名稱。"""
        result = get_table_schema(db_path=temp_db, table_name="sales")
        assert "invoice_id" in result
        assert "branch" in result
        assert "rating" in result

    def test_contains_example_values(self, temp_db):
        """回傳結果應包含範例值。"""
        result = get_table_schema(db_path=temp_db, table_name="sales")
        assert "Yangon" in result or "INV-001" in result

    def test_nonexistent_db_raises(self):
        """不存在的資料庫應拋出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            get_table_schema(db_path="/nonexistent/path.db")

    def test_nonexistent_table_raises(self, temp_db):
        """不存在的資料表應拋出 OperationalError。"""
        with pytest.raises(sqlite3.OperationalError):
            get_table_schema(db_path=temp_db, table_name="nonexistent_table")


class TestGetColumnNames:
    """測試 get_column_names 的欄位名稱讀取。"""

    def test_returns_list(self, temp_db):
        """應回傳字串列表。"""
        result = get_column_names(db_path=temp_db, table_name="sales")
        assert isinstance(result, list)
        assert all(isinstance(col, str) for col in result)

    def test_correct_column_count(self, temp_db):
        """應回傳正確的欄位數量。"""
        result = get_column_names(db_path=temp_db, table_name="sales")
        assert len(result) == 8

    def test_contains_expected_columns(self, temp_db):
        """應包含預期的欄位名稱。"""
        result = get_column_names(db_path=temp_db, table_name="sales")
        assert "invoice_id" in result
        assert "branch" in result
        assert "sales" in result

    def test_nonexistent_db_raises(self):
        """不存在的資料庫應拋出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            get_column_names(db_path="/nonexistent/path.db")

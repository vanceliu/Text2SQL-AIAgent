"""資料庫 Schema 讀取工具模組。

提供從 SQLite 資料庫讀取表結構資訊的功能，
供 LLM 生成 SQL 時作為上下文參考。
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent.parent
DEFAULT_DATABASE_PATH = str(project_root / os.getenv("DATABASE_PATH"))

def get_table_schema(db_path: Optional[str] = None, table_name: str = "sales") -> str:
    """讀取指定資料表的完整 Schema 資訊，格式化為 LLM 可理解的文字。

    Args:
        db_path: SQLite 資料庫檔案路徑。若為 None 則使用預設路徑。
        table_name: 要讀取 Schema 的資料表名稱，預設為 "sales"。

    Process:
        1. 連接 SQLite 資料庫
        2. 使用 PRAGMA table_info 取得欄位定義（名稱、型別、是否可為 NULL）
        3. 查詢每個欄位的前 5 筆不重複範例值
        4. 組合為結構化的文字描述

    Returns:
        str: 格式化的 Schema 描述文字，包含欄位名稱、型別、範例值，
             可直接嵌入 LLM prompt 中使用。

    Raises:
        FileNotFoundError: 資料庫檔案不存在。
        sqlite3.OperationalError: 資料表不存在或查詢失敗。
    """
    if db_path is None:
        db_path = DEFAULT_DATABASE_PATH

    if not Path(db_path).exists():
        raise FileNotFoundError(f"資料庫檔案不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        if not columns:
            raise sqlite3.OperationalError(f"資料表 '{table_name}' 不存在或無欄位")

        row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        schema_parts = []
        schema_parts.append(f"Table: {table_name}")
        schema_parts.append(f"Total rows: {row_count}")
        schema_parts.append(f"Columns ({len(columns)}):")
        schema_parts.append("-" * 60)

        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col

            try:
                cursor.execute(
                    f"SELECT DISTINCT [{col_name}] FROM {table_name} "
                    f"WHERE [{col_name}] IS NOT NULL LIMIT 5"
                )
                examples = [str(row[0]) for row in cursor.fetchall()]
                example_str = ", ".join(examples)
            except Exception:
                example_str = "(無法取得範例)"

            schema_parts.append(
                f"  - {col_name} ({col_type or 'TEXT'})"
                f"    Examples: {example_str}"
            )

        return "\n".join(schema_parts)

    finally:
        conn.close()


def get_column_names(db_path: Optional[str] = None, table_name: str = "sales") -> list[str]:
    """取得指定資料表的所有欄位名稱列表。

    Args:
        db_path: SQLite 資料庫檔案路徑。若為 None 則使用預設路徑。
        table_name: 資料表名稱，預設為 "sales"。

    Process:
        1. 連接資料庫
        2. 使用 PRAGMA table_info 取得欄位清單

    Returns:
        list[str]: 欄位名稱列表，例如 ["invoice_id", "branch", "city", ...]。

    Raises:
        FileNotFoundError: 資料庫檔案不存在。
    """
    if db_path is None:
        db_path = DEFAULT_DATABASE_PATH

    if not Path(db_path).exists():
        raise FileNotFoundError(f"資料庫檔案不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [col[1] for col in cursor.fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        schema = get_table_schema()
        print(schema)
    except FileNotFoundError as e:
        print(f"[錯誤] {e}")

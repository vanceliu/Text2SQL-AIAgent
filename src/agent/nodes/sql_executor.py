"""SQL 執行節點模組。

安全地在 SQLite 資料庫上執行已驗證通過的 SQL 查詢，
並回傳結構化的查詢結果。
"""

import sqlite3
from typing import Any

from agent.state import AgentState
from db.schema import DEFAULT_DATABASE_PATH


MAX_RESULT_ROWS = 100


def execute_sql(state: AgentState) -> dict:
    """在 SQLite 資料庫上執行已驗證的 SQL 查詢並回傳結果。

    Args:
        state: Agent 當前狀態，需要讀取：
               - state.generated_sql: 已通過驗證的 SQL 查詢語句

    Process:
        1. 從 state 取得已驗證的 SQL
        2. 連接 SQLite 資料庫
        3. 設定查詢逾時保護（避免惡意或低效查詢佔用過多資源）
        4. 執行 SQL 查詢
        5. 取得欄位名稱，將結果轉為 list of dict 格式
        6. 限制最大回傳筆數（MAX_RESULT_ROWS）
        7. 處理執行異常（逾時、資料庫錯誤等）

    Returns:
        dict: 包含以下鍵的字典：
              - "query_result": list[dict]，查詢結果，每筆為一個字典
              若執行失敗則回傳：
              - "error": str，錯誤描述訊息
              - "query_result": None
    """
    sql = state.generated_sql
    db_path = DEFAULT_DATABASE_PATH

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        cursor = conn.cursor()

        clean_sql = sql.rstrip(";")
        cursor.execute(clean_sql)

        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(MAX_RESULT_ROWS)

        result = [dict(zip(columns, row)) for row in rows]

        conn.close()

        return {"query_result": result}

    except sqlite3.OperationalError as e:
        return {
            "error": f"SQL 執行失敗: {str(e)}",
            "query_result": None,
        }
    except Exception as e:
        return {
            "error": f"查詢過程發生異常: {str(e)}",
            "query_result": None,
        }

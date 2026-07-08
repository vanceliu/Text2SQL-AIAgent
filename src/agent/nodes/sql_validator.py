"""SQL 驗證/修正節點模組。

驗證 LLM 生成的 SQL 語法是否正確、是否僅包含安全的 SELECT 操作。
驗證失敗時回饋錯誤訊息供重新生成。
"""

import sqlite3
import sqlparse

from agent.state import AgentState
from db.schema import DEFAULT_DATABASE_PATH


DANGEROUS_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "TRUNCATE", "REPLACE", "MERGE",
    "EXEC", "EXECUTE", "GRANT", "REVOKE",
]

MAX_RETRY_COUNT = 3


def validate_sql(state: AgentState) -> dict:
    """驗證生成的 SQL 語法正確性與安全性。

    Args:
        state: Agent 當前狀態，需要讀取：
               - state.generated_sql: 待驗證的 SQL 語句
               - state.retry_count: 當前重試次數

    Process:
        1. 檢查 SQL 是否為空
        2. 使用 sqlparse 解析 SQL，檢查語句類型是否為 SELECT
        3. 掃描是否包含危險關鍵字（INSERT、DROP 等）
        4. 使用 SQLite EXPLAIN 驗證語法可執行性
        5. 若驗證通過，標記 sql_valid = True
        6. 若驗證失敗且未超過重試上限，標記錯誤並增加 retry_count
        7. 若超過重試上限，設定 error 訊息終止流程

    Returns:
        dict: 包含以下鍵的字典：
              - "sql_valid": bool，驗證是否通過
              - "sql_error": str 或 None，失敗時的錯誤描述
              - "retry_count": int，更新後的重試計數
              - "error": str 或 None，超過重試上限時的終止訊息
    """
    sql = state.generated_sql
    retry_count = state.retry_count

    if not sql or not sql.strip():
        return _validation_failed("SQL 為空", retry_count)

    safety_error = _check_sql_safety(sql)
    if safety_error:
        return _validation_failed(safety_error, retry_count)

    syntax_error = _check_sql_syntax(sql)
    if syntax_error:
        return _validation_failed(syntax_error, retry_count)

    return {
        "sql_valid": True,
        "sql_error": None,
        "retry_count": retry_count,
    }


def _check_sql_safety(sql: str) -> str | None:
    """檢查 SQL 是否包含危險操作關鍵字。

    Args:
        sql: 待檢查的 SQL 語句。

    Process:
        1. 使用 sqlparse 解析 SQL 語句
        2. 檢查語句類型是否為 SELECT（或 UNKNOWN 因為子查詢等情況）
        3. 逐一比對危險關鍵字清單

    Returns:
        str | None: 若不安全則回傳錯誤描述，安全則回傳 None。
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return "無法解析 SQL 語句"

    stmt = parsed[0]
    stmt_type = stmt.get_type()

    if stmt_type and stmt_type.upper() not in ("SELECT", "UNKNOWN"):
        return f"不允許的 SQL 類型: {stmt_type}（僅允許 SELECT）"

    sql_upper = sql.upper()
    for keyword in DANGEROUS_KEYWORDS:
        tokens = sql_upper.split()
        if keyword in tokens:
            return f"偵測到危險操作關鍵字: {keyword}（僅允許 SELECT 查詢）"

    return None


def _check_sql_syntax(sql: str) -> str | None:
    """使用 SQLite EXPLAIN 驗證 SQL 語法可執行性。

    Args:
        sql: 待驗證的 SQL 語句。

    Process:
        1. 連接 SQLite 資料庫
        2. 使用 EXPLAIN 指令測試 SQL 是否可被解析執行
        3. 捕獲 OperationalError 作為語法錯誤

    Returns:
        str | None: 若語法錯誤則回傳錯誤描述，正確則回傳 None。
    """
    db_path = DEFAULT_DATABASE_PATH
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        clean_sql = sql.rstrip(";")
        cursor.execute(f"EXPLAIN {clean_sql}")
        conn.close()
        return None
    except sqlite3.OperationalError as e:
        return f"SQL 語法錯誤: {str(e)}"
    except Exception as e:
        return f"驗證過程異常: {str(e)}"


def _validation_failed(error_msg: str, retry_count: int) -> dict:
    """處理驗證失敗的情況，判斷是否超過重試上限。

    Args:
        error_msg: 驗證失敗的錯誤描述。
        retry_count: 當前已重試的次數。

    Process:
        1. 增加 retry_count
        2. 若超過 MAX_RETRY_COUNT，設定終止 error
        3. 否則回傳錯誤訊息供下一次重試

    Returns:
        dict: 包含 sql_valid、sql_error、retry_count、error 的字典。
    """
    new_retry_count = retry_count + 1

    if new_retry_count >= MAX_RETRY_COUNT:
        return {
            "sql_valid": False,
            "sql_error": error_msg,
            "retry_count": new_retry_count,
            "error": "抱歉，我無法理解您的問題或生成正確的查詢。請換個方式提問。",
        }

    return {
        "sql_valid": False,
        "sql_error": error_msg,
        "retry_count": new_retry_count,
        "error": None,
    }

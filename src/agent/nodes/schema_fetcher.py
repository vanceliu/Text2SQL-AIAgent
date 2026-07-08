"""Schema 擷取節點模組。

從 SQLite 資料庫自動讀取資料表 Schema 資訊，
組成 prompt 的一部分供 LLM 生成 SQL 時參考。
"""

from agent.state import AgentState
from db.schema import get_table_schema, DEFAULT_DATABASE_PATH


def fetch_schema(state: AgentState) -> dict:
    """讀取資料庫 Schema 資訊並存入 Agent State。

    Args:
        state: Agent 當前狀態。此節點不需讀取 state 中的特定欄位，
               僅負責產出 schema_info。

    Process:
        1. 取得資料庫路徑（使用預設路徑）
        2. 呼叫 get_table_schema() 讀取 sales 資料表的完整結構
        3. 結構包含：表名、總筆數、每個欄位的名稱/型別/範例值
        4. 若讀取失敗，將錯誤訊息存入 error 欄位

    Returns:
        dict: 包含 "schema_info" 鍵的字典，值為格式化的 Schema 文字描述。
              若失敗則回傳包含 "error" 鍵的字典。
    """
    try:
        db_path = DEFAULT_DATABASE_PATH
        schema_info = get_table_schema(db_path=db_path, table_name="sales")
        return {"schema_info": schema_info}
    except (FileNotFoundError, Exception) as e:
        return {"error": f"無法讀取資料庫 Schema: {str(e)}"}

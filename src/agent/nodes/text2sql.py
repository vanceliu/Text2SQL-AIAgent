"""SQL 生成節點模組。

將使用者意圖結合 Schema 資訊，透過 Function Calling 讓 LLM 生成對應的 SQL 查詢語句。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from agent.llm import get_llm
from pydantic import BaseModel, Field

TEXT2SQL_SYSTEM_PROMPT = """你是一個專業的 SQL 生成助手。根據使用者的查詢意圖和資料庫 Schema，
生成正確的 SQLite SQL 查詢語句。

請呼叫 generate_sql tool 來回傳生成的 SQL。

重要規則：
1. 僅生成 SELECT 語句，嚴禁 INSERT、UPDATE、DELETE、DROP、ALTER 等任何修改操作
2. 注意日期欄位格式為 'YYYY-MM-DD'（如 '2019-01-05'）
3. 確保 SQL 語法正確，可在 SQLite 上直接執行
4. 若需要聚合，使用適當的 GROUP BY
5. 若查詢明細，適當加上 LIMIT 限制筆數（預設最多 50 筆）
6. 數值計算結果請用 ROUND() 保留 2 位小數
7. table名稱為 sales
"""

class SQLGenerationResult(BaseModel):
    """SQL 生成結果的結構，作為 function calling 的回傳 schema。"""
    sql_query: str = Field(description="生成的 SELECT SQL 查詢語句，僅允許 SELECT 操作")
    explanation: str = Field(description="簡短說明這個 SQL 查詢的邏輯")

def generate_sql(state: AgentState) -> dict:
    """透過 Function Calling，根據使用者意圖和 Schema 資訊生成 SQL 查詢語句。

    Args:
        state: Agent 當前狀態，需要讀取：
               - state.intent: 意圖解析結果（查詢類型、實體）
               - state.schema_info: 資料庫 Schema 資訊
               - state.sql_error: 若為重試，包含前一次的錯誤訊息
               - state.generated_sql: 若為重試，包含前一次生成的 SQL
               - state.messages: 對話歷史（取最新一筆使用者訊息）

    Process:
        1. 建立 LLM 實例並綁定 SQLGenerationResult 作為 tool schema
        2. 組合 prompt：Schema 資訊 + 意圖描述 + 使用者原始問題
        3. 若為重試（有 sql_error），在 prompt 中附加錯誤訊息要求修正
        4. LLM 透過 function calling 回傳 SQL 語句
        5. 從 tool_calls 中提取 SQL
        6. 若 LLM 未正確呼叫 tool，嘗試從回應文字提取 SQL 作為 fallback

    Returns:
        dict: 包含以下鍵的字典：
              - "generated_sql": 生成的 SQL 查詢語句字串
              - "sql_valid": 重置為 None（待驗證節點確認）
              - "sql_error": 重置為 None
    """
    llm = get_llm()
    llm_with_tool = llm.bind_tools([SQLGenerationResult], tool_choice="any")

    intent = state.intent or {}
    schema_info = state.schema_info or ""

    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    user_question = user_messages[-1].content if user_messages else ""

    user_prompt_parts = [
        f"## 資料庫 Schema\n{schema_info}",
        f"\n## 使用者問題\n{user_question}",
        f"\n## 意圖分析\n類型: {intent.get('intent_type', 'unknown')}",
        f"描述: {intent.get('description', '')}",
        f"實體: {json.dumps(intent.get('entities', {}), ensure_ascii=False)}",
    ]

    if state.sql_error and state.generated_sql:
        user_prompt_parts.append(
            f"\n## 前一次 SQL 生成失敗\n"
            f"失敗的 SQL: {state.generated_sql}\n"
            f"錯誤訊息: {state.sql_error}\n"
            f"請修正上述錯誤，重新生成正確的 SQL。"
        )

    messages = [
        SystemMessage(content=TEXT2SQL_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(user_prompt_parts)),
    ]

    response = llm_with_tool.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]
        sql = args.get("sql_query", "")
    else:
        sql = _fallback_extract_sql(response.content)

    sql = _clean_sql(sql)

    return {
        "generated_sql": sql,
        "sql_valid": None,
        "sql_error": None,
    }


def _fallback_extract_sql(content: str) -> str:
    """當 LLM 未正確呼叫 tool 時，嘗試從文字回應提取 SQL。

    Args:
        content: LLM 的文字回應內容。

    Process:
        1. 檢查是否有 markdown code block，提取其中內容
        2. 否則尋找 SELECT 開頭的語句
        3. 若都找不到，回傳原始內容

    Returns:
        str: 提取出的 SQL 語句。
    """
    cleaned = content.strip()

    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            sql_block = parts[1]
            if sql_block.startswith("sql"):
                sql_block = sql_block[3:]
            return sql_block.strip()

    for line in cleaned.split("\n"):
        if line.strip().upper().startswith("SELECT"):
            return line.strip()

    return cleaned


def _clean_sql(sql: str) -> str:
    """清理 SQL 語句，確保格式正確。

    Args:
        sql: 原始 SQL 語句。

    Process:
        1. 去除前後空白
        2. 確保以分號結尾
        3. 移除多餘的分號

    Returns:
        str: 清理後的 SQL 語句。
    """
    sql = sql.strip()
    sql = sql.rstrip(";").strip()
    if sql:
        sql += ";"
    return sql

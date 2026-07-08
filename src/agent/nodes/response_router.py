"""回覆格式決策節點模組。

透過 Function Calling，根據使用者問題的意圖類型與查詢結果的資料特徵，
自主決策最適合的回覆格式（口語/列表/圖表）。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from agent.llm import get_llm
from pydantic import BaseModel, Field
from typing import Optional


ROUTER_SYSTEM_PROMPT = """你是一個回覆格式決策模組。根據使用者的查詢意圖和查詢結果，
決定最適合的回覆呈現方式。

請呼叫 decide_format tool 來回傳決策結果。

回覆格式有三種：
1. "text" - 口語回覆：適合單一數值結果、簡單答案
2. "table" - 列表/表格：適合多筆明細結果（≤20 筆）
3. "chart" - 圖表：適合趨勢分析、比較分析、佔比分析

圖表類型：
- "bar" - 長條圖：比較不同類別的數值
- "line" - 折線圖：顯示時間趨勢變化
- "pie" - 圓餅圖：顯示佔比分佈

決策邏輯：
- 使用者問「總共」「平均」「最高」等 → 結果是單一數值 → text
- 使用者問「列出」「哪些」「有哪些」 → 結果是多筆紀錄 → table
- 使用者問「趨勢」「變化」「按月」 → 時序資料 → chart (line)
- 使用者問「比較」「各XX的」「分佈」 → 分類統計 → chart (bar 或 pie)
- 使用者問「佔比」「比例」 → 佔比資料 → chart (pie)
- 結果筆數 = 1 且只有一個數值欄位 → text
- 結果筆數 2-20 → table 或 chart（看是否適合視覺化）
- 結果筆數 > 20 → table（截斷顯示）
"""

class ResponseFormatResult(BaseModel):
    """回覆格式決策結果的結構，作為 function calling 的回傳 schema。"""
    response_format: str = Field(
        description="回覆格式：text(口語回覆), table(列表/表格), chart(圖表)"
    )
    chart_type: Optional[str] = Field(
        None,
        description="圖表類型（僅 response_format=chart 時需要）：bar(長條圖), line(折線圖), pie(圓餅圖)"
    )
    reason: str = Field(description="簡短說明決策理由")


def route_response(state: AgentState) -> dict:
    """透過 Function Calling，根據意圖和查詢結果決定回覆格式。

    Args:
        state: Agent 當前狀態，需要讀取：
               - state.intent: 意圖解析結果
               - state.query_result: SQL 查詢結果（list of dict）
               - state.messages: 對話歷史（取最新使用者訊息）

    Process:
        1. 建立 LLM 實例並綁定 ResponseFormatResult 作為 tool schema
        2. 組合決策資訊：意圖類型 + 查詢結果摘要（筆數、欄位、前幾筆資料）
        3. LLM 透過 function calling 回傳格式決策
        4. 從 tool_calls 中提取決策結果
        5. 若 LLM 未正確呼叫 tool，使用規則引擎進行 fallback 決策

    Returns:
        dict: 包含以下鍵的字典：
              - "response_format": str，"text" | "table" | "chart"
              - "chart_type": str 或 None，"bar" | "line" | "pie" | None
    """
    llm = get_llm()
    llm_with_tool = llm.bind_tools([ResponseFormatResult], tool_choice="any")

    intent = state.intent or {}
    result = state.query_result or []

    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    user_question = user_messages[-1].content if user_messages else ""

    result_summary = _build_result_summary(result)

    user_prompt = (
        f"## 使用者問題\n{user_question}\n\n"
        f"## 意圖分析\n"
        f"類型: {intent.get('intent_type', 'unknown')}\n"
        f"描述: {intent.get('description', '')}\n\n"
        f"## 查詢結果摘要\n{result_summary}"
    )

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm_with_tool.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]
        response_format = args.get("response_format", "text")
        chart_type = args.get("chart_type")
    else:
        response_format, chart_type = _fallback_decision(intent, result)

    if response_format not in ("text", "table", "chart"):
        response_format = "text"
    if response_format == "chart" and chart_type not in ("bar", "line", "pie"):
        chart_type = "bar"
    if response_format != "chart":
        chart_type = None

    return {
        "response_format": response_format,
        "chart_type": chart_type,
    }


def _build_result_summary(result: list[dict]) -> str:
    """建立查詢結果的摘要文字，供 LLM 做格式決策參考。

    Args:
        result: SQL 查詢結果，list of dict 格式。

    Process:
        1. 計算結果筆數
        2. 取得欄位名稱列表
        3. 取前 3 筆作為範例
        4. 組合為摘要文字

    Returns:
        str: 結果摘要，包含筆數、欄位名稱、前幾筆範例資料。
    """
    if not result:
        return "查詢無結果（0 筆）"

    row_count = len(result)
    columns = list(result[0].keys())
    sample = result[:3]

    summary_parts = [
        f"筆數: {row_count}",
        f"欄位: {', '.join(columns)}",
        f"前 {min(3, row_count)} 筆資料:",
    ]
    for i, row in enumerate(sample, 1):
        summary_parts.append(f"  {i}. {json.dumps(row, ensure_ascii=False)}")

    return "\n".join(summary_parts)


def _fallback_decision(intent: dict, result: list[dict]) -> tuple[str, str | None]:
    """當 LLM 未正確呼叫 tool 時，使用規則引擎做 fallback 決策。

    Args:
        intent: 意圖解析結果字典。
        result: SQL 查詢結果 list of dict。

    Process:
        1. 根據 intent_type 和結果筆數做基本判斷
        2. aggregate → text
        3. detail → table
        4. trend → chart (line)
        5. comparison → chart (bar 或 pie)

    Returns:
        tuple[str, str | None]: (response_format, chart_type) 的元組。
    """
    intent_type = intent.get("intent_type", "aggregate")
    row_count = len(result) if result else 0

    if row_count == 0:
        return "text", None
    if row_count == 1 and len(result[0]) <= 2:
        return "text", None

    if intent_type == "aggregate":
        return "text", None
    elif intent_type == "detail":
        return "table", None
    elif intent_type == "trend":
        return "chart", "line"
    elif intent_type == "comparison":
        if row_count <= 6:
            return "chart", "pie"
        return "chart", "bar"
    else:
        if row_count <= 5:
            return "text", None
        return "table", None

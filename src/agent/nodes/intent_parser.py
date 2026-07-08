"""意圖理解節點模組。

分析使用者的自然語言輸入，透過 Function Calling 判斷查詢意圖類型，
並提取相關實體（日期、分店、產品線等）。
包含對話歷史摘要壓縮機制，避免 context 過大。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from agent.llm import get_llm
from pydantic import BaseModel, Field
from typing import Optional, Literal

MAX_HISTORY_TOKENS = 6000

INTENT_SYSTEM_PROMPT = """你是一個超市銷售數據查詢助手的意圖分析模組。
你的任務是分析使用者的自然語言問題，判斷查詢意圖並提取關鍵實體。

請呼叫 parse_intent tool 來回傳分析結果。

意圖類型說明：
- aggregate: 聚合統計（總共、平均、最大、最小等），通常回傳單一數值或少量結果
- detail: 明細查詢（列出、哪些、查看等），回傳多筆具體紀錄
- trend: 趨勢分析（趨勢、變化、按月/日等），適合用折線圖呈現
- comparison: 比較分析（比較、佔比、各XX的等），適合用長條圖或圓餅圖呈現
- error: 無法回答的問題（預測、建議、超出資料範圍等）

如果問題超出資料集能力（如預測未來、提供建議等），設 is_answerable 為 false，
intent_type 設為 "error"。

注意：使用者可能在對話中使用指代（如「那 Giza 呢？」），
此時請根據對話上下文推斷完整意圖。

"""

SUMMARIZE_PROMPT = """請將以下對話歷史壓縮為一段簡短摘要（300字以內），
保留關鍵資訊：使用者問了什麼、得到了什麼結果。
只輸出摘要文字，不要其他內容。

對話歷史：
"""


class IntentEntities(BaseModel):
    """意圖解析提取的實體結構。"""
    date_range: Optional[str] = Field(None, description="提及的日期範圍，如 '2019-01' 或 '2019-01 to 2019-03'")
    branch: Optional[str] = Field(None, description="提及的分店，如 'A', 'B', 'C'")
    city: Optional[str] = Field(None, description="提及的城市，如 'Yangon', 'Naypyitaw', 'Mandalay'")
    product_line: Optional[str] = Field(None, description="提及的產品線，如 'Health and beauty'")
    customer_type: Optional[str] = Field(None, description="提及的客戶類型，如 'Member', 'Normal'")
    gender: Optional[str] = Field(None, description="提及的性別，如 'Male', 'Female'")
    payment: Optional[str] = Field(None, description="提及的付款方式，如 'Ewallet', 'Cash', 'Credit card'")
    metric: Optional[str] = Field(None, description="想查詢的指標，如 'sales', 'rating', 'quantity'")


class IntentResult(BaseModel):
    """意圖解析結果的完整結構，作為 function calling 的回傳 schema。"""
    intent_type: Literal["aggregate", "detail", "trend", "comparison", "error"] = Field(
        description="查詢意圖類型：aggregate(聚合統計), detail(明細查詢), trend(趨勢分析), comparison(比較分析), error(無法回答)"
    )
    description: str = Field(description="簡短描述使用者想查什麼")
    entities: IntentEntities = Field(default_factory=IntentEntities, description="從問題中提取的關鍵實體")
    is_answerable: bool = Field(default=True, description="該問題是否能用現有資料集回答")


def parse_intent(state: AgentState) -> dict:
    """透過 Function Calling 分析使用者自然語言輸入，判斷查詢意圖與提取關鍵實體。

    Args:
        state: Agent 當前狀態，包含 messages 對話歷史。

    Process:
        1. 建立 LLM 實例並綁定 IntentResult 作為 tool schema
        2. 檢查對話歷史長度，超過門檻則壓縮舊歷史為摘要
        3. 組合 system prompt + 摘要（若有） + 最近對話送給 LLM
        4. LLM 透過 function calling 回傳結構化的意圖結果
        5. 從 tool_calls 中提取意圖資訊
        6. 若 LLM 未正確呼叫 tool，嘗試從回應文字解析 JSON 作為 fallback

    Returns:
        dict: 包含 "intent" 鍵的字典，值為意圖解析結果字典，
              包含 intent_type、description、entities、is_answerable 欄位。
              若解析失敗則回傳 error 類型的意圖。
    """
    llm = get_llm()
    llm_with_tool = llm.bind_tools([IntentResult], tool_choice="any")

    conversation_messages = _prepare_messages(state.messages, state.total_tokens_used, llm)

    messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT)]
    messages.extend(conversation_messages)

    response = llm_with_tool.invoke(messages)

    token_usage = response.response_metadata.get("token_usage", {})
    total_tokens = token_usage.get("total_tokens", 0)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        intent = tool_call["args"]
    else:
        intent = _fallback_parse(response.content)

    return {"intent": intent, "total_tokens_used": state.total_tokens_used + total_tokens}


def _prepare_messages(messages: list, total_tokens_used: int, llm) -> list:
    """準備送給 LLM 的對話歷史，超過 token 門檻時壓縮舊對話為摘要。

    Args:
        messages: 完整的對話歷史 messages 列表。
        total_tokens_used: 累計的歷史 token 數（由 LLM response_metadata 取得）。
        llm: LLM 實例，用於生成摘要。

    Process:
        1. 若累計 token 數未超過 MAX_HISTORY_TOKENS，直接回傳全部
        2. 若超過門檻，從最新的 messages 往回保留直到接近門檻的一半
        3. 將更早的 messages 壓縮為摘要
        4. 回傳 [摘要 SystemMessage] + 最近 messages

    Returns:
        list: 處理後的 messages 列表，可直接送給 LLM。
    """
    if total_tokens_used <= MAX_HISTORY_TOKENS:
        return list(messages)

    keep_budget = MAX_HISTORY_TOKENS // 2
    keep_tokens = 0
    split_index = len(messages)

    # 將較早的 message 壓縮為摘要，保留最近的 messages
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = _estimate_message_tokens(messages[i])
        if keep_tokens + msg_tokens > keep_budget:
            split_index = i + 1
            break
        keep_tokens += msg_tokens

    if split_index <= 0:
        split_index = 1

    old_messages = messages[:split_index]
    recent_messages = messages[split_index:]

    history_text = _messages_to_text(old_messages)
    summary = _summarize_history(history_text, llm)

    result = [SystemMessage(content=f"[對話歷史摘要] {summary}")]
    result.extend(recent_messages)
    return result


def _estimate_tokens(messages: list) -> int:
    """估算 messages 列表的總 token 數。

    Args:
        messages: messages 列表。

    Process:
        中文約 1.5 字 ≈ 1 token，英文約 4 字元 ≈ 1 token。
        使用簡單的字元數公式粗估。

    Returns:
        int: 估算的 token 數。
    """
    return sum(_estimate_message_tokens(msg) for msg in messages)


def _estimate_message_tokens(msg) -> int:
    """估算單則 message 的 token 數。

    Args:
        msg: 一則 message 物件。

    Process:
        取得 content 文字長度，用字元數 / 2 作為粗估（中英混合取平均）。

    Returns:
        int: 估算的 token 數。
    """
    content = msg.content if hasattr(msg, "content") else ""
    return max(len(content) // 2, 1)


def _messages_to_text(messages: list) -> str:
    """將 messages 列表轉為純文字，供摘要用。

    Args:
        messages: 要轉換的 messages 列表。

    Process:
        遍歷 messages，根據類型加上 "使用者:" 或 "助手:" 前綴，

    Returns:
        str: 格式化的對話歷史純文字。
    """
    parts = []
    for msg in messages:
        if hasattr(msg, "type"):
            role = "使用者" if msg.type == "human" else "助手"
        else:
            role = "系統"
        content = msg.content
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _summarize_history(history_text: str, llm) -> str:
    """呼叫 LLM 將對話歷史壓縮為簡短摘要。

    Args:
        history_text: 格式化的對話歷史純文字。
        llm: LLM 實例。

    Process:
        1. 組合摘要 prompt + 對話歷史
        2. 呼叫 LLM 生成摘要
        3. 若失敗則回傳截斷的原始文字作為 fallback

    Returns:
        str: 壓縮後的對話摘要文字。
    """
    try:
        response = llm.invoke([
            HumanMessage(content=SUMMARIZE_PROMPT + history_text)
        ])
        content = response.content
        assert isinstance(content, str)
        return content.strip()
    except Exception:
        return history_text[:300] + "..."


def _fallback_parse(content: str) -> dict:
    """當 LLM 未正確呼叫 tool 時，嘗試從文字回應解析意圖。

    Args:
        content: LLM 的文字回應內容。

    Process:
        1. 嘗試從回應中找到 JSON 格式內容
        2. 解析 JSON 取得意圖結構
        3. 若完全無法解析，回傳 error 類型的預設意圖

    Returns:
        dict: 意圖結果字典，包含 intent_type、description、entities、is_answerable。
    """
    try:
        cleaned = content.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "intent_type": "error",
            "description": "無法解析使用者意圖",
            "entities": {},
            "is_answerable": False,
        }

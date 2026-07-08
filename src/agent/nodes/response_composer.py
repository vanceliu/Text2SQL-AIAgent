"""最終回覆組合節點模組。

根據回覆格式決策結果，透過 Function Calling 組合口語化的文字回覆，
並在需要時生成表格或圖表。
使用 bind_tools 強制 LLM 將回覆內容放入 tool_calls，
避免 Gemma 4 在關閉 thinking mode 時將推理過程洩漏到 content 中。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.llm import get_llm
from visualization.chart_generator import generate_chart


COMPOSER_SYSTEM_PROMPT = """你是一個超市銷售數據查詢助手。

請根據提供的查詢結果，呼叫 compose_reply tool 回傳最終回覆。

回覆規則：
- 使用自然、口語化的繁體中文
- 不要重複使用者問題
- 回覆簡潔且包含重要數字
- 金額用千分位格式（如 106,200.37）
- 若結果有多筆，可簡短整理重點（如指出最大值、趨勢）
- 不提及 SQL、資料庫或查詢過程
- 不要輸出任何思考過程或分析步驟

回覆格式對應：
- response_format=text：只需口語化文字回覆
- response_format=table：先寫一段口語摘要，接著附上完整的 Markdown Table
- response_format=chart：只需文字摘要（圖表會由系統另外生成）
"""


class ComposerResult(BaseModel):
    """回覆組合結果的結構，作為 function calling 的回傳 schema。

    此 schema 強制 LLM 將最終回覆放入結構化的 tool_calls 中，
    而非自由生成到 content（避免 Gemma 4 thinking 外洩問題）。
    """
    response_text: str = Field(
        description=(
            "給使用者看的最終回覆文字。"
            "必須是繁體中文、口語化、自然流暢。"
            "若 response_format=table，需包含 Markdown 格式的表格。"
            "若 response_format=chart，只需文字摘要。"
            "金額數字使用千分位格式（如 106,200.37）。"
            "不要包含任何思考過程、分析步驟或英文推理內容。"
        )
    )


def compose_response(state: AgentState) -> dict:
    """透過 Function Calling 組合最終的口語化回覆。

    Args:
        state: Agent 當前狀態，需要讀取：
               - state.query_result: SQL 查詢結果（list of dict）
               - state.response_format: 回覆格式決策（text/table/chart）
               - state.chart_type: 圖表類型（bar/line/pie），僅 chart 時使用
               - state.intent: 意圖資訊，用於判斷是否可回答
               - state.messages: 對話歷史（取最新一筆使用者訊息）
               - state.error: 流程中的錯誤訊息（若有則直接回覆錯誤）

    Process:
        1. 若 state.error 有值，直接回傳錯誤訊息（不呼叫 LLM）
        2. 若意圖為不可回答，回傳預設的拒絕回覆（不呼叫 LLM）
        3. 建立 LLM 實例，綁定 ComposerResult schema，設定 tool_choice="any"
           強制模型必須呼叫 tool（推理過程不會出現在結果中）
        4. 組合 user prompt：使用者問題 + 回覆格式 + 查詢結果 JSON
        5. 呼叫 LLM，從 response.tool_calls[0]["args"]["response_text"] 取得回覆
        6. 若 FC 未觸發（fallback），從 content 中提取最後一段中文文字
        7. 若格式為 chart，呼叫 generate_chart 生成圖表並附加路徑
        8. 將最終回覆加入 messages（供 checkpointer 持久化，支援多輪對話）

    Returns:
        dict: 包含以下鍵的字典：
              - "final_response": str，最終回覆文字（直接顯示給使用者）
              - "chart_path": str 或 None，圖表檔案路徑（若有生成）
              - "messages": list[AIMessage]，新增的 AI 回覆（寫入 checkpointer）
    """
    # 錯誤處理：流程中有 error 時直接回覆，不再呼叫 LLM
    if state.error:
        error_response = state.error
        return {
            "final_response": error_response,
            "chart_path": None,
            "messages": [AIMessage(content=error_response)],
        }

    # 不可回答的問題：回傳預設拒絕訊息
    if not state.intent or not state.intent.get("is_answerable", True):
        unanswerable_response = (
            "抱歉，目前的資料集僅包含歷史交易紀錄，我無法進行預測分析。\n"
            "不過我可以協助您查看過去的銷售趨勢。您想了解哪方面的歷史數據呢？"
        )
        return {
            "final_response": unanswerable_response,
            "chart_path": None,
            "messages": [AIMessage(content=unanswerable_response)],
        }

    # 建立 LLM 並綁定 ComposerResult tool schema
    # tool_choice="any" 強制模型必須呼叫 tool，避免 thinking 外洩到 content
    llm = get_llm(temperature=0.3)
    llm_with_tool = llm.bind_tools([ComposerResult], tool_choice="any")

    result = state.query_result or []
    response_format = state.response_format or "text"

    # 取得最新一筆使用者訊息作為回覆的語境
    user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    user_question = user_messages[-1].content if user_messages else ""

    # 將查詢結果轉為 JSON 文字（最多 30 筆，避免 prompt 過大）
    result_text = json.dumps(result[:30], ensure_ascii=False, indent=2)

    user_prompt = (
        f"## 使用者問題\n{user_question}\n\n"
        f"## 回覆格式\n{response_format}\n\n"
        f"## 查詢結果（共 {len(result)} 筆）\n{result_text}"
    )

    messages = [
        SystemMessage(content=COMPOSER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm_with_tool.invoke(messages)

    # 從 tool_calls 取得結構化回覆（正常路徑）
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        final_response = tool_call["args"].get("response_text", "")
    else:
        # FC 未觸發時的 fallback：從 content 提取最後一段中文
        final_response = _fallback_extract(response.content)

    # 若格式為 chart，生成圖表檔案並附加路徑到回覆中
    chart_path = None
    if response_format == "chart" and result:
        chart_path = generate_chart(
            data=result,
            chart_type=state.chart_type or "bar",
            intent=state.intent,
        )
        if chart_path:
            final_response += f"\n\n[圖表已生成: {chart_path}]"

    return {
        "final_response": final_response,
        "chart_path": chart_path,
        "messages": [AIMessage(content=final_response)],
    }


def _fallback_extract(content: str) -> str:
    """當 LLM 未正確呼叫 tool 時，從文字回應中提取最終回覆。

    Args:
        content: LLM 的原始 content 文字，可能混雜英文推理過程與中文回覆。

    Process:
        1. 從文字末尾往前掃描
        2. 收集包含 3 個以上中文字元的連續行
        3. 遇到非中文行時停止（認為前面都是推理過程）
        4. 若完全找不到中文行，回傳原始 content

    Returns:
        str: 提取出的中文回覆文字。若無法提取則回傳原始內容。
    """
    lines = content.strip().split("\n")
    chinese_lines = []

    for line in reversed(lines):
        line = line.strip()
        if not line:
            if chinese_lines:
                break
            continue
        chinese_chars = sum(1 for c in line if '一' <= c <= '鿿')
        if chinese_chars >= 3:
            chinese_lines.insert(0, line)
        elif chinese_lines:
            break

    if chinese_lines:
        return "\n".join(chinese_lines)

    return content

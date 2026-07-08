"""Agent State 定義模組。

定義 LangGraph Agent 的狀態結構，
所有節點共用此狀態進行資料傳遞。
"""

from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """LangGraph Agent 的全域狀態定義。

    Args:
        messages: 對話歷史訊息列表，使用 LangGraph 的 add_messages reducer
                  自動管理訊息累加（支援多輪對話）。
        intent: 意圖解析結果，包含查詢類型與提取的實體。
        schema_info: 從資料庫讀取的 Schema 資訊文字。
        generated_sql: LLM 生成的 SQL 查詢語句。
        sql_valid: SQL 驗證是否通過。
        sql_error: SQL 驗證失敗時的錯誤訊息。
        retry_count: SQL 生成重試次數（上限 3 次）。
        query_result: SQL 執行結果（list of dict 格式）。
        response_format: 回覆格式決策結果（text/table/chart）。
        chart_type: 若決策為圖表，指定圖表類型（bar/line/pie）。
        final_response: 最終組合的回覆文字。
        chart_path: 生成的圖表檔案路徑。
        error: 流程中發生的錯誤訊息。
        total_tokens_used: 累計的對話歷史 token 數（由 LLM response_metadata 取得），
                           用於判斷是否需要壓縮歷史對話。

    Process:
        此 State 在 LangGraph 的各節點間傳遞，
        每個節點讀取所需欄位、處理後更新對應欄位。
        messages 欄位透過 add_messages reducer 自動累加對話歷史。

    Returns:
        無（作為資料結構定義使用）。
    """

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    intent: Optional[dict[str, Any]] = None
    schema_info: Optional[str] = None
    generated_sql: Optional[str] = None
    sql_valid: Optional[bool] = None
    sql_error: Optional[str] = None
    retry_count: int = 0
    query_result: Optional[list[dict[str, Any]]] = None
    response_format: Optional[str] = None
    chart_type: Optional[str] = None
    final_response: Optional[str] = None
    chart_path: Optional[str] = None
    error: Optional[str] = None
    total_tokens_used: int = 0

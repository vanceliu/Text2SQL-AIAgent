"""LangGraph 工作流定義模組。

定義 Text2SQL Agent 的 StateGraph，編排所有節點的執行順序與條件路由。
包含 7 個核心節點：意圖理解 → Schema 擷取 → SQL 生成 → SQL 驗證 →
SQL 執行 → 回覆格式決策 → 回覆組合。
"""

import os
import sqlite3
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import AgentState
from agent.nodes.intent_parser import parse_intent
from agent.nodes.schema_fetcher import fetch_schema
from agent.nodes.text2sql import generate_sql
from agent.nodes.sql_validator import validate_sql
from agent.nodes.sql_executor import execute_sql
from agent.nodes.response_router import route_response
from agent.nodes.response_composer import compose_response


def build_graph() -> StateGraph:
    """建構 Text2SQL Agent 的 LangGraph StateGraph（未編譯）。

    Args:
        無。

    Process:
        1. 以 AgentState 為狀態定義建立 StateGraph
        2. 依序加入 7 個節點：
           - intent_parser: 分析使用者意圖
           - schema_fetcher: 讀取資料庫 Schema
           - text2sql: 生成 SQL 查詢
           - sql_validator: 驗證 SQL 安全性與語法
           - sql_executor: 執行 SQL 查詢
           - response_router: 決策回覆格式
           - response_composer: 組合最終回覆
        3. 設定節點間的邊（包含條件路由）：
           - START → intent_parser
           - intent_parser → 條件路由（可回答 → schema_fetcher，不可回答 → response_composer）
           - schema_fetcher → text2sql
           - text2sql → sql_validator
           - sql_validator → 條件路由（通過 → sql_executor，失敗且可重試 → text2sql，超過重試 → response_composer）
           - sql_executor → 條件路由（成功 → response_router，失敗 → response_composer）
           - response_router → response_composer
           - response_composer → END

    Returns:
        StateGraph: 已建構完成但尚未編譯的 StateGraph 物件。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_parser", parse_intent)
    workflow.add_node("schema_fetcher", fetch_schema)
    workflow.add_node("text2sql", generate_sql)
    workflow.add_node("sql_validator", validate_sql)
    workflow.add_node("sql_executor", execute_sql)
    workflow.add_node("response_router", route_response)
    workflow.add_node("response_composer", compose_response)

    workflow.add_edge(START, "intent_parser")
    workflow.add_conditional_edges("intent_parser", _route_after_intent)
    workflow.add_edge("schema_fetcher", "text2sql")
    workflow.add_edge("text2sql", "sql_validator")
    workflow.add_conditional_edges("sql_validator", _route_after_validation)
    workflow.add_conditional_edges("sql_executor", _route_after_execution)
    workflow.add_edge("response_router", "response_composer")
    workflow.add_edge("response_composer", END)

    return workflow


def compile_graph(checkpointer=None):
    """編譯 LangGraph 工作流，加入 SQLite checkpointer 支援多輪對話與歷史持久化。

    Args:
        checkpointer: LangGraph checkpointer 實例，用於持久化對話狀態。
                      若為 None，預設使用 SqliteSaver 存到 data/checkpoints.db。

    Process:
        1. 呼叫 build_graph() 取得 StateGraph
        2. 若無傳入 checkpointer，建立 SqliteSaver（SQLite 持久化）
        3. 使用 checkpointer 編譯 graph

    Returns:
        CompiledGraph: 已編譯完成、可直接 invoke 的 LangGraph 實例。
    """
    workflow = build_graph()

    if checkpointer is None:
        project_root = Path(__file__).parent.parent.parent
        checkpoint_db_path = str(project_root / os.getenv("CHECKPOINT_DB_PATH"))
        Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


def _route_after_intent(state: AgentState) -> Literal["schema_fetcher", "response_composer"]:
    """意圖解析後的條件路由：判斷問題是否可回答。

    Args:
        state: 當前 Agent 狀態，讀取 state.intent。

    Process:
        若意圖為 error 類型或 is_answerable 為 False，
        直接跳到 response_composer 回覆無法回答。
        否則進入 schema_fetcher 繼續處理。

    Returns:
        str: 下一個節點名稱，"schema_fetcher" 或 "response_composer"。
    """
    intent = state.intent or {}

    if intent.get("intent_type") == "error" or not intent.get("is_answerable", True):
        return "response_composer"

    return "schema_fetcher"


def _route_after_validation(state: AgentState) -> Literal["sql_executor", "text2sql", "response_composer"]:
    """SQL 驗證後的條件路由：通過/重試/放棄。

    Args:
        state: 當前 Agent 狀態，讀取 state.sql_valid、state.error。

    Process:
        - 若驗證通過（sql_valid=True）→ 進入 sql_executor 執行
        - 若有終止錯誤（error 不為 None）→ 跳到 response_composer 回報錯誤
        - 否則（驗證失敗但可重試）→ 回到 text2sql 重新生成

    Returns:
        str: 下一個節點名稱。
    """
    if state.sql_valid:
        return "sql_executor"

    if state.error:
        return "response_composer"

    return "text2sql"


def _route_after_execution(state: AgentState) -> Literal["response_router", "response_composer"]:
    """SQL 執行後的條件路由：成功或失敗。

    Args:
        state: 當前 Agent 狀態，讀取 state.query_result、state.error。

    Process:
        若有查詢結果（query_result 不為 None）→ 進入 response_router 決策格式。
        若執行失敗（error 不為 None 或 query_result 為 None）→ 跳到 response_composer。

    Returns:
        str: 下一個節點名稱。
    """
    if state.query_result is not None:
        return "response_router"

    return "response_composer"

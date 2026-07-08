"""LangGraph 工作流定義的單元測試。

測試條件路由邏輯。
"""

import pytest
from agent.graph import (
    _route_after_intent,
    _route_after_validation,
    _route_after_execution,
)
from agent.state import AgentState


class TestRouteAfterIntent:
    """測試意圖解析後的路由邏輯。"""

    def test_answerable_goes_to_schema(self):
        """可回答的問題應路由到 schema_fetcher。"""
        state = AgentState(intent={"intent_type": "aggregate", "is_answerable": True})
        assert _route_after_intent(state) == "schema_fetcher"

    def test_unanswerable_goes_to_composer(self):
        """不可回答的問題應路由到 response_composer。"""
        state = AgentState(intent={"intent_type": "error", "is_answerable": False})
        assert _route_after_intent(state) == "response_composer"

    def test_error_type_goes_to_composer(self):
        """error 類型應路由到 response_composer。"""
        state = AgentState(intent={"intent_type": "error", "is_answerable": True})
        assert _route_after_intent(state) == "response_composer"

    def test_none_intent_goes_to_schema(self):
        """intent 為 None 時（預設 is_answerable=True）應路由到 schema_fetcher。"""
        state = AgentState(intent=None)
        assert _route_after_intent(state) == "schema_fetcher"


class TestRouteAfterValidation:
    """測試 SQL 驗證後的路由邏輯。"""

    def test_valid_goes_to_executor(self):
        """驗證通過應路由到 sql_executor。"""
        state = AgentState(sql_valid=True)
        assert _route_after_validation(state) == "sql_executor"

    def test_invalid_with_error_goes_to_composer(self):
        """驗證失敗且有終止 error 應路由到 response_composer。"""
        state = AgentState(sql_valid=False, error="超過重試上限")
        assert _route_after_validation(state) == "response_composer"

    def test_invalid_no_error_retries(self):
        """驗證失敗但無終止 error 應回到 text2sql 重試。"""
        state = AgentState(sql_valid=False, error=None)
        assert _route_after_validation(state) == "text2sql"


class TestRouteAfterExecution:
    """測試 SQL 執行後的路由邏輯。"""

    def test_has_result_goes_to_router(self):
        """有查詢結果應路由到 response_router。"""
        state = AgentState(query_result=[{"count": 1000}])
        assert _route_after_execution(state) == "response_router"

    def test_no_result_goes_to_composer(self):
        """無結果（None）應路由到 response_composer。"""
        state = AgentState(query_result=None)
        assert _route_after_execution(state) == "response_composer"

    def test_empty_list_goes_to_router(self):
        """空列表（[]）仍算有結果，應路由到 response_router。"""
        state = AgentState(query_result=[])
        assert _route_after_execution(state) == "response_router"

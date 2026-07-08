"""Agent State 的單元測試。

測試 AgentState 的 Pydantic model 定義與預設值。
"""

import pytest
from agent.state import AgentState


class TestAgentState:
    """測試 AgentState model 的基本行為。"""

    def test_default_values(self):
        """建立空 state 時所有欄位應有正確預設值。"""
        state = AgentState()
        assert state.messages == []
        assert state.intent is None
        assert state.schema_info is None
        assert state.generated_sql is None
        assert state.sql_valid is None
        assert state.sql_error is None
        assert state.retry_count == 0
        assert state.query_result is None
        assert state.response_format is None
        assert state.chart_type is None
        assert state.final_response is None
        assert state.chart_path is None
        assert state.error is None
        assert state.total_tokens_used == 0

    def test_partial_update(self):
        """部分更新欄位不應影響其他欄位。"""
        state = AgentState(intent={"intent_type": "aggregate"}, retry_count=2)
        assert state.intent == {"intent_type": "aggregate"}
        assert state.retry_count == 2
        assert state.messages == []
        assert state.generated_sql is None

    def test_total_tokens_used_field(self):
        """total_tokens_used 應接受整數值。"""
        state = AgentState(total_tokens_used=5000)
        assert state.total_tokens_used == 5000

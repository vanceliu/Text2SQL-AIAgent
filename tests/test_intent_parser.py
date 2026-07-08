"""意圖解析節點的單元測試。

測試 fallback 解析、token 估算、messages 準備邏輯。
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.nodes.intent_parser import (
    _fallback_parse,
    _estimate_tokens,
    _estimate_message_tokens,
    _messages_to_text,
    _prepare_messages,
    MAX_HISTORY_TOKENS,
)


class TestFallbackParse:
    """測試 _fallback_parse 的 JSON 解析 fallback 邏輯。"""

    def test_valid_json(self):
        """合法 JSON 應正確解析。"""
        content = '{"intent_type": "aggregate", "description": "test", "entities": {}, "is_answerable": true}'
        result = _fallback_parse(content)
        assert result["intent_type"] == "aggregate"
        assert result["is_answerable"] is True

    def test_json_in_code_block(self):
        """包在 markdown code block 中的 JSON 應正確提取。"""
        content = '```json\n{"intent_type": "detail", "description": "test", "entities": {}, "is_answerable": true}\n```'
        result = _fallback_parse(content)
        assert result["intent_type"] == "detail"

    def test_invalid_json(self):
        """無法解析的文字應回傳 error 類型。"""
        result = _fallback_parse("this is not json at all")
        assert result["intent_type"] == "error"
        assert result["is_answerable"] is False

    def test_empty_content(self):
        """空字串應回傳 error 類型。"""
        result = _fallback_parse("")
        assert result["intent_type"] == "error"


class TestTokenEstimation:
    """測試 token 估算函式。"""

    def test_estimate_message_tokens_short(self):
        """短訊息的 token 估算。"""
        msg = HumanMessage(content="hello")
        tokens = _estimate_message_tokens(msg)
        assert tokens >= 1

    def test_estimate_message_tokens_long(self):
        """長訊息應估算出較多 tokens。"""
        msg = AIMessage(content="這是一段很長的回覆" * 100)
        tokens = _estimate_message_tokens(msg)
        assert tokens > 100

    def test_estimate_tokens_list(self):
        """多則 messages 的 token 總和。"""
        msgs = [
            HumanMessage(content="問題一"),
            AIMessage(content="回答一" * 50),
            HumanMessage(content="問題二"),
        ]
        total = _estimate_tokens(msgs)
        assert total > 0


class TestMessagesToText:
    """測試 _messages_to_text 轉換。"""

    def test_basic_conversion(self):
        """基本的 messages 轉文字。"""
        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="你好，有什麼可以幫你？"),
        ]
        text = _messages_to_text(msgs)
        assert "使用者" in text
        assert "助手" in text
        assert "你好" in text

    def test_long_message_not_truncated(self):
        """長 message 不應被截斷（完整保留供摘要用）。"""
        long_content = "x" * 500
        msgs = [AIMessage(content=long_content)]
        text = _messages_to_text(msgs)
        assert "..." not in text
        assert len(text) > 500


class TestPrepareMessages:
    """測試 _prepare_messages 壓縮邏輯。"""

    def test_below_threshold_no_compression(self):
        """低於 token 門檻時不壓縮，直接回傳全部。"""
        msgs = [
            HumanMessage(content="短問題"),
            AIMessage(content="短回答"),
        ]
        llm = MagicMock()
        result = _prepare_messages(msgs, 100, llm)
        assert len(result) == 2
        llm.invoke.assert_not_called()

    def test_above_threshold_compresses(self):
        """超過 token 門檻時應壓縮舊歷史。"""
        msgs = [
            HumanMessage(content="問題一"),
            AIMessage(content="很長的回答" * 500),
            HumanMessage(content="問題二"),
            AIMessage(content="很長的回答" * 500),
            HumanMessage(content="最新問題"),
        ]
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "摘要：使用者之前問了兩個問題"
        mock_llm.invoke.return_value = mock_response

        result = _prepare_messages(msgs, MAX_HISTORY_TOKENS + 1000, mock_llm)

        assert any(
            isinstance(m, SystemMessage) and "摘要" in m.content
            for m in result
        )
        mock_llm.invoke.assert_called_once()

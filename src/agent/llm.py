"""LLM 初始化模組。

統一管理 LLM 實例的建立，透過 LM Studio 的 OpenAI-compatible API 連接地端模型。
所有節點共用此模組取得 LLM 實例。
"""

import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """建立並回傳連接 LM Studio 的 ChatOpenAI 實例。

    Args:
        temperature: LLM 生成溫度，0.0 為最確定性輸出，1.0 為最隨機。
                     預設 0.0 以確保 SQL 生成的一致性。

    Process:
        1. 從環境變數讀取 LM_STUDIO_BASE_URL 和 LM_STUDIO_MODEL
        2. 使用 ChatOpenAI 建立連接 LM Studio local server 的實例
        3. 設定 api_key 為 "lm-studio"（LM Studio 不需要真正的 key）

    Returns:
        ChatOpenAI: 已設定好的 LangChain ChatOpenAI 實例，
                    指向 LM Studio 的本地 API endpoint。
    """
    base_url = os.getenv("LM_STUDIO_BASE_URL")
    model = os.getenv("LM_STUDIO_MODEL")

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key="lm-studio",
        temperature=temperature,
    )

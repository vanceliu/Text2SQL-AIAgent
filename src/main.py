"""Text2SQL AI Agent 主程式入口。

提供 CLI 互動介面，使用者可用自然語言查詢超市銷售數據。
支援多輪對話，輸入 'exit' 或 'quit' 結束。
"""

import os
import uuid

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage

from agent.graph import compile_graph


def setup_langsmith():
    """設定 LangSmith 追蹤環境變數。

    Args:
        無。

    Process:
        1. 檢查 .env 中是否已設定 LANGSMITH_API_KEY
        2. 若有設定，啟用 LANGSMITH_TRACING
        3. 設定 LANGSMITH_PROJECT 名稱

    Returns:
        無（直接修改環境變數）。
    """

    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "your-langsmith-api-key-here":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "text2sql-agent")
        print("[INFO] LangSmith Tracing 已啟用")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        print("[INFO] LangSmith 未設定 API Key，Tracing 已停用")


def run_cli():
    """啟動 CLI 互動迴圈，接收使用者輸入並呼叫 Agent 處理。

    Args:
        無。

    Process:
        1. 初始化 LangSmith 設定
        2. 編譯 LangGraph（含 InMemorySaver checkpointer）
        3. 生成唯一的 thread_id 用於多輪對話
        4. 進入互動迴圈：
           a. 讀取使用者輸入
           b. 將訊息包裝為 HumanMessage
           c. 呼叫 graph.invoke() 執行 Agent 工作流
           d. 輸出最終回覆
           e. 若有圖表路徑，提示使用者查看
        5. 使用者輸入 exit/quit 時結束

    Returns:
        無（持續執行直到使用者結束）。
    """
    setup_langsmith()

    print("\n" + "=" * 60)
    print("  Text2SQL AI Agent")
    print("  輸入自然語言問題，我會幫您查詢銷售數據")
    print("  輸入 'exit' 或 'quit' 結束對話")
    print("=" * 60 + "\n")

    graph = compile_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nExit!")
            break

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            final_response = result.get("final_response", "抱歉，處理過程發生錯誤。")
            print(f"\nAgent: {final_response}\n")

            chart_path = result.get("chart_path")
            if chart_path:
                print(f"  [圖表位置: {chart_path}]\n")

        except Exception as e:
            print(f"\n[錯誤] Agent 執行失敗: {str(e)}\n")


if __name__ == "__main__":
    run_cli()

"""Text2SQL AI Agent - Streamlit Web UI。

提供網頁互動介面，使用者可透過瀏覽器以自然語言查詢超市銷售數據。
支援多輪對話、表格顯示、圖表顯示、session 歷史管理。
"""

import os
import sys
import uuid
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import compile_graph

EXAMPLE_QUESTIONS = [
    "這個超市總共有多少筆交易？",
    "各分店的總銷售額是多少？",
    "比較各產品線的銷售佔比",
    "列出所有評分高於 9 分的會員交易",
    "每月的銷售趨勢如何？",
    "幫我預測下個月的銷售額",
]


def init_session():
    """初始化 Streamlit session state，設定對話歷史與 Agent。

    Args:
        無。

    Process:
        1. 載入環境變數
        2. 設定 LangSmith tracing
        3. 若 session 中尚未建立 graph，編譯 LangGraph
        4. 為每個 session 產生唯一 thread_id
        5. 初始化 messages 對話歷史列表
        6. 初始化 session 列表

    Returns:
        無（直接修改 st.session_state）。
    """
    load_dotenv()

    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "your-langsmith-api-key-here":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "text2sql-agent")

    if "graph" not in st.session_state:
        st.session_state.graph = compile_graph()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sessions" not in st.session_state:
        st.session_state.sessions = _load_sessions()


def _load_sessions() -> list[dict]:
    """從 SQLite checkpointer 讀取所有 session（thread_id）列表。

    Args:
        無。

    Process:
        1. 連接 checkpoints.db
        2. 查詢所有不重複的 thread_id
        3. 取得每個 thread 最新的 checkpoint 時間
        4. 回傳按時間降序排列的 session 列表

    Returns:
        list[dict]: session 列表，每筆含 thread_id 和 last_active 時間。
    """
    project_root = Path(__file__).parent
    checkpoint_db = project_root / os.getenv("CHECKPOINT_DB_PATH")

    if not checkpoint_db.exists():
        return []

    try:
        conn = sqlite3.connect(str(checkpoint_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id, MAX(checkpoint_id) as last_cp "
            "FROM checkpoints GROUP BY thread_id ORDER BY last_cp DESC"
        )
        sessions = [
            {"thread_id": row[0], "last_checkpoint": row[1]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return sessions
    except Exception:
        return []


def _load_messages_from_checkpoint(thread_id: str) -> list[dict]:
    """從 checkpointer 讀取指定 thread 的對話歷史。

    Args:
        thread_id: 要讀取的 thread_id。

    Process:
        1. 更新 session_state.thread_id
        1. 使用 graph.get_state() 取得該 thread 的最新 state
        2. 從 state 的 messages 中提取 HumanMessage 和 AIMessage
        3. 轉換為 UI 顯示格式 [{"role": "user/assistant", "content": "..."}]

    Returns:
        list[dict]: 對話歷史列表，可直接用於 render_chat_history()。
    """

    st.session_state.thread_id = thread_id
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = st.session_state.graph.get_state(config)

        if not state or not state.values:
            st.session_state.messages = []

        messages = state.values.get("messages", [])
        chat_history = []

        for msg in messages:
            if hasattr(msg, "type"):
                if msg.type == "human":
                    chat_history.append({"role": "user", "content": msg.content})
                elif msg.type == "ai" and msg.content:
                    content = msg.content
                    chart_path = _extract_chart_path(content)
                    if chart_path:
                        content = content.replace(f"\n\n[圖表已生成: {chart_path}]", "")
                    chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "chart_path": chart_path,
                    })

        st.session_state.messages = chat_history
    except Exception:
        st.session_state.messages = []


def _extract_chart_path(content: str) -> str | None:
    """從 AI 回覆文字中提取圖表檔案路徑。

    Args:
        content: AI 回覆的完整文字內容。

    Process:
        搜尋 "[圖表已生成: /path/to/chart.png]" 格式的文字，
        提取其中的檔案路徑。

    Returns:
        str | None: 圖表路徑（若存在且檔案確實存在），否則回傳 None。
    """
    marker = "[圖表已生成: "
    if marker in content:
        start = content.index(marker) + len(marker)
        end = content.index("]", start)
        path = content[start:end]
        if os.path.exists(path):
            return path
    return None


def _delete_session(thread_id: str):
    """刪除指定 session 的所有 checkpoint 資料。

    Args:
        thread_id: 要刪除的 thread_id。

    Process:
        1. 連接 checkpoints.db
        2. 刪除該 thread_id 的所有 checkpoints 和 writes 紀錄
        3. 若刪除的是目前 session，重新產生新的 thread_id
        4. 刷新 session 列表

    Returns:
        無（直接修改資料庫與 st.session_state）。
    """
    project_root = Path(__file__).parent
    checkpoint_db = project_root / os.getenv("CHECKPOINT_DB_PATH")

    try:
        conn = sqlite3.connect(str(checkpoint_db))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass

    if st.session_state.thread_id == thread_id:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []

    st.session_state.sessions = _load_sessions()


def render_sidebar():
    """渲染左側邊欄：session 列表與操作按鈕。

    Args:
        無。

    Process:
        1. 顯示標題與說明
        2. 「新對話」按鈕
        3. 列出所有歷史 session，點擊可切換
        4. 「清除所有對話」按鈕

    Returns:
        無（直接渲染到 Streamlit sidebar）。
    """
    with st.sidebar:
        st.title("超市銷售查詢助手")
        st.caption("Text2SQL AI Agent")
        st.markdown("---")

        if st.button("➕ 新對話", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.markdown("### 對話紀錄")
        sessions = st.session_state.sessions
        if sessions:
            for i, session in enumerate(sessions[:20]):
                tid = session["thread_id"]
                label = f"{tid[:8]}..."
                is_current = tid == st.session_state.thread_id

                col_btn, col_del = st.columns([4, 1])
                with col_btn:
                    if is_current:
                        st.markdown(f"**▶ {label}** (目前)")
                    else:
                        if st.button(label, key=f"session_{i}", use_container_width=True):
                            _load_messages_from_checkpoint(tid)
                            st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{i}", help="刪除此對話"):
                        _delete_session(tid)
                        st.rerun()
        else:
            st.caption("尚無對話紀錄")


def render_chat_history():
    """渲染聊天歷史訊息，包含文字和圖表。

    Args:
        無。

    Process:
        遍歷 session_state.messages，根據角色（user/assistant）
        分別渲染訊息氣泡。若訊息附有圖表路徑，顯示圖片。

    Returns:
        無（直接渲染到 Streamlit 頁面）。
    """
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
                st.image(msg["chart_path"])


def render_suggestion_chips():
    """渲染 suggestion chips（範例問題），位於對話區域底部、輸入框上方。

    Args:
        無。

    Process:
        1. 使用 st.pills 顯示範例問題作為可點擊的 chips
        2. 使用者點擊後觸發該問題的查詢
        3. 透過動態 key 強制重置 widget 選取狀態

    Returns:
        無（直接渲染到 Streamlit 頁面，若有點選則觸發查詢）。
    """

    if "pills_key_counter" not in st.session_state:
        st.session_state.pills_key_counter = 0

    selected = st.pills(
        "範例問題",
        EXAMPLE_QUESTIONS,
        selection_mode="single",
        label_visibility="collapsed",
        key=f"suggestion_pills_{st.session_state.pills_key_counter}",
    )

    if selected:
        st.session_state.pills_key_counter += 1
        handle_user_input(selected)
        st.rerun()


def handle_user_input(user_input: str):
    """處理使用者輸入，呼叫 Agent 並顯示回覆。

    Args:
        user_input: 使用者的自然語言問題字串。

    Process:
        1. 將使用者訊息加入 session_state.messages
        2. 顯示使用者訊息氣泡
        3. 顯示 "thinking" spinner
        4. 呼叫 graph.invoke() 執行 Agent 工作流
        5. 取得回覆文字和圖表路徑
        6. 在 assistant 訊息氣泡中顯示回覆
        7. 若有圖表，顯示圖片
        8. 將 assistant 訊息加入對話歷史
        9. 刷新 session 列表

    Returns:
        無（直接渲染到 Streamlit 頁面）。
    """
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

                final_response = result.get("final_response", "抱歉，處理過程發生錯誤。")
                chart_path = result.get("chart_path")

                st.markdown(final_response)

                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "chart_path": chart_path,
                })

            except Exception as e:
                error_msg = f"Agent 執行發生錯誤: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

    st.session_state.sessions = _load_sessions()
    st.rerun()


def main():
    """Streamlit App 主程式，設定頁面配置並渲染 UI 元件。

    Args:
        無。

    Process:
        1. 設定頁面標題、icon、layout
        2. 初始化 session
        3. 渲染左側 session 列表
        4. 渲染聊天歷史
        5. 渲染 suggestion chips
        6. 監聽 chat_input 事件

    Returns:
        無（Streamlit App 持續運行）。
    """
    st.set_page_config(
        page_title="Text2SQL AI Agent",
        page_icon="🛒",
        layout="wide",
    )

    init_session()
    render_sidebar()

    st.title("Text2SQL AI Agent")
    st.caption("基於地端模型 Gemma 4 E4B + LangGraph")

    render_chat_history()
    render_suggestion_chips()

    if prompt := st.chat_input("請輸入您的問題..."):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()

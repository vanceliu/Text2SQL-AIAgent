# Text2SQL AI Agent 架構設計文件

## 一、系統概述

本專案實作一個基於地端模型的 Text2SQL AI Agent，使用者可用自然語言查詢超市銷售數據，Agent 自動將自然語言轉換為 SQL，執行查詢後根據語意與資料特徵智慧決定回覆格式（口語/列表/圖表）。

### 技術選型

| 元件 | 選擇 | 理由 |
|------|------|------|
| LLM | Gemma 4 E4B | Google 開源模型，4B 有效參數，支援 128K context，內建 Function Calling |
| LLM 服務 | LM Studio | OpenAI-compatible API，支援 tools 參數，GUI 友善 |
| Agent 框架 | LangGraph 1.2.8 | 支援 StateGraph、conditional edges、SqliteSaver checkpointer |
| 資料庫 | SQLite | 輕量嵌入式，零配置，適合單檔案資料集 |
| Checkpointer | SqliteSaver | 對話歷史持久化至 SQLite，程式重啟後可恢復 |
| Tracing | LangSmith | LangChain 生態系原生整合，免費方案即可 |
| 圖表 | matplotlib | 成熟穩定，支援中文，無外部依賴 |
| Web UI | Streamlit | 快速建立互動 UI，內建 chat 元件、session 管理 |
| 資料驗證 | Pydantic | CSV 匯入時逐筆驗證型別與格式 |

### 為什麼使用原生 Function Calling

Gemma 4 E4B 內建函式呼叫（Function Calling）支援，LM Studio 的 OpenAI-compatible API
亦支援 `tools` 參數，因此本專案使用原生 FC 路線：
- 透過 `llm.bind_tools([PydanticSchema], tool_choice="any")` 強制 LLM 呼叫 tool
- 從 `response.tool_calls[0]["args"]` 直接取得結構化結果，免去 JSON 解析
- 以 Pydantic BaseModel 定義 tool schema，提供型別安全與欄位驗證
- 保留 fallback 邏輯（文字解析 / 規則引擎），確保 FC 不穩定時仍能運作

三個使用 FC 的節點：
- **Intent Parser** → `IntentResult` schema（意圖類型 + 實體提取）
- **Text2SQL** → `SQLGenerationResult` schema（SQL 語句 + 說明）
- **Response Router** → `ResponseFormatResult` schema（格式決策 + 理由）

## 二、LangGraph 工作流設計

### State 設計

```python
class AgentState(BaseModel):
    messages: Annotated[list, add_messages]  # 對話歷史（多輪）
    intent: Optional[dict]                    # 意圖解析結果
    schema_info: Optional[str]               # DB Schema 文字
    generated_sql: Optional[str]             # 生成的 SQL
    sql_valid: Optional[bool]                # 驗證結果
    sql_error: Optional[str]                 # 驗證錯誤訊息
    retry_count: int                          # 重試計數
    query_result: Optional[list[dict]]       # 查詢結果
    response_format: Optional[str]           # 格式決策
    chart_type: Optional[str]                # 圖表類型
    final_response: Optional[str]            # 最終回覆
    chart_path: Optional[str]                # 圖表路徑
    error: Optional[str]                      # 錯誤訊息
    total_tokens_used: int                    # 累計 token 數（壓縮門檻判斷）
```

設計原則：
- 使用 `add_messages` reducer 自動累加對話歷史，支援多輪對話
- 透過 `SqliteSaver` checkpointer + `thread_id` 實現對話狀態持久化
- 每個節點只讀取/寫入自己需要的欄位，保持低耦合
- `total_tokens_used` 用於判斷是否需要壓縮歷史對話

### 工作流程圖

```
START
  │
  ▼
┌─────────────────┐
│  Intent Parser  │ ← 分析意圖，判斷是否可回答（含歷史摘要壓縮）
└────────┬────────┘
         │
    ┌────┴─────┐
    │ 可回答?  │
    └────┬─────┘
    Yes  │  No ──────────────────────────┐
         ▼                                │
┌─────────────────┐                       │
│ Schema Fetcher  │ ← 讀取 DB Schema       │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│    Text2SQL     │ ← LLM 生成 SQL (FC)    │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│  SQL Validator  │ ← 驗證安全性/語法       │
└────────┬────────┘                       │
    ┌────┴─────┐                          │
    │ 通過?     │                          │
    └────┬─────┘                          │
  Pass   │  Fail ──→ retry < 3? ──Yes──→ Text2SQL
         │                    No ─────────┤
         ▼                                │
┌─────────────────┐                       │
│  SQL Executor   │ ← 執行查詢             │
└────────┬────────┘                       │
    ┌────┴─────┐                          │
    │ 成功?     │                          │
    └────┬─────┘                          │
  Yes    │  No ───────────────────────────┤
         ▼                                │
┌─────────────────┐                       │
│Response Router  │ ← 決策回覆格式 (FC)     │
└────────┬────────┘                       │
         ▼                                │
┌─────────────────┐                       │
│Response Composer│ ← 組合最終回覆  ◄───────┘
└────────┬────────┘
         ▼
        END
```

## 三、各節點職責詳述

### 1. Intent Parser（意圖理解）
- 輸入：使用者訊息 + 對話歷史
- 處理：透過 Function Calling（`IntentResult` schema）分析意圖類型
- 輸出：意圖結構（type、description、entities、is_answerable）
- 特點：
  - 支援多輪對話的指代消解（如「那 Giza 呢？」）
  - 內建歷史摘要壓縮：當 `total_tokens_used` 超過 6000 時，將舊對話壓縮為摘要送給 LLM，checkpointer 中的原始歷史不受影響

### 2. Schema Fetcher（Schema 擷取）
- 輸入：無（直接讀取 DB）
- 處理：PRAGMA table_info + 每欄前 5 筆不重複範例值
- 輸出：格式化的 Schema 文字描述
- 特點：範例值讓 LLM 知道欄位的精確拼寫與格式，降低 SQL 生成錯誤率

### 3. Text2SQL（SQL 生成）
- 輸入：意圖 + Schema + 使用者問題 + 前次錯誤（若重試）
- 處理：透過 Function Calling（`SQLGenerationResult` schema）生成 SQL
- 輸出：SELECT SQL 語句 + 邏輯說明
- 特點：重試時會附帶前次錯誤訊息，讓 LLM 修正

### 4. SQL Validator（SQL 驗證）
- 輸入：生成的 SQL
- 處理：安全性檢查（危險關鍵字）+ 語法檢查（EXPLAIN）
- 輸出：驗證結果 + 錯誤訊息 + 重試計數
- 特點：三重防護（sqlparse 語句類型 + 關鍵字掃描 + SQLite EXPLAIN）

### 5. SQL Executor（SQL 執行）
- 輸入：已驗證的 SQL
- 處理：在 SQLite 上安全執行，轉為 list of dict
- 輸出：查詢結果
- 特點：設有逾時保護（30 秒）和最大筆數限制（100 筆）

### 6. Response Router（回覆格式決策）⭐
- 輸入：意圖 + 查詢結果 + 使用者問題
- 處理：透過 Function Calling（`ResponseFormatResult` schema）決策格式
- 輸出：格式決策（text/table/chart）+ 圖表類型
- 特點：雙層決策（LLM FC 主判斷 + 規則引擎 fallback 兜底）

### 7. Response Composer（回覆組合）
- 輸入：查詢結果 + 格式決策 + 使用者問題
- 處理：LLM 生成口語回覆（few-shot prompt）+ 圖表生成（若需要）
- 輸出：最終文字回覆 + 圖表路徑
- 特點：
  - 使用 few-shot examples 確保 LLM 直接輸出最終回覆（避免 thinking 外洩）
  - 一律包含口語文字，圖表/表格為附加

## 四、回覆格式決策邏輯

決策引擎綜合考量以下因素：

| 判斷依據 | 口語 (text) | 列表 (table) | 圖表 (chart) |
|----------|-------------|--------------|--------------|
| 使用者語意 | 「總共」「平均」 | 「列出」「哪些」 | 「趨勢」「比較」「佔比」 |
| 結果筆數 | 1 筆 | 2-20 筆 | 2+ 筆分類統計 |
| 結果結構 | 單欄單值 | 多欄多列 | 含可比較數值 |
| 意圖類型 | aggregate | detail | trend/comparison |

圖表類型選擇：
- **bar（長條圖）**：比較各類別數值，如各分店銷售額
- **line（折線圖）**：時間序列趨勢，如月銷售變化
- **pie（圓餅圖）**：佔比分佈，如產品線銷售佔比

## 五、多輪對話與歷史管理

### Checkpointer 持久化

透過 LangGraph 的 SqliteSaver checkpointer 機制：

```python
conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "unique-session-id"}}
graph.invoke({"messages": [HumanMessage(content="...")]}, config)
```

- 每次 invoke 後，完整的 AgentState 被持久化到 SQLite
- 程式重啟後，同 thread_id 的 invoke 可自動恢復對話歷史
- Intent Parser 能讀取歷史 messages，理解上下文指代
- Streamlit UI 可列出所有歷史 session 並切換

### 對話歷史摘要壓縮

當 `total_tokens_used` 超過 6000 tokens 門檻時：
1. 從最新的 messages 往回保留約 3000 tokens
2. 將更早的 messages 壓縮為一段摘要（由 LLM 生成，100 字以內）
3. 壓縮結果**僅用於送給 LLM 的上下文**，不影響 checkpointer 中的原始歷史
4. 確保 LLM 推論效能不隨對話輪次增加而劣化

## 六、安全性設計

1. **SQL Injection 防護**：
   - sqlparse 語句類型檢查（僅允許 SELECT）
   - 危險關鍵字逐一比對（INSERT/UPDATE/DELETE/DROP 等 13 個）
   - SQLite EXPLAIN 語法驗證
   - 三重防護，任一不通過即拒絕執行

2. **執行安全**：
   - 查詢逾時保護（30 秒）
   - 結果筆數限制（最多 100 筆）
   - 重試上限（3 次），防止無限迴圈

3. **資料驗證**：
   - CSV 匯入使用 Pydantic model 逐筆驗證型別與格式
   - 日期、時間欄位透過 `@field_validator` 嚴格解析

4. **地端執行**：
   - LLM 推論完全在本機（LM Studio），不外傳資料
   - 僅 LangSmith Tracing 需網路連線（可選關閉）

## 七、技術挑戰與解決方案

| 挑戰 | 解決方案 |
|------|---------|
| 地端模型 SQL 生成品質 | Schema 資訊注入（含範例值）+ Function Calling 結構化輸出 + 重試機制 |
| Gemma 4 thinking mode 外洩 | 全節點改用 Function Calling（bind_tools + tool_choice="any"），從 tool_calls 取值避免 content 中的推理外洩 |
| 多輪對話 context 膨脹 | token 門檻（6000）觸發歷史壓縮摘要，不影響 checkpointer 原始資料 |
| 回覆格式決策準確性 | LLM Function Calling 主判斷 + 規則引擎 fallback 雙層決策 |
| 對話歷史持久化 | SqliteSaver checkpointer，程式重啟後可恢復任何 session |
| 圖表中文字體 | matplotlib 設定 Arial Unicode MS / Noto Sans CJK |

## 八、Web UI 設計（Streamlit）

- **左側 Sidebar**：session 列表（從 checkpoints.db 讀取）、新對話按鈕、刪除按鈕
- **右側主區域**：對話歷史（含圖表渲染）、suggestion chips（st.pills）、chat input
- **Session 管理**：每個瀏覽器 tab 為獨立 session，可切換歷史 session 恢復對話

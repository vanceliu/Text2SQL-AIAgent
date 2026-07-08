# Text2SQL AI Agent

基於地端模型（Gemma 4 E4B）打造的 Text2SQL AI Agent。使用者可用自然語言查詢超市銷售數據，Agent 透過 Function Calling 自動將自然語言轉換為 SQL 查詢，執行後根據語意與資料特徵智慧決定回覆格式（口語回覆、列表、圖表）。

## 技術架構

| 元件 | 選擇 |
|------|------|
| LLM | Gemma 4 E4B（透過 LM Studio 本地運行） |
| Agent 框架 | LangGraph 1.2.8（StateGraph + SqliteSaver） |
| Function Calling | 原生 FC，Pydantic schema 定義 tool |
| Observability | LangSmith Tracing |
| 資料庫 | SQLite |
| Checkpointer | SqliteSaver（對話歷史持久化） |
| 資料驗證 | Pydantic（CSV 匯入逐筆驗證） |
| 圖表 | matplotlib |
| Web UI | Streamlit |

## 專案結構

```
text2sql-agent/
├── setup.sh                        # 環境檢測與自動安裝腳本
├── requirements.txt                # Python 依賴（版本鎖定）
├── pytest.ini                      # 測試設定
├── .env.example                    # 環境變數範本
├── app.py                          # Streamlit Web UI（從根目錄執行）
├── data/
│   ├── SuperMarket Analysis.csv    # 原始資料（需自行下載）
│   ├── supermarket.db              # SQLite 資料庫（init_db 產生）
│   └── checkpoints.db             # 對話歷史持久化
├── src/
│   ├── db/
│   │   ├── init_db.py              # CSV → SQLite（Pydantic 驗證）
│   │   └── schema.py              # Schema 讀取工具
│   ├── agent/
│   │   ├── graph.py                # LangGraph 工作流定義
│   │   ├── state.py                # Agent State 定義
│   │   ├── llm.py                  # LLM 實例管理
│   │   └── nodes/
│   │       ├── intent_parser.py    # 意圖理解（FC + 摘要壓縮）
│   │       ├── schema_fetcher.py   # Schema 擷取
│   │       ├── text2sql.py         # SQL 生成（FC）
│   │       ├── sql_validator.py    # SQL 驗證（三重防護）
│   │       ├── sql_executor.py     # SQL 執行
│   │       ├── response_router.py  # 回覆格式決策（FC + fallback）
│   │       └── response_composer.py # 回覆組合（few-shot）
│   ├── visualization/
│   │   └── chart_generator.py      # 圖表生成
│   └── main.py                     # CLI 入口
├── tests/                          # 單元測試（96 tests）
├── output/                         # 生成的圖表
└── docs/
    └── architecture.md             # 架構設計文件
```

## 快速開始

### 前置條件

- macOS 或 Linux
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) 已安裝並下載 Gemma 4 E4B 模型

### 安裝步驟

```bash
# 1. 執行環境準備腳本
chmod +x setup.sh
./setup.sh

# 2. 下載資料集放到 data/ 目錄
# https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales

# 3. 設定環境變數
cp .env.example .env
# 編輯 .env 填入 LangSmith API Key（可選）

# 4. 啟動 LM Studio
#    - 載入 Gemma 4 E4B 模型
#    - 啟動 Local Server（預設 port 1234）

# 5. 初始化資料庫
source .venv/bin/activate
cd src && python db/init_db.py
```

### 使用方式

```bash
# CLI 模式（在 src/ 目錄下執行）
cd src && python main.py

# Web UI 模式（在根目錄執行）
streamlit run app.py
```

### 執行測試

```bash
# 在根目錄執行
source .venv/bin/activate
pytest tests/ -v
```

## 核心功能

### Function Calling

三個節點使用原生 Function Calling，以 Pydantic BaseModel 定義 tool schema：

- **Intent Parser** → `IntentResult`（意圖類型 + 實體提取）
- **Text2SQL** → `SQLGenerationResult`（SQL 語句 + 邏輯說明）
- **Response Router** → `ResponseFormatResult`（格式決策 + 理由）

### 多輪對話

- SqliteSaver checkpointer 持久化對話歷史至 `data/checkpoints.db`
- 程式重啟後可恢復任何歷史 session
- 支援上下文指代（如「那 Giza 呢？」）

### 對話歷史摘要壓縮

- 當累計 token 超過 6000 時自動觸發
- 舊歷史壓縮為 300 字摘要送給 LLM
- checkpointer 中的原始歷史不受影響

### 回覆格式智慧決策

| 判斷依據 | 口語 (text) | 列表 (table) | 圖表 (chart) |
|----------|-------------|--------------|--------------|
| 使用者語意 | 「總共」「平均」 | 「列出」「哪些」 | 「趨勢」「比較」「佔比」 |
| 結果筆數 | 1 筆 | 2-20 筆 | 2+ 筆分類統計 |
| 意圖類型 | aggregate | detail | trend/comparison |

### Web UI（Streamlit）

- 左側 Sidebar：session 列表、新增/刪除對話
- 右側主區域：對話歷史、suggestion chips、圖表渲染
- 切換歷史 session 可查看完整對話紀錄

## 示範場景

| 場景 | 問題範例 | 回覆格式 |
|------|---------|---------|
| 簡單聚合 | 「這個超市總共有多少筆交易？」 | 口語回覆 |
| 分組統計 | 「各分店的總銷售額是多少？」 | 口語 + 表格 |
| 趨勢比較 | 「比較各產品線的銷售佔比」 | 口語 + 圓餅圖 |
| 明細查詢 | 「列出所有評分高於 9 分的會員交易」 | 口語 + 表格 |
| 錯誤處理 | 「預測下個月的銷售額」 | 友善拒絕 + 建議 |

## 安全性

- SQL Injection 三重防護（sqlparse + 關鍵字掃描 + SQLite EXPLAIN）
- 僅允許 SELECT 操作
- 查詢逾時保護（30 秒）+ 結果筆數限制（100 筆）
- SQL 生成重試上限（3 次）

## LangSmith 整合

設定 `.env`：

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-api-key
LANGSMITH_PROJECT=text2sql-agent
```

可在 LangSmith Dashboard 觀察：
- 每個 LangGraph 節點的輸入/輸出
- Function Calling 的 tool_calls 詳情
- SQL 生成與重試過程
- 整體 Latency 與 Token 使用量

## 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| LM_STUDIO_BASE_URL | LM Studio API 位址 | http://localhost:1234/v1 |
| LM_STUDIO_MODEL | 模型名稱 | gemma-4-e4b |
| DATABASE_PATH | 銷售資料庫路徑 | data/supermarket.db |
| CSV_PATH | 原始 CSV 路徑 | data/SuperMarket Analysis.csv |
| CHECKPOINT_DB_PATH | 對話歷史資料庫路徑 | data/checkpoints.db |
| LANGSMITH_TRACING | 是否啟用 tracing | true |
| LANGSMITH_API_KEY | LangSmith API Key | — |
| LANGSMITH_PROJECT | LangSmith 專案名稱 | text2sql-agent |

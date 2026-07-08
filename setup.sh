#!/bin/bash
# =============================================================================
# Text2SQL AI Agent - 環境準備腳本 (macOS)
# =============================================================================
# 用途：自動檢測與安裝專案所需的執行環境
# 流程：檢測平台 → 檢查 LM Studio → 檢查模型 → 安裝 Python 依賴 → 初始化資料庫
# =============================================================================

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 輔助函式
print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# =============================================================================
# Step 1: 檢測系統平台
# =============================================================================
print_step "1/5 檢測系統平台..."

OS_TYPE=$(uname -s)
if [ "$OS_TYPE" = "Darwin" ]; then
    print_success "偵測到 macOS 系統"
elif [ "$OS_TYPE" = "Linux" ]; then
    print_success "偵測到 Linux 系統"
else
    print_error "不支援的作業系統: $OS_TYPE（僅支援 macOS / Linux）"
    exit 1
fi

# 檢查 CPU 架構
ARCH=$(uname -m)
print_success "CPU 架構: $ARCH"

# =============================================================================
# Step 2: 檢查 LM Studio 是否已安裝
# =============================================================================
print_step "2/5 檢查 LM Studio..."

LM_STUDIO_INSTALLED=false

if [ "$OS_TYPE" = "Darwin" ]; then
    if [ -d "/Applications/LM Studio.app" ]; then
        LM_STUDIO_INSTALLED=true
    fi
elif [ "$OS_TYPE" = "Linux" ]; then
    if command -v lms &> /dev/null || [ -f "$HOME/.local/bin/lms" ]; then
        LM_STUDIO_INSTALLED=true
    fi
fi

if [ "$LM_STUDIO_INSTALLED" = true ]; then
    print_success "LM Studio 已安裝"
else
    print_warning "LM Studio 尚未安裝"
    echo ""
    echo "  請手動下載安裝 LM Studio："
    echo "  https://lmstudio.ai/download"
    echo ""
    echo "  安裝完成後，請："
    echo "  1. 開啟 LM Studio"
    echo "  2. 下載 Gemma 4 E4B 模型"
    echo "  3. 啟動 Local Server（預設 port 1234）"
    echo ""
    read -p "  是否繼續執行後續步驟？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =============================================================================
# Step 3: 檢查 LM Studio Local Server 是否啟動
# =============================================================================
print_step "3/5 檢查 LM Studio Local Server..."

LM_STUDIO_URL="${LM_STUDIO_BASE_URL:-http://localhost:1234/v1}"

if curl -s "${LM_STUDIO_URL}/models" > /dev/null 2>&1; then
    print_success "LM Studio Local Server 運行中 (${LM_STUDIO_URL})"

    # 列出可用模型
    MODELS=$(curl -s "${LM_STUDIO_URL}/models" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['id'] for m in data.get('data', [])]
    if models:
        for m in models:
            print(f'    - {m}')
    else:
        print('    (無可用模型)')
except:
    print('    (無法解析模型列表)')
" 2>/dev/null)

    echo "  可用模型："
    echo "$MODELS"
else
    print_warning "LM Studio Local Server 未啟動"
    echo "  請開啟 LM Studio 並啟動 Local Server（預設 port 1234）"
    echo "  確保已載入 Gemma 4 E4B 模型"
fi

# =============================================================================
# Step 4: 安裝 Python 依賴
# =============================================================================
print_step "4/5 安裝 Python 依賴..."

# 檢查 Python 版本
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "未找到 Python，請先安裝 Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_success "Python 版本: $PYTHON_VERSION"

# 檢查 / 建立虛擬環境
if [ ! -d ".venv" ]; then
    print_warning "建立虛擬環境..."
    $PYTHON_CMD -m venv .venv
    print_success "虛擬環境已建立 (.venv)"
else
    print_success "虛擬環境已存在 (.venv)"
fi

# 啟動虛擬環境並安裝依賴
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

print_success "Python 依賴已安裝完成"

# =============================================================================
# Step 5: 初始化資料庫
# =============================================================================
print_step "5/5 初始化資料庫..."

# 檢查 CSV 資料檔是否存在
if [ -f "data/SuperMarket Analysis.csv" ]; then
    print_success "資料檔已存在: data/SuperMarket Analysis.csv"
else
    print_warning "資料檔不存在"
    echo "  請從 Kaggle 下載 SuperMarket Analysis.csv 並放到 data/ 目錄："
    echo "  https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales"
fi

# 執行資料庫初始化
if [ -f "data/SuperMarket Analysis.csv" ]; then
    $PYTHON_CMD src/db/init_db.py
    print_success "SQLite 資料庫初始化完成"
else
    print_warning "跳過資料庫初始化（缺少 CSV 檔案）"
fi

# =============================================================================
# 設定 .env 檔案
# =============================================================================
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "已建立 .env 設定檔（請填入 LangSmith API Key）"
else
    print_success ".env 設定檔已存在"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ 環境準備完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  使用方式："
echo "  1. 確認 LM Studio 已啟動 Local Server 並載入 Gemma 4 E4B"
echo "  2. 執行 Agent：source .venv/bin/activate && python src/main.py"
echo "  3. 執行 Web UI：source .venv/bin/activate && streamlit run app.py"
echo ""

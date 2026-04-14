#!/bin/bash
# Savorithm 一鍵安裝（學校電腦也能用，不需要 admin 密碼）
set -e

echo "🍜 Savorithm 安裝中..."
echo ""

# ── 1. 確認 Python ──
if ! command -v python3 &> /dev/null; then
    echo "⚠️  找不到 python3，嘗試安裝..."

    # 觸發 Xcode Command Line Tools（跳系統對話框，不需要 admin）
    if ! xcode-select -p &> /dev/null 2>&1; then
        echo "📦 正在安裝 Command Line Tools..."
        echo "   會跳出一個對話框，請按「安裝」"
        xcode-select --install 2>/dev/null || true
        echo ""
        echo "⏳ 請等對話框跑完（約 5 分鐘），裝好後重新執行："
        echo "   bash $(cd "$(dirname "$0")" && pwd)/install.sh"
        exit 0
    fi

    # CLT 裝了但還是沒有 python3 → 用 Homebrew
    if ! command -v python3 &> /dev/null; then
        if ! command -v brew &> /dev/null; then
            echo "📦 正在安裝 Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
            eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        fi
        echo "📦 正在安裝 Python..."
        brew install python
    fi
fi

PYTHON=$(command -v python3)
echo "✅ Python: $($PYTHON --version)"

# ── 2. Python 套件 ──
echo "📦 安裝 Python 套件..."
$PYTHON -m pip install --user mlx safetensors playwright matplotlib mcp huggingface_hub 2>/dev/null \
  || $PYTHON -m pip install --user --break-system-packages mlx safetensors playwright matplotlib mcp huggingface_hub 2>/dev/null \
  || $PYTHON -m pip install mlx safetensors playwright matplotlib mcp huggingface_hub

# ── 3. Chromium ──
echo "🌐 安裝 Chromium 瀏覽器..."
$PYTHON -m playwright install chromium

# ── 4. CKIP 模型 ──
if [ ! -d "ckip_models/ws" ]; then
    echo "🧠 下載 CKIP 模型（約 600MB）..."
    $PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download('FakeRockert543/ckip-mlx', local_dir='ckip_models')" \
      || hf download FakeRockert543/ckip-mlx --local-dir ckip_models \
      || $PYTHON -m huggingface_hub.cli download FakeRockert543/ckip-mlx --local-dir ckip_models
else
    echo "✅ CKIP 模型已存在"
fi

# ── 5. bert_mlx.py ──
if [ ! -f "bert_mlx.py" ]; then
    echo "📥 下載 bert_mlx.py..."
    curl -sL "https://raw.githubusercontent.com/FakeRocket543/ckip-mlx/main/bert_mlx.py" -o bert_mlx.py
fi

# ── 6. cwebp ──
if ! command -v cwebp &> /dev/null; then
    echo "🖼️  安裝 webp 工具..."
    if command -v brew &> /dev/null; then
        brew install webp
    else
        echo "⚠️  cwebp 未安裝（圖表會用 PNG 替代，不影響功能）"
    fi
fi

# ── 7. Output dir ──
mkdir -p output

# ── 8. 測試 ──
echo ""
echo "🧪 測試 MCP Server..."
timeout 5 $PYTHON -m savorithm mcp &>/dev/null &
PID=$!
sleep 2
if kill -0 $PID 2>/dev/null; then
    kill $PID 2>/dev/null
    echo "✅ savorithm MCP Server 正常"
else
    echo "⚠️  MCP Server 啟動失敗，請找老師"
fi

# ── 9. 產出 MCP 設定 ──
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
MCP_CONFIG='{
  "mcpServers": {
    "savorithm": {
      "command": "'$PYTHON'",
      "args": ["-m", "savorithm", "mcp"],
      "cwd": "'$INSTALL_DIR'"
    }
  }
}'

# 自動寫入 kiro-cli 設定
mkdir -p ~/.kiro
echo "$MCP_CONFIG" > ~/.kiro/mcp.json

echo ""
echo "═══════════════════════════════════════════"
echo "✅ 安裝完成！"
echo "═══════════════════════════════════════════"
echo ""
echo "📋 MCP 設定已自動寫入 ~/.kiro/mcp.json"
echo ""
echo "📋 如果你用 Kiro IDE，請把以下內容貼到 MCP 設定裡："
echo ""
echo "$MCP_CONFIG"
echo ""
echo "🚀 下一步："
echo "   1. 重啟 Kiro IDE（或 kiro-cli）"
echo "   2. 在聊天中輸入：「列出已分析的店家」"
echo "   3. 如果有回應，就成功了！"
echo ""
echo "   試試分析一家店："
echo "   「分析這家店 https://maps.app.goo.gl/你的連結」"
echo ""

# Savorithm 🍜 評論分析系統

Google Maps 評論爬取 → CKIP 中文斷詞 → BM25 關鍵詞分析 → HTML 報告生成

**⚠️ 本程式為 MCP 工具，請透過 AI Agent 操作，不要直接執行 Python。**

## 系統需求

- macOS（Apple Silicon — M1/M2/M3/M4）
- Python 3.10+
- 約 1GB 磁碟空間（模型 + 瀏覽器）

## 安裝（3 步驟）

### Step 1：Clone

```bash
git clone https://github.com/FakeRocket543/savorithm.git
cd savorithm
```

### Step 2：安裝

```bash
chmod +x install.sh
./install.sh
```

這會自動安裝：
- Python 套件（MLX、Playwright、MCP⋯）
- Chromium 瀏覽器（用於爬取 Google Maps）
- CKIP 中文斷詞模型（從 HuggingFace 下載）

### Step 3：設定 MCP

把 `mcp.json` 的內容加入你的 AI Agent 設定。

**Kiro CLI**：
```bash
# 編輯 ~/.kiro/settings/mcp.json，加入：
{
  "mcpServers": {
    "savorithm": {
      "command": "python3",
      "args": ["-m", "savorithm", "mcp"],
      "cwd": "/你的路徑/savorithm"
    }
  }
}
```

**opencode**：
```bash
# 編輯 .opencode/config.json 的 mcp 區段
```

## 使用方式

在 AI Agent 中直接對話：

```
你：分析這家店 https://maps.app.goo.gl/xxx
AI：（自動爬取 → 斷詞分析 → 生成圖表 → 寫報告）
```

```
你：比較這三家店
AI：（自動生成比較報告）
```

```
你：打包報告
AI：（生成 tar.gz，可上傳網頁伺服器）
```

## MCP 工具

| 工具 | 說明 |
|------|------|
| `resolve_url` | 解析 Google Maps 短網址 |
| `scrape_reviews` | 爬取所有評論（含精確日期） |
| `analyze_reviews` | CKIP 斷詞 + BM25 + 圖表 |
| `generate_report` | 生成 Tailwind CSS HTML 報告 |
| `package_report` | 打包成可上傳的 tar.gz |
| `list_stores` | 列出已分析的店家 |

## 專案結構

```
savorithm/
├── savorithm/           ← Python 套件
│   ├── __main__.py      ← 入口（MCP only）
│   ├── scraper.py       ← Google Maps 爬蟲
│   ├── analyzer.py      ← CKIP + BM25 + 圖表
│   └── mcp_server.py    ← MCP Server
├── ckip_models/         ← CKIP 模型（install.sh 自動下載）
├── bert_mlx.py          ← MLX BERT 推論引擎
├── output/              ← 分析結果
├── mcp.json             ← MCP 設定範例
├── install.sh           ← 一鍵安裝
└── README.md
```

## 技術細節

- **爬蟲**：Playwright + Chromium，攔截 Google Maps API 取得精確日期
- **斷詞**：CKIP BERT-base，Apple MLX 原生推論，比 PyTorch 快 5 倍
- **分析**：BM25 關鍵詞排序（unigram / 2-gram / 3-gram）
- **圖表**：matplotlib 生成 PNG → cwebp 轉 WEBP
- **報告**：Tailwind CSS HTML，行動裝置友善

## 模型來源

- CKIP BERT：[FakeRockert543/ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx)
- 原始模型：[中研院 CKIP Lab](https://ckip.iis.sinica.edu.tw/)

## 課程

大學新聞課程
- W07：AI CLI 工具
- W08：MCP 與 Agent 工作流
- W09：資料新聞實作

## License

教學用途 / GPL-3.0（CKIP 模型授權）

## ⚠️ 免責聲明

本專案僅供教學與學術研究用途（Educational and research purposes only）。

- 爬取 Google Maps 評論可能違反 Google 的服務條款（Terms of Service）
- 評論為用戶公開發表的內容，本工具不收集任何個人隱私資料（email、電話等）
- 使用者應自行評估法律風險，本專案作者不承擔任何因使用本工具而產生的法律責任
- 如需商業用途，請使用 [Google Places API](https://developers.google.com/maps/documentation/places/web-service) 等官方管道
- 類似的開源專案在 GitHub 上廣泛存在（如 [omkarcloud/google-maps-reviews-scraper](https://github.com/omkarcloud/google-maps-reviews-scraper) 等），本專案的技術手段並無特殊之處

**請負責任地使用本工具，尊重資料來源與平台規範。**

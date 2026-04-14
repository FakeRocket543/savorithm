"""
⚠️  Savorithm — 評論分析學習系統
═══════════════════════════════════════════════════════════════

本程式為教學用途，設計為 MCP (Model Context Protocol) 工具。

🚫 請勿直接以 python 執行本程式
✅ 請透過 MCP 連接你的 AI Agent（kiro-cli / opencode / Kiro IDE）

設定方式請參考 README.md 或執行：
    python -m savorithm --help

課程：世新大學 2026 網路新聞學
教師：FL

⚠️ 免責聲明：本工具僅供教學與學術研究用途。
   爬取 Google Maps 評論可能違反其服務條款，使用者應自行評估法律風險。
═══════════════════════════════════════════════════════════════
"""

import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        import asyncio
        from savorithm.mcp_server import serve
        asyncio.run(serve())
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        print("""
使用方式：

  1. 啟動 MCP Server：
     python -m savorithm mcp

  2. 在你的 AI Agent 設定中加入 MCP 連線（見 mcp.json）

  3. 開始對話：
     「分析這家店 https://maps.app.goo.gl/xxx」

可用的 MCP 工具：
  • resolve_url     — 解析 Google Maps 短網址
  • scrape_reviews   — 爬取所有評論
  • analyze_reviews  — CKIP 斷詞 + BM25 + 圖表
  • generate_report  — 生成 Tailwind HTML 報告
  • compare_stores   — 多店比較分析
  • package_report   — 打包成可上傳的 tar.gz
""")
    else:
        print(__doc__)
        print("💡 提示：執行 python -m savorithm --help 查看使用方式")
        print("💡 啟動 MCP：python -m savorithm mcp")
        sys.exit(1)

if __name__ == "__main__":
    main()

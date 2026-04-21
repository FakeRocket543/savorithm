"""Savorithm MCP Server — 評論分析工具"""
import asyncio, json, os, subprocess, shutil
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("savorithm")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


@app.list_tools()
async def list_tools():
    return [
        Tool(name="resolve_url", description="解析 Google Maps 短網址，回傳店名、地址、place_url、建議 slug", inputSchema={
            "type": "object", "properties": {"url": {"type": "string", "description": "Google Maps 短網址"}}, "required": ["url"]}),
        Tool(name="scrape_reviews", description="爬取指定店家的所有 Google Maps 評論", inputSchema={
            "type": "object", "properties": {
                "place_url": {"type": "string", "description": "完整的 Google Maps place URL"},
                "slug": {"type": "string", "description": "店家代號（英文，用於資料夾名稱）"}
            }, "required": ["place_url", "slug"]}),
        Tool(name="analyze_reviews", description="對已爬取的評論執行 CKIP 斷詞 + BM25 關鍵詞分析 + 生成圖表", inputSchema={
            "type": "object", "properties": {"slug": {"type": "string", "description": "店家代號"}}, "required": ["slug"]}),
        Tool(name="generate_report", description="根據分析結果生成 Tailwind CSS HTML 報告（含 WEBP 圖表）", inputSchema={
            "type": "object", "properties": {
                "slug": {"type": "string", "description": "店家代號"},
                "name": {"type": "string", "description": "店家名稱"},
                "analysis": {"type": "object", "description": "analyze_reviews 的回傳結果"}
            }, "required": ["slug", "name"]}),
        Tool(name="package_report", description="將報告打包成 tar.gz，可直接上傳網頁伺服器", inputSchema={
            "type": "object", "properties": {"slug": {"type": "string", "description": "店家代號"}}, "required": ["slug"]}),
        Tool(name="list_stores", description="列出所有已爬取的店家", inputSchema={"type": "object", "properties": {}}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "resolve_url":
            from savorithm.scraper import resolve_url
            result = await resolve_url(arguments["url"])
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "scrape_reviews":
            from savorithm.scraper import scrape_reviews
            result = await scrape_reviews(arguments["place_url"], arguments["slug"], OUTPUT_DIR)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "analyze_reviews":
            from savorithm.analyzer import analyze_reviews
            result = analyze_reviews(arguments["slug"], OUTPUT_DIR)
            # 加入 agent 行為指引
            if "data_summary" in result:
                result["_agent_instructions"] = (
                    "你剛完成了評論分析。請讀取 skills/coaching.md 的引導框架，"
                    "根據 data_summary 裡的數據，跟使用者進行 3-5 輪討論。\n\n"
                    "規則：\n"
                    "1. 先用 2-3 句話摘要分析結果（總評論數、平均星等、負評率）\n"
                    "2. 每次只問一個問題，等使用者回答再往下\n"
                    "3. 不要替使用者回答\n"
                    "4. 討論完後引導使用者寫文章\n\n"
                    "如果找不到 skills/coaching.md，就根據 data_summary 自行判斷該問什麼。"
                )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "generate_report":
            slug = arguments["slug"]
            name_str = arguments.get("name", slug)
            out = os.path.join(OUTPUT_DIR, slug)
            # Agent 會根據分析結果自行生成 HTML，這裡提供基本框架
            return [TextContent(type="text", text=json.dumps({
                "slug": slug,
                "output_dir": out,
                "webp_files": [f for f in os.listdir(out) if f.endswith(".webp")],
                "hint": f"請根據分析數據為「{name_str}」生成 Tailwind CSS HTML 報告，圖片使用相對路徑引用 .webp 檔案"
            }, ensure_ascii=False, indent=2))]

        elif name == "package_report":
            slug = arguments["slug"]
            out = os.path.join(OUTPUT_DIR, slug)
            tar_path = os.path.join(OUTPUT_DIR, f"{slug}_report.tar.gz")
            # Collect HTML + WEBP into a flat package
            pkg_dir = os.path.join(OUTPUT_DIR, f"_pkg_{slug}")
            os.makedirs(pkg_dir, exist_ok=True)
            for f in os.listdir(out):
                if f.endswith((".html", ".webp")):
                    src = os.path.join(out, f)
                    dst = os.path.join(pkg_dir, "index.html" if f == "report.html" else f)
                    shutil.copy2(src, dst)
            subprocess.run(["tar", "czf", tar_path, "-C", OUTPUT_DIR, f"_pkg_{slug}"], capture_output=True)
            shutil.rmtree(pkg_dir)
            return [TextContent(type="text", text=json.dumps({"path": tar_path, "size_kb": os.path.getsize(tar_path) // 1024}, ensure_ascii=False))]

        elif name == "list_stores":
            stores = []
            if os.path.exists(OUTPUT_DIR):
                for d in sorted(os.listdir(OUTPUT_DIR)):
                    rp = os.path.join(OUTPUT_DIR, d, "reviews.json")
                    if os.path.exists(rp):
                        with open(rp) as f:
                            revs = json.load(f)
                        stores.append({"slug": d, "reviews": len(revs)})
            return [TextContent(type="text", text=json.dumps(stores, ensure_ascii=False, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def serve():
    """啟動 MCP Server（stdio 模式）"""
    print("🚀 Savorithm MCP Server 啟動中...", flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

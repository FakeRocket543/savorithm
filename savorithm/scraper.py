"""Google Maps 評論爬取模組"""
import asyncio, json, os, re
from datetime import datetime
from collections import defaultdict
from playwright.async_api import async_playwright


async def resolve_url(short_url: str) -> dict:
    """解析 Google Maps 短網址，回傳店名、地址、完整 URL"""
    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        pg = await br.new_page(locale="zh-TW")
        await pg.goto(short_url, wait_until="domcontentloaded", timeout=60000)
        await pg.wait_for_timeout(8000)
        title = (await pg.title()).replace(" - Google 地圖", "").strip()
        url = pg.url
        addr = ""
        els = await pg.query_selector_all('button[aria-label*="地址"]')
        if els:
            addr = (await els[0].get_attribute("aria-label") or "").replace("地址: ", "").strip()
        await br.close()
    # Extract place_url (clean version without entry params)
    place_url = re.sub(r'\?entry=.*$', '', url)
    # Generate slug from title
    slug = re.sub(r'[^\w]', '_', title).strip('_').lower()[:30]
    return {"name": title, "address": addr, "place_url": place_url, "slug": slug}


async def scrape_reviews(place_url: str, slug: str, output_dir: str = "output") -> dict:
    """爬取指定店家的所有 Google Maps 評論"""
    review_url = place_url.replace("!4m6!3m5", "!4m8!3m7").replace("!16s", "!9m1!1b1!16s") + "?entry=ttu"
    out_dir = os.path.join(output_dir, slug)
    os.makedirs(out_dir, exist_ok=True)
    ugc = []

    async with async_playwright() as p:
        br = await p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        ctx = await br.new_context(
            locale="zh-TW", timezone_id="Asia/Taipei",
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        await ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        pg = await ctx.new_page()

        async def on_resp(r):
            if "listugcposts" in r.url and r.status == 200:
                try:
                    b = await r.text()
                    if len(b) > 1000:
                        ugc.append(b)
                except:
                    pass
        pg.on("response", on_resp)

        # Load place page
        await pg.goto(place_url, wait_until="domcontentloaded", timeout=60000)
        await pg.wait_for_timeout(6000)
        for t in ["全部接受", "Accept all"]:
            btn = pg.locator(f'button:has-text("{t}")')
            if await btn.count() > 0:
                await btn.first.click()
                await pg.wait_for_timeout(2000)
                break

        # Navigate to reviews
        await pg.goto(review_url, wait_until="domcontentloaded", timeout=60000)
        await pg.wait_for_timeout(8000)
        tab = pg.locator('button[role="tab"]:has-text("評論")')
        if await tab.count() > 0:
            await tab.first.click()
            await pg.wait_for_timeout(3000)

        # Sort by newest
        sm = pg.locator('button[aria-label="排序評論"],button[data-value="排序"]')
        if await sm.count() > 0:
            await sm.first.click()
            await pg.wait_for_timeout(1000)
            nw = pg.locator('div[data-index="1"],li[data-index="1"]')
            if await nw.count() > 0:
                await nw.first.click()
                await pg.wait_for_timeout(2000)

        # Find scrollable container
        sc = None
        for sel in ['div.m6QErb.DxyBCb.kA9KIf.dS8AEf', 'div.m6QErb.DxyBCb.kA9KIf', 'div.m6QErb.DxyBCb']:
            loc = pg.locator(sel)
            if await loc.count() > 0:
                sc = loc.first
                break
        if not sc:
            sc = pg.locator('div.m6QErb').first

        # Scroll to load all reviews
        prev = stall = 0
        while stall < 12:
            await sc.evaluate('el=>el.scrollTop=el.scrollHeight')
            await pg.wait_for_timeout(2000)
            now = await pg.locator('div.jftiEf').count()
            if now != prev:
                stall = 0
            else:
                stall += 1
                await pg.wait_for_timeout(1500)
            prev = now

        # Expand all "more" buttons
        await pg.evaluate('''()=>{
            document.querySelectorAll('button.w8nwRe.kyuRq,button.w8nwRe').forEach(b=>b.click());
            document.querySelectorAll('.jftiEf button,.jftiEf span[role="button"]').forEach(b=>{
                const t=b.textContent?.trim();
                if(t==="更多"||t==="More"||t==="查看原文"||t==="See original")b.click();
            });
        }''')
        await pg.wait_for_timeout(2000)

        # Extract review data
        data = await pg.evaluate(r'''()=>[...document.querySelectorAll('.jftiEf')].map(r=>{
            const s=r.querySelector('.kvMYJc'),rt=s?.getAttribute('aria-label')||'',rm=rt.match(/(\d)/);
            const bl=r.querySelectorAll('.wiI7pd');let tx='',tr='';
            if(bl.length>=2){tr=bl[0]?.textContent?.trim()||'';tx=bl[1]?.textContent?.trim()||'';}
            else if(bl.length===1){tx=bl[0]?.textContent?.trim()||'';}
            const ft=tx||tr;
            let lang='zh';
            if(/[\u3040-\u309F\u30A0-\u30FF]/.test(ft))lang='ja';
            else if(/[\uAC00-\uD7AF]/.test(ft))lang='ko';
            else if(/^[a-zA-ZÀ-ÿ\s\d.,!?'"()\-:;]+$/.test(ft.replace(/\n/g,'')))lang='en';
            return{name:r.querySelector('.d4r55')?.textContent?.trim()||'',
                rating:rm?parseInt(rm[1]):0,text:ft,
                text_translated:tr&&tr!==ft?tr:'',lang,
                date:r.querySelector('.rsqaWe')?.textContent?.trim()||'',
                owner_reply:r.querySelector('.CDe7pd')?.textContent?.trim()||'',
                photos:r.querySelectorAll('button.Tya61d').length};
        })''')
        await br.close()

    # Match absolute dates from API intercepts
    rd = []
    for body in ugc:
        cl = body[4:].strip() if body.startswith(")]}'") else body
        try:
            jd = json.loads(cl)
        except:
            continue
        arr = jd[2] if len(jd) > 2 and isinstance(jd[2], list) else []
        for rev in arr:
            try:
                meta = rev[0][1]
                ts = meta[2]
                nm = rev[0][1][4][5][0]
                if isinstance(ts, (int, float)) and ts > 1400000000000000:
                    rd.append({"name": nm or "", "date_abs": datetime.fromtimestamp(ts / 1000000).strftime("%Y-%m-%d"), "timestamp_us": int(ts)})
            except:
                continue

    nd = defaultdict(list)
    for r in rd:
        nd[r["name"]].append(r)
    for n in nd:
        nd[n].sort(key=lambda x: -x["timestamp_us"])
    ni = defaultdict(int)
    matched = 0
    for r in data:
        n = r["name"]
        if n in nd and ni[n] < len(nd[n]):
            r["date_abs"] = nd[n][ni[n]]["date_abs"]
            r["timestamp_us"] = nd[n][ni[n]]["timestamp_us"]
            ni[n] += 1
            matched += 1

    out_path = os.path.join(out_dir, "reviews.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"slug": slug, "count": len(data), "matched_dates": matched, "path": out_path}

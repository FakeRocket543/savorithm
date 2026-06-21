"""批量搜尋 Gelato 店 — Google Maps Places Discovery"""
import asyncio, json, os, re
from playwright.async_api import async_playwright

# 各國搜尋詞 + 城市座標
TARGETS = {
    "TW": {
        "keywords": ["gelato", "義式冰淇淋", "手工冰淇淋"],
        "cities": [
            ("台北", 25.033, 121.565),
            ("台中", 24.147, 120.673),
            ("台南", 22.999, 120.227),
            ("高雄", 22.627, 120.301),
        ],
        "lang": "zh-TW",
    },
    "JP": {
        "keywords": ["ジェラート", "gelato"],
        "cities": [
            ("東京", 35.682, 139.769),
            ("大阪", 34.686, 135.520),
            ("京都", 35.012, 135.768),
            ("福岡", 33.590, 130.402),
        ],
        "lang": "ja",
    },
    "KR": {
        "keywords": ["젤라또", "gelato"],
        "cities": [
            ("서울", 37.566, 126.978),
            ("부산", 35.180, 129.075),
            ("제주", 33.499, 126.531),
        ],
        "lang": "ko",
    },
    "CN": {
        "keywords": ["gelato", "意式冰淇淋"],
        "cities": [
            ("上海", 31.230, 121.474),
            ("北京", 39.904, 116.407),
            ("杭州", 30.274, 120.155),
        ],
        "lang": "zh-CN",
    },
    "TH": {
        "keywords": ["gelato", "เจลาโต้"],
        "cities": [("กรุงเทพ", 13.756, 100.502)],
        "lang": "th",
    },
    "MY": {
        "keywords": ["gelato"],
        "cities": [("Kuala Lumpur", 3.139, 101.687)],
        "lang": "ms",
    },
    "VN": {
        "keywords": ["gelato"],
        "cities": [("Ho Chi Minh", 10.823, 106.630)],
        "lang": "vi",
    },
    "IN": {
        "keywords": ["gelato"],
        "cities": [
            ("Mumbai", 19.076, 72.878),
            ("Delhi", 28.614, 77.209),
        ],
        "lang": "en",
    },
}


async def search_gelato_shops(country: str, max_per_city: int = 40) -> list[dict]:
    """Search Google Maps for gelato shops in a country. Returns list of {name, url, rating, reviews_count, city}."""
    cfg = TARGETS[country]
    results = []
    seen = set()

    async with async_playwright() as p:
        br = await p.chromium.launch(headless=True)
        ctx = await br.new_context(
            locale=cfg["lang"],
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        pg = await ctx.new_page()

        for city_name, lat, lng in cfg["cities"]:
            for kw in cfg["keywords"]:
                query = f"{kw} {city_name}"
                url = f"https://www.google.com/maps/search/{query}/@{lat},{lng},13z?hl={cfg['lang']}"
                await pg.goto(url, wait_until="domcontentloaded", timeout=60000)
                await pg.wait_for_timeout(5000)

                # Scroll the results panel to load more
                panel = await pg.query_selector('div[role="feed"]')
                if panel:
                    for _ in range(8):
                        await panel.evaluate("el => el.scrollTop = el.scrollHeight")
                        await pg.wait_for_timeout(2000)

                # Extract results
                items = await pg.query_selector_all('a[href*="/maps/place/"]')
                for item in items:
                    href = await item.get_attribute("href") or ""
                    label = await item.get_attribute("aria-label") or ""
                    if not href or href in seen:
                        continue
                    seen.add(href)

                    # Parse rating and review count from aria-label
                    rating_m = re.search(r'(\d[.,]\d)', label)
                    count_m = re.search(r'(\d[\d,]*)\s*(?:則評論|件の口コミ|개 리뷰|reviews|条评价)', label)

                    results.append({
                        "name": label.split("·")[0].strip() if "·" in label else label[:50],
                        "url": href,
                        "rating": float(rating_m.group(1).replace(",", ".")) if rating_m else None,
                        "reviews_count": int(count_m.group(1).replace(",", "")) if count_m else None,
                        "city": city_name,
                        "country": country,
                    })

                    if len([r for r in results if r["city"] == city_name]) >= max_per_city:
                        break

                await pg.wait_for_timeout(3000)  # rate limit

        await br.close()

    # Deduplicate and sort by review count
    results.sort(key=lambda x: -(x.get("reviews_count") or 0))
    return results


async def discover_all(countries: list[str] = None, max_per_city: int = 40, output: str = "discovery") -> str:
    """Run discovery for all specified countries. Save JSON."""
    if countries is None:
        countries = list(TARGETS.keys())

    os.makedirs(output, exist_ok=True)
    all_results = {}

    for country in countries:
        print(f"Discovering {country}...")
        shops = await search_gelato_shops(country, max_per_city)
        all_results[country] = shops
        # Save per-country
        with open(os.path.join(output, f"{country}.json"), "w") as f:
            json.dump(shops, f, ensure_ascii=False, indent=2)
        print(f"  {country}: {len(shops)} shops found")

    # Save combined
    out_path = os.path.join(output, "all_shops.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in all_results.values())
    print(f"\nTotal: {total} shops across {len(countries)} countries")
    return out_path

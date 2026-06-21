"""批量爬取 + 分析 pipeline"""
import asyncio, json, os, random
from savorithm.scraper import scrape_reviews
from savorithm.discovery import discover_all


async def batch_scrape(country: str, top_n: int = 20, output_dir: str = "output"):
    """Scrape reviews for top N shops in a country (by review count)."""
    discovery_path = f"discovery/{country}.json"
    if not os.path.exists(discovery_path):
        print(f"Run discovery first for {country}")
        return

    with open(discovery_path) as f:
        shops = json.load(f)

    # Filter: must have reviews, sort by count
    shops = [s for s in shops if s.get("reviews_count") and s["reviews_count"] >= 50]
    shops.sort(key=lambda x: -x["reviews_count"])
    shops = shops[:top_n]

    print(f"{country}: scraping top {len(shops)} shops")

    for i, shop in enumerate(shops):
        slug = f"{country}_{i:02d}_{shop['name'][:20].replace(' ','_')}"
        slug = "".join(c for c in slug if c.isalnum() or c == "_").lower()
        out_check = os.path.join(output_dir, slug, "reviews.json")

        if os.path.exists(out_check):
            print(f"  SKIP {slug} (already scraped)")
            continue

        print(f"  [{i+1}/{len(shops)}] {shop['name']} ({shop['reviews_count']} reviews)")
        try:
            await scrape_reviews(shop["url"], slug, output_dir)
        except Exception as e:
            print(f"    ERROR: {e}")

        # Random delay to avoid detection
        delay = random.uniform(10, 25)
        await asyncio.sleep(delay)

    print(f"Done: {country}")


async def batch_all(countries: list[str] = None, top_n: int = 15):
    """Full pipeline: discover → scrape top shops per country."""
    if countries is None:
        countries = ["TW", "JP", "KR"]  # Start with core 3

    # Step 1: Discovery
    await discover_all(countries)

    # Step 2: Batch scrape
    for country in countries:
        await batch_scrape(country, top_n=top_n)


if __name__ == "__main__":
    import sys
    countries = sys.argv[1:] if len(sys.argv) > 1 else ["TW", "JP", "KR"]
    asyncio.run(batch_all(countries))

"""跨語言甜度分析模組"""
import json, os, re
from collections import Counter

# 各語言「不太甜」等價表達
SWEET_LEXICON = {
    "zh": {
        "not_too_sweet": ["不會太甜", "不過甜", "甜度適中", "甜而不膩", "不死甜", "不太甜", "不甜膩"],
        "sweet_positive": ["香甜", "甜蜜", "甜度剛好", "微甜"],
        "sweet_negative": ["太甜", "偏甜", "死甜", "甜膩", "過甜", "好甜"],
    },
    "ja": {
        "not_too_sweet": ["甘さ控えめ", "甘すぎない", "くどくない", "さっぱり", "後味すっきり", "上品な甘さ", "甘さが控えめ"],
        "sweet_positive": ["甘くて美味しい", "ほんのり甘い", "優しい甘さ", "程よい甘さ"],
        "sweet_negative": ["甘すぎる", "くどい", "しつこい", "甘ったるい"],
    },
    "ko": {
        "not_too_sweet": ["덜 달아서", "안 달아서", "달지 않아서", "느끼하지 않", "깔끔하", "달지않아"],
        "sweet_positive": ["달콤하", "달달하"],
        "sweet_negative": ["너무 달", "달아서 느끼", "너무 단"],
    },
}


def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja"
    if re.search(r'[\uAC00-\uD7AF]', text):
        return "ko"
    if re.search(r'[\u4E00-\u9FFF]', text):
        return "zh"
    return "en"


def analyze_sweetness(reviews: list[dict]) -> dict:
    """Analyze sweetness discourse in a set of reviews.
    
    Each review should have: {text: str, rating: int}
    Returns frequency stats for sweetness-related terms.
    """
    if not reviews:
        return {}

    lang = detect_language(reviews[0].get("text", ""))
    lexicon = SWEET_LEXICON.get(lang, SWEET_LEXICON["zh"])

    stats = {
        "lang": lang,
        "total_reviews": len(reviews),
        "not_too_sweet": {"count": 0, "in_high_rating": 0, "examples": []},
        "sweet_positive": {"count": 0},
        "sweet_negative": {"count": 0},
        "matches": [],
    }

    for review in reviews:
        text = review.get("text", "")
        rating = review.get("rating", 0)

        for term in lexicon["not_too_sweet"]:
            if term in text:
                stats["not_too_sweet"]["count"] += 1
                if rating >= 4:
                    stats["not_too_sweet"]["in_high_rating"] += 1
                if len(stats["not_too_sweet"]["examples"]) < 10:
                    stats["not_too_sweet"]["examples"].append(text[:100])
                break

        for term in lexicon["sweet_positive"]:
            if term in text:
                stats["sweet_positive"]["count"] += 1
                break

        for term in lexicon["sweet_negative"]:
            if term in text:
                stats["sweet_negative"]["count"] += 1
                break

    # Rates
    n = stats["total_reviews"]
    stats["not_too_sweet"]["rate"] = round(stats["not_too_sweet"]["count"] / n * 100, 2) if n else 0
    stats["sweet_positive"]["rate"] = round(stats["sweet_positive"]["count"] / n * 100, 2) if n else 0
    stats["sweet_negative"]["rate"] = round(stats["sweet_negative"]["count"] / n * 100, 2) if n else 0

    return stats


def cross_national_report(output_dir: str = "output") -> dict:
    """Run sweetness analysis across all scraped shops, grouped by country."""
    report = {}

    for folder in sorted(os.listdir(output_dir)):
        reviews_path = os.path.join(output_dir, folder, "reviews.json")
        if not os.path.exists(reviews_path):
            continue

        country = folder.split("_")[0] if "_" in folder else "unknown"
        with open(reviews_path) as f:
            reviews = json.load(f)

        stats = analyze_sweetness(reviews)
        if not stats:
            continue

        if country not in report:
            report[country] = {"shops": [], "total_reviews": 0, "total_nts": 0}

        report[country]["shops"].append({
            "slug": folder,
            "reviews": stats["total_reviews"],
            "nts_rate": stats["not_too_sweet"]["rate"],
        })
        report[country]["total_reviews"] += stats["total_reviews"]
        report[country]["total_nts"] += stats["not_too_sweet"]["count"]

    # Compute country-level rates
    for country, data in report.items():
        n = data["total_reviews"]
        data["nts_rate_overall"] = round(data["total_nts"] / n * 100, 2) if n else 0
        data["shop_count"] = len(data["shops"])

    return report

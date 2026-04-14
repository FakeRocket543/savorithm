"""CKIP 斷詞 + BM25 + 圖表生成模組（純 Python MLX 版本）"""
import json, os, re, math, subprocess
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates

# ── CKIP MLX (Pure Python) ──
import mlx.core as mx

STOP_POS = {"PERIODCATEGORY","COMMACATEGORY","PARENTHESISCATEGORY","PAUSECATEGORY","SEMICOLONCATEGORY",
            "DASHCATEGORY","COLONCATEGORY","EXCLAMATIONCATEGORY","QUESTIONCATEGORY","ETCCATEGORY","WHITESPACE"}
STOPWORDS = {"有","是","的","了","在","都","也","就","很","會","不","我","他","她","你","這","那","到","被",
             "把","跟","和","與","或","但","而","所","以","因","為","又","再","還","才","只","從","讓","給",
             "對","向","比","等","著","過","得","地","…","～","~","＾＾","&","ＦＢ","ＩＧ"}
VERB_POS = {"VA","VAC","VB","VC","VCL","VD","VE","VF","VG","VH","VHC","VI","VJ","VK","VL","V_2","Nv"}

_FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
FP = FontProperties(fname=_FONT_PATH, size=10)
FP_T = FontProperties(fname=_FONT_PATH, size=14)
FP_L = FontProperties(fname=_FONT_PATH, size=9)

_ckip = None

def _get_model_dir():
    """找到 CKIP 模型目錄"""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckip_models"),  # savorithm/ckip_models
        os.path.expanduser("~/Python/ckip_mlx/models"),  # 開發者路徑
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "ws", "config.json")):
            return d
    raise FileNotFoundError("找不到 CKIP 模型。請執行：huggingface-cli download FakeRockert543/ckip-mlx --local-dir ckip_models")

def _get_bert_module():
    """動態載入 bert_mlx.py"""
    model_dir = _get_model_dir()
    # bert_mlx.py 可能在 ckip_models 的上層或同層
    for candidate in [
        os.path.join(os.path.dirname(model_dir), "bert_mlx.py"),
        os.path.join(model_dir, "..", "bert_mlx.py"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "bert_mlx.py"),
    ]:
        if os.path.exists(candidate):
            import importlib.util
            spec = importlib.util.spec_from_file_location("bert_mlx", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("找不到 bert_mlx.py")

def _load_ckip():
    global _ckip
    if _ckip:
        return _ckip

    model_dir = _get_model_dir()
    bert_mod = _get_bert_module()
    BertForTokenClassification = bert_mod.BertForTokenClassification

    # Load vocab
    vocab = {}
    vocab_path = os.path.join(model_dir, "ws", "vocab.txt")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(model_dir, "vocab.txt")
    with open(vocab_path) as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i

    models = {}
    for task in ["ws", "pos"]:
        # Prefer fp16 for speed/memory, fallback to fp32
        for variant in [f"{task}-fp16", task]:
            task_dir = os.path.join(model_dir, variant)
            if os.path.exists(os.path.join(task_dir, "config.json")):
                with open(os.path.join(task_dir, "config.json")) as f:
                    config = json.load(f)
                config["num_labels"] = len(config.get("id2label", {}))
                model = BertForTokenClassification(config)
                model.load_weights(os.path.join(task_dir, "weights.safetensors"))
                mx.eval(model.parameters())
                models[task] = {"model": model, "config": config}
                break

    _ckip = {"vocab": vocab, "models": models}
    return _ckip

def _tokenize(text, vocab, max_len=510):
    """單字切分 tokenization"""
    ids = [vocab.get("[CLS]", 101)]
    for ch in text[:max_len]:
        ids.append(vocab.get(ch, vocab.get("[UNK]", 100)))
    ids.append(vocab.get("[SEP]", 102))
    return ids

def _segment(text, ckip):
    """斷詞"""
    vocab = ckip["vocab"]
    ws_model = ckip["models"]["ws"]["model"]
    ids = _tokenize(text, vocab)
    input_ids = mx.array([ids])
    attention_mask = mx.array([[1] * len(ids)])
    logits = ws_model(input_ids, attention_mask=attention_mask)
    mx.eval(logits)
    preds = mx.argmax(logits, axis=-1).tolist()[0]
    words, cur = [], ""
    for i, ch in enumerate(text[:len(ids)-2]):
        p = preds[i + 1]
        if p == 0 and cur:
            words.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        words.append(cur)
    return words

def _pos_tag(text, words, ckip):
    """詞性標注"""
    vocab = ckip["vocab"]
    pos_model = ckip["models"]["pos"]["model"]
    config = ckip["models"]["pos"]["config"]
    id2label = config.get("id2label", {})
    ids = _tokenize(text, vocab)
    input_ids = mx.array([ids])
    attention_mask = mx.array([[1] * len(ids)])
    logits = pos_model(input_ids, attention_mask=attention_mask)
    mx.eval(logits)
    preds = mx.argmax(logits, axis=-1).tolist()[0]
    # Map char-level POS to word-level (take first char's POS)
    tags = []
    idx = 1  # skip [CLS]
    for w in words:
        tag = id2label.get(str(preds[idx]), "X") if idx < len(preds) - 1 else "X"
        tags.append(tag)
        idx += len(w)
    return tags

def _analyze_text(text, ckip):
    """斷詞 + 詞性標注"""
    words = _segment(text, ckip)
    tags = _pos_tag(text, words, ckip)
    return list(zip(words, tags))

def _keep(w, p):
    return p not in STOP_POS and w not in STOPWORDS and not (len(w) == 1 and p not in VERB_POS)

def _bm25(docs, k1=1.5, b=0.75):
    N = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    avgdl = sum(len(d) for d in docs) / N if N else 1
    sc = Counter()
    for d in docs:
        dl = len(d)
        tc = Counter(d)
        for t, f in tc.items():
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            sc[t] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return sc

def _bar_chart(path, labels, values, title, color):
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.3)))
    ax.barh(range(len(labels)), values, color=color, alpha=0.85, height=0.75)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=FP)
    ax.set_title(title, fontproperties=FP_T, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _to_webp(png_path):
    webp = png_path.replace(".png", ".webp")
    subprocess.run(["cwebp", "-q", "85", png_path, "-o", webp], capture_output=True)
    return webp


def analyze_reviews(slug: str, output_dir: str = "output") -> dict:
    """對指定店家執行 CKIP 分析 + BM25 + 圖表生成"""
    out = os.path.join(output_dir, slug)
    with open(os.path.join(out, "reviews.json"), encoding="utf-8") as f:
        reviews = json.load(f)

    for r in reviews:
        if r.get("owner_reply") and r.get("text"):
            rc = re.sub(r'^業主回應\s*', '', r["owner_reply"])
            rc = re.sub(r'^\d+\s*(年|個月|週|天)前', '', rc).strip()
            if rc and r["text"].strip().startswith(rc[:15]):
                r["text"] = r.get("text_translated", "") or ""

    texts = [r["text"] for r in reviews if r.get("text")]
    if len(texts) < 3:
        return {"slug": slug, "error": "too few texts", "count": len(texts)}

    ckip = _load_ckip()
    all_toks = []
    for text in texts:
        toks = _analyze_text(text, ckip)
        all_toks.append(toks)

    docs = [[w for w, p in toks if _keep(w, p)] for toks in all_toks]
    wf = Counter(w for d in docs for w in d)

    def save(fn, d):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    charts = []

    # Word freq
    save("word_freq.json", [{"word": w, "count": c} for w, c in wf.most_common(100)])
    top30 = wf.most_common(30)
    p = os.path.join(out, "word_freq.png")
    _bar_chart(p, [w for w, _ in top30][::-1], [c for _, c in top30][::-1], f"{slug} 詞頻 TOP 30", "#E67E22")
    charts.append(_to_webp(p))

    # BM25
    for n_g in [1, 2, 3]:
        label = "unigram" if n_g == 1 else f"{n_g}gram"
        ds = docs if n_g == 1 else [[" ".join(d[i:i + n_g]) for i in range(len(d) - n_g + 1)] for d in docs]
        bm = _bm25(ds)
        save(f"bm25_{label}.json", [{"term": t, "score": round(s, 2)} for t, s in sorted(bm.items(), key=lambda x: -x[1])[:100]])
        top25 = sorted(bm.items(), key=lambda x: -x[1])[:25]
        p = os.path.join(out, f"bm25_{label}.png")
        _bar_chart(p,
                   [re.sub(r'[^\w\s\u4e00-\u9fff]', '', t).strip() or '?' for t, _ in top25][::-1],
                   [s for _, s in top25][::-1],
                   f"{slug} BM25 {label}",
                   {"unigram": "#E74C3C", "2gram": "#8E44AD", "3gram": "#2E86C1"}[label])
        charts.append(_to_webp(p))

    # Negative word freq
    neg = [r for r in reviews if r.get("rating", 5) <= 2 and r.get("text")]
    neg_wf = []
    if neg:
        neg_text = " ".join(r["text"] for r in neg)
        neg_wf = sorted([(w, neg_text.count(w)) for w, _ in wf.most_common(100) if neg_text.count(w) > 0], key=lambda x: -x[1])[:50]
        save("neg_word_freq.json", [{"word": w, "count": c} for w, c in neg_wf])
        if len(neg_wf) > 5:
            p = os.path.join(out, "neg_word_freq.png")
            _bar_chart(p, [w for w, _ in neg_wf[:25]][::-1], [c for _, c in neg_wf[:25]][::-1],
                       f"{slug} 負評詞頻（{len(neg)} 則）", "#C0392B")
            charts.append(_to_webp(p))

    # Monthly chart
    dated = [r for r in reviews if r.get("date_abs") and r.get("text")]
    monthly = defaultdict(lambda: {"count": 0, "ratings": []})
    for r in dated:
        m = r["date_abs"][:7]
        monthly[m]["count"] += 1
        monthly[m]["ratings"].append(r["rating"])
    months = sorted(m for m in monthly if monthly[m]["count"] >= 2)
    if len(months) >= 3:
        save("monthly_word_freq.json", [{"month": m, "reviews": monthly[m]["count"],
             "avg_rating": round(sum(monthly[m]["ratings"]) / len(monthly[m]["ratings"]), 2)} for m in months])
        md = [datetime.strptime(m, "%Y-%m") for m in months]
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.bar(md, [monthly[m]["count"] for m in months], width=25, color="#3498DB", alpha=0.5, label="評論數")
        ax1.set_ylabel("評論數", fontproperties=FP)
        ax2 = ax1.twinx()
        ax2.plot(md, [sum(monthly[m]["ratings"]) / len(monthly[m]["ratings"]) for m in months], "o-", color="#E74C3C", linewidth=2, markersize=3, label="星等")
        ax2.set_ylim(1.0, 5.5)
        ax2.set_ylabel("平均星等", fontproperties=FP)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.set_title(f"{slug} 月度評論量與星等", fontproperties=FP_T, pad=10)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, prop=FP_L, loc="upper left")
        fig.tight_layout()
        p = os.path.join(out, "monthly_volume_rating.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        charts.append(_to_webp(p))

    total = len(reviews)
    avg = sum(r["rating"] for r in reviews) / total if total else 0
    ratings = Counter(r["rating"] for r in reviews)

    neg_rate = round(len(neg) / total * 100, 1) if total else 0
    neg_top5 = [w for w, _ in (neg_wf[:5] if neg else [])]
    top5 = [w for w, _ in wf.most_common(5)]

    # ── 教學引導（coaching） ──
    coaching_questions = []

    # 趨勢引導
    if len(months) >= 6:
        recent = months[-3:]
        early = months[:3]
        recent_avg = sum(sum(monthly[m]["ratings"]) / len(monthly[m]["ratings"]) for m in recent) / 3
        early_avg = sum(sum(monthly[m]["ratings"]) / len(monthly[m]["ratings"]) for m in early) / 3
        if recent_avg < early_avg - 0.2:
            coaching_questions.append(f"⚠️ 這家店的星等從早期 {early_avg:.1f} 下降到近期 {recent_avg:.1f}。請問使用者：你覺得可能的原因是什麼？是換了廚師、裝潢變了、還是競爭對手變多了？")
        elif recent_avg > early_avg + 0.2:
            coaching_questions.append(f"📈 這家店的星等從早期 {early_avg:.1f} 上升到近期 {recent_avg:.1f}。請問使用者：你覺得他們做對了什麼？")

    # 負評引導
    if neg_rate > 10:
        coaching_questions.append(f"🔴 負評率 {neg_rate}% 偏高。負評關鍵詞：{', '.join(neg_top5)}。請問使用者：這些負評可以分成幾類？（衛生、服務、口味、價格、等候）哪一類最嚴重？")
    elif neg_rate > 5:
        coaching_questions.append(f"🟡 負評率 {neg_rate}%。負評關鍵詞：{', '.join(neg_top5)}。請問使用者：如果你是店家，你會優先解決哪個問題？為什麼？")
    elif neg_rate < 2 and total > 50:
        coaching_questions.append(f"✨ 負評率只有 {neg_rate}%，非常低。請問使用者：你覺得這是真的很好，還是可能有灌水嫌疑？怎麼判斷？")

    # 關鍵詞引導
    hygiene_words = {'髒', '不乾淨', '蟑螂', '頭髮', '衛生', '噁心', '臭'}
    service_words = {'態度', '冷漠', '不理', '兇', '白眼', '服務差'}
    wait_words = {'排隊', '等', '久', '慢', '等候'}
    found_hygiene = [w for w in neg_top5 if w in hygiene_words]
    found_service = [w for w in neg_top5 if w in service_words]
    found_wait = [w for w in neg_top5 if w in wait_words]

    if found_hygiene:
        coaching_questions.append(f"🔴 衛生警訊！負評中出現：{', '.join(found_hygiene)}。這是最嚴重的問題，可能上新聞。請問使用者：這個問題是一直都有，還是最近才出現？（看月度趨勢）")
    if found_service:
        coaching_questions.append(f"🟠 服務態度問題：{', '.join(found_service)}。請問使用者：你自己去過這家店嗎？你的體驗跟評論一致嗎？")
    if found_wait:
        coaching_questions.append(f"🟡 等候問題：{', '.join(found_wait)}。這通常代表生意好但體驗差。請問使用者：如果你是店家，你會怎麼解決排隊問題？")

    # 差異化引導
    if 'CP值' in top5 or 'cp值' in top5 or 'CP' in top5:
        coaching_questions.append("💰 客人很在意 CP 值。請問使用者：這代表這家店的定價策略是什麼？如果要漲價，應該怎麼做才不會流失客人？")
    if '老闆' in top5 or '老闆娘' in top5:
        coaching_questions.append("👤 客人記住了老闆/老闆娘。這是人格化服務的優勢，但也是風險——老闆不在的時候怎麼辦？請問使用者：你覺得這家店應該怎麼把「老闆的魅力」轉化成「品牌的魅力」？")

    # 通用引導
    coaching_questions.append(f"📊 請問使用者：看完這份分析，你覺得這家店最大的優勢是什麼？最大的風險是什麼？如果你是店家，你明天會先做什麼？")
    coaching_questions.append("✍️ 請引導使用者用 #skill:商業文案 或 #skill:資料分析，根據以上數據寫一篇 300 字的分析報告。提醒他們要引用具體數據，不要只寫「還不錯」。")

    return {
        "slug": slug,
        "total": total,
        "with_text": len(texts),
        "avg_rating": round(avg, 2),
        "ratings": {k: ratings.get(k, 0) for k in [5, 4, 3, 2, 1]},
        "neg_rate_pct": neg_rate,
        "charts": charts,
        "top_words": [w for w, _ in wf.most_common(15)],
        "neg_top_words": neg_top5,
        "coaching": coaching_questions,
    }

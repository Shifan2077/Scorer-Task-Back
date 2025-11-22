# scoring.py
"""
Combined rule-based + optional semantic scoring engine for Nirmaan rubric.
- Semantic scoring will try (in order): fastembed, sentence-transformers.
- If neither is available, semantic scoring is skipped and the rule-based score is returned.
- LanguageTool (language-tool-python) and VADER are optional; code falls back to heuristics if missing.

Output:
{
  "overall_score": int,
  "overall_raw_points": float,
  "words": int,
  "sentences": int,
  "per_criteria": [...],
  "semantic_details": {...},
  "debug": {...}
}
"""

import re
import math
import warnings
from collections import Counter

# Optional NLP libs
HAS_FASTEMBED = False
HAS_ST = False
HAS_VADER = False
HAS_LANGTOOL = False

_fastembed = None
_st_model = None
_vader = None
_langtool = None

# Try fastembed first (lightweight embedding provider)
try:
    import fastembed as _fe  # fastembed API may vary by version
    HAS_FASTEMBED = True
    _fastembed = _fe
except Exception:
    HAS_FASTEMBED = False

# fallback to sentence-transformers if fastembed not available
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ST = True
    _st_model = None  # lazy load when needed
except Exception:
    HAS_ST = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
    _vader = SentimentIntensityAnalyzer()
except Exception:
    HAS_VADER = False

try:
    import language_tool_python
    HAS_LANGTOOL = True
    _langtool = language_tool_python.LanguageTool('en-US')
except Exception:
    HAS_LANGTOOL = False

# ---------------- Rubric config (encoded from screenshots) ----------------
RUBRIC = {
    "Content & Structure": {
        "weight": 40,
        "sub": {
            "Salutation": {"max": 5},
            "Keyword Presence": {"max": 20},
            "Flow": {"max": 5}
        }
    },
    "Speech Rate": {"weight": 10, "max": 10},
    "Language & Grammar": {"weight": 20, "sub": {"Grammar": {"max": 10}, "Vocabulary": {"max": 10}}},
    "Clarity": {"weight": 15, "sub": {"Filler Rate": {"max": 15}}},
    "Engagement": {"weight": 15, "max": 15}
}

KEYWORD_GROUPS = {
    "must_have": {
        "Name": ["name", "my name is", "i am", "i'm"],
        "Age": ["age", "years old"],
        "School/Class": ["school", "class", "grade"],
        "Family": ["family", "mother", "father", "parents"],
        "Hobbies/Interests": ["hobby", "hobbies", "like to", "interests"],
        "Goals": ["goal", "ambition", "dream"],
        "Unique": ["unique", "special", "one thing", "fun fact"]
    },
    "good_to_have": {
        "Origin": ["from", "i am from"],
        "Ambition": ["ambition", "aim", "aspire", "want to be"],
        "Achievements": ["won", "award", "prize", "achievement", "achieved"],
        "Strengths": ["strength", "strong", "skill"]
    }
}

FILLER_WORDS = set([
    "um","uh","like","you know","so","actually","basically","right","i mean",
    "well","kinda","sort of","okay","hmm","ah","eh","erm","okay"
])

SALUTATION_PATTERNS = {
    "excellent": [r"i am excited to introduce", r"i'm excited to introduce", r"i am excited to introduce myself"],
    "good": [r"good morning", r"good afternoon", r"good evening", r"hello everyone", r"hello everybody"],
    "normal": [r"\bhi\b", r"\bhello\b", r"\bhey\b"]
}

SPEECH_RATE_BANDS = [
    (161, float("inf"), 0),
    (141, 160, 2),
    (111, 140, 10),
    (81, 110, 6),
    (0, 80, 2)
]

VOCAB_BANDS = [
    (0.9, 1.0, 10),
    (0.7, 0.89, 8),
    (0.5, 0.69, 6),
    (0.3, 0.49, 4),
    (0.0, 0.29, 2)
]

GRAMMAR_BANDS = [
    (0.7, 0.89, 10),
    (0.5, 0.69, 8),
    (0.3, 0.49, 6),
    (0.0, 0.29, 4)
]

FILLER_BANDS = [
    (0, 3, 15),
    (4, 6, 12),
    (7, 9, 9),
    (10, 12, 6),
    (13, 1000, 3)
]

ENGAGEMENT_BANDS = [
    (0.9, 1.0, 15),
    (0.7, 0.89, 12),
    (0.5, 0.69, 9),
    (0.3, 0.49, 6),
    (0.0, 0.29, 3)
]

# ---------------- Utilities ----------------

def normalize_text(s: str):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return normalize_text(s).split()

def sentence_count(s: str):
    # crude sentence split
    return max(1, len([t for t in re.split(r'[.!?]+', s) if t.strip()]))

def compute_wpm(words: int, duration_sec: float):
    if duration_sec and duration_sec > 0:
        minutes = duration_sec / 60.0
        return words / minutes
    return None

def match_any_pattern(text: str, patterns):
    txt = (text or "").lower()
    for pat in patterns:
        if re.search(pat, txt):
            return True
    return False

def count_filler_words(tokens):
    token_text = " ".join(tokens)
    count = 0
    for fw in FILLER_WORDS:
        count += len(re.findall(r'\b' + re.escape(fw) + r'\b', token_text))
    return count

def ttr_score(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def grammar_error_rate(text):
    tokens = tokenize(text)
    W = len(tokens) or 1
    if HAS_LANGTOOL and _langtool:
        matches = _langtool.check(text)
        # Use total matches as rough error count (LanguageTool returns many types)
        errors = len(matches)
        errors_per_100 = (errors / W) * 100
        grammar_score = max(0.0, 1.0 - (errors_per_100 / 10.0))
        return max(0.0, min(1.0, grammar_score))
    else:
        # heuristic fallback
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        errors = 0
        for s in sentences:
            if s and not re.match(r'^[A-Z0-9]', s.strip()):
                errors += 1
        errors_per_100 = (errors / (W or 1)) * 100
        grammar_score = max(0.0, 1.0 - (errors_per_100 / 10.0))
        return max(0.0, min(1.0, grammar_score))

# ---------------- Semantic similarity wrapper ----------------

def _init_st_model():
    global _st_model
    try:
        if _st_model is None:
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        warnings.warn(f"Failed to initialize sentence-transformers: {e}")
        _st_model = None
    return _st_model

def semantic_similarity_score(text: str, target: str):
    """
    Try fastembed first, then sentence-transformers.
    Return float in [0,1] or None if no model available.
    """
    # fastembed attempt
    if HAS_FASTEMBED and _fastembed is not None:
        try:
            # NOTE: fastembed API may differ between versions.
            # Attempt common patterns; if they fail, we silently continue to ST fallback.
            if hasattr(_fastembed, "embed"):
                # fastembed.embed returns list of embeddings
                emb_t = _fastembed.embed([text])[0]
                emb_d = _fastembed.embed([target])[0]
            elif hasattr(_fastembed, "load_model"):
                # load a model instance and call encode or embed
                # user may need to adjust model name; we attempt default if available
                model = _fastembed.load_model()  # may raise if arguments required
                if hasattr(model, "encode"):
                    emb_t = model.encode([text])[0]
                    emb_d = model.encode([target])[0]
                elif hasattr(model, "embed"):
                    emb_t = model.embed([text])[0]
                    emb_d = model.embed([target])[0]
                else:
                    raise RuntimeError("fastembed model missing encode/embed")
            else:
                raise RuntimeError("fastembed import available but API unsupported")
            # compute cosine similarity manually
            dot = sum(a*b for a,b in zip(emb_t, emb_d))
            norm_t = math.sqrt(sum(a*a for a in emb_t)) or 1e-9
            norm_d = math.sqrt(sum(a*a for a in emb_d)) or 1e-9
            sim = dot / (norm_t * norm_d)
            sim = max(0.0, min(1.0, sim))
            return float(sim)
        except Exception:
            # fallback to sentence-transformers
            pass

    # fallback to sentence-transformers if available
    if HAS_ST:
        try:
            model = _init_st_model()
            if model is None:
                return None
            emb_t = model.encode([text], convert_to_numpy=True)
            emb_d = model.encode([target], convert_to_numpy=True)
            # sklearn cosine_similarity not imported globally to minimize deps; compute manually
            # But if sklearn available, we could use it. Use manual dot/norm to be safe.
            t = emb_t[0]
            d = emb_d[0]
            dot = float((t * d).sum())
            nt = float((t * t).sum()) ** 0.5 or 1e-9
            nd = float((d * d).sum()) ** 0.5 or 1e-9
            sim = dot / (nt * nd)
            sim = max(0.0, min(1.0, sim))
            return float(sim)
        except Exception:
            return None
    return None

# ---------------- Scoring helpers (rubric) ----------------

def score_salutation(text):
    txt = (text or "").lower()
    if match_any_pattern(txt, SALUTATION_PATTERNS["excellent"]):
        return 5, "excellent"
    if match_any_pattern(txt, SALUTATION_PATTERNS["good"]):
        return 4, "good"
    if match_any_pattern(txt, SALUTATION_PATTERNS["normal"]):
        return 2, "normal"
    return 0, "none"

def score_keyword_groups(text):
    txt = normalize_text(text)
    must_found = {}
    good_found = {}
    for k, kw_list in KEYWORD_GROUPS["must_have"].items():
        present = any(kw in txt for kw in kw_list)
        must_found[k] = present
    for k, kw_list in KEYWORD_GROUPS["good_to_have"].items():
        present = any(kw in txt for kw in kw_list)
        good_found[k] = present
    must_count = sum(1 for v in must_found.values() if v)
    must_score = min(20, must_count * 4)
    good_count = sum(1 for v in good_found.values() if v)
    good_score = min(10, good_count * 2)
    combined = min(20, must_score + good_score)
    details = {"must_found": must_found, "good_found": good_found, "must_score": must_score, "good_score": good_score, "combined": combined}
    return combined, details

def score_flow(text):
    txt = (text or "").lower()
    idx_sal = None
    idx_basic = None
    idx_optional = None
    idx_close = None
    for pat in SALUTATION_PATTERNS["good"] + SALUTATION_PATTERNS["normal"] + SALUTATION_PATTERNS["excellent"]:
        m = re.search(pat, txt)
        if m:
            idx_sal = m.start()
            break
    for kw in ["my name", "i am", "i'm", "age", "years old", "school", "class", "grade"]:
        m = re.search(kw, txt)
        if m:
            idx_basic = m.start()
            break
    for kw in ["hobby", "hobbies", "interest", "family", "my favorite", "one thing", "fun fact"]:
        m = re.search(kw, txt)
        if m:
            idx_optional = m.start()
            break
    m = re.search(r"thank you|thanks for listening|thanks", txt)
    if m:
        idx_close = m.start()
    indices = [i for i in [idx_sal, idx_basic, idx_optional, idx_close] if i is not None]
    order_ok = (len(indices) >= 2 and indices == sorted(indices))
    return (5 if order_ok else 0), {"idx_sal": idx_sal, "idx_basic": idx_basic, "idx_optional": idx_optional, "idx_close": idx_close, "order_ok": order_ok}

def score_speech_rate(words, duration_sec):
    wpm = compute_wpm(words, duration_sec) if duration_sec else None
    if wpm is None:
        return 10, {"wpm": None}
    for low, high, pts in SPEECH_RATE_BANDS:
        if low <= wpm <= high:
            return pts, {"wpm": round(wpm, 1)}
    return 0, {"wpm": round(wpm, 1)}

def score_grammar(text):
    gscore = grammar_error_rate(text)
    pts = 4
    for low, high, p in GRAMMAR_BANDS:
        if low <= gscore <= high:
            pts = p
            break
    return pts, {"grammar_score": round(gscore, 3)}

def score_vocabulary(tokens):
    ttr = ttr_score(tokens)
    pts = 2
    for low, high, p in VOCAB_BANDS:
        if low <= ttr <= high:
            pts = p
            break
    return pts, {"ttr": round(ttr, 3)}

def score_filler_rate(tokens):
    fill_count = count_filler_words(tokens)
    W = max(1, len(tokens))
    filler_percent = (fill_count / W) * 100
    pts = 3
    for low, high, p in FILLER_BANDS:
        if low <= filler_percent <= high:
            pts = p
            break
    return pts, {"filler_count": fill_count, "filler_percent": round(filler_percent, 2)}

def score_engagement(text):
    if HAS_VADER and _vader:
        vs = _vader.polarity_scores(text)
        pos = vs.get("pos", 0.0)
        pts = 3
        for low, high, p in ENGAGEMENT_BANDS:
            if low <= pos <= high:
                pts = p
                break
        return pts, {"vader": vs}
    else:
        pos_words = ["excited", "happy", "grateful", "confident", "enthusiastic", "love"]
        txt = (text or "").lower()
        count = sum(1 for w in pos_words if w in txt)
        if count >= 2:
            return 15, {"pos_matches": count}
        if count == 1:
            return 9, {"pos_matches": count}
        return 3, {"pos_matches": count}

# ---------------- Main pipeline ----------------

def score_transcript(transcript_text, duration_sec=None, alpha=0.5):
    text = transcript_text or ""
    tokens = tokenize(text)
    W = len(tokens)
    sentences = sentence_count(text)

    sal_score, sal_meta = score_salutation(text)
    kw_score, kw_meta = score_keyword_groups(text)
    flow_score, flow_meta = score_flow(text)
    content_subtotal = sal_score + kw_score + flow_score  # 0-30
    content_score_scaled = (content_subtotal / 30.0) * RUBRIC["Content & Structure"]["weight"]

    sp_pts, sp_meta = score_speech_rate(W, duration_sec)
    speech_score_scaled = (sp_pts / RUBRIC["Speech Rate"]["max"]) * RUBRIC["Speech Rate"]["weight"]

    gram_pts, gram_meta = score_grammar(text)
    vocab_pts, vocab_meta = score_vocabulary(tokens)
    lang_pts_total = gram_pts + vocab_pts
    language_score_scaled = (lang_pts_total / 20.0) * RUBRIC["Language & Grammar"]["weight"]

    fill_pts, fill_meta = score_filler_rate(tokens)
    clarity_score_scaled = (fill_pts / RUBRIC["Clarity"]["sub"]["Filler Rate"]["max"]) * RUBRIC["Clarity"]["weight"]

    eng_pts, eng_meta = score_engagement(text)
    engagement_score_scaled = (eng_pts / RUBRIC["Engagement"]["max"]) * RUBRIC["Engagement"]["weight"]

    overall_raw = content_score_scaled + speech_score_scaled + language_score_scaled + clarity_score_scaled + engagement_score_scaled
    overall_score = int(round(overall_raw))

    # Semantic nudges (optional)
    semantic_details = {}
    if alpha is not None and 0 <= alpha <= 1:
        # Compute semantic scores for a few category descriptions
        descriptions = {
            "content": "Introduce yourself with name, age, school, family, hobbies and close the introduction.",
            "engagement": "Speak with positive and enthusiastic tone.",
            "language": "Use correct grammar and a varied vocabulary."
        }
        sem_scores = {}
        for k, desc in descriptions.items():
            s = semantic_similarity_score(text, desc)
            if s is not None:
                sem_scores[k] = round(s, 3)
        if sem_scores:
            # Nudge content & engagement slightly using similarity and alpha
            if "content" in sem_scores:
                sim = sem_scores["content"]
                adjust = (sim - 0.5) * 4 * (1 - alpha)
                content_score_scaled = max(0, min(RUBRIC["Content & Structure"]["weight"], content_score_scaled + adjust))
            if "engagement" in sem_scores:
                sim = sem_scores["engagement"]
                adjust = (sim - 0.5) * 2 * (1 - alpha)
                engagement_score_scaled = max(0, min(RUBRIC["Engagement"]["weight"], engagement_score_scaled + adjust))
            semantic_details = sem_scores
            overall_raw = content_score_scaled + speech_score_scaled + language_score_scaled + clarity_score_scaled + engagement_score_scaled
            overall_score = int(round(overall_raw))

    per_criteria = [
        {
            "category": "Content & Structure",
            "sub": [
                {"name": "Salutation", "score": sal_score, "max": 5, "meta": sal_meta},
                {"name": "Keyword Presence", "score": kw_score, "max": 20, "meta": kw_meta},
                {"name": "Flow", "score": flow_score, "max": 5, "meta": flow_meta}
            ],
            "category_score_raw": round(content_subtotal, 2),
            "category_score_scaled": round(content_score_scaled, 2),
            "category_weight": RUBRIC["Content & Structure"]["weight"]
        },
        {
            "category": "Speech Rate",
            "sub": [{"name": "WPM", "score": sp_pts, "max": 10, "meta": sp_meta}],
            "category_score_scaled": round(speech_score_scaled, 2),
            "category_weight": RUBRIC["Speech Rate"]["weight"]
        },
        {
            "category": "Language & Grammar",
            "sub": [
                {"name": "Grammar", "score": gram_pts, "max": 10, "meta": gram_meta},
                {"name": "Vocabulary", "score": vocab_pts, "max": 10, "meta": vocab_meta}
            ],
            "category_score_scaled": round(language_score_scaled, 2),
            "category_weight": RUBRIC["Language & Grammar"]["weight"]
        },
        {
            "category": "Clarity",
            "sub": [{"name": "Filler Rate", "score": fill_pts, "max": 15, "meta": fill_meta}],
            "category_score_scaled": round(clarity_score_scaled, 2),
            "category_weight": RUBRIC["Clarity"]["weight"]
        },
        {
            "category": "Engagement",
            "sub": [{"name": "Sentiment/Positivity", "score": eng_pts, "max": 15, "meta": eng_meta}],
            "category_score_scaled": round(engagement_score_scaled, 2),
            "category_weight": RUBRIC["Engagement"]["weight"]
        }
    ]

    output = {
        "overall_score": overall_score,
        "overall_raw_points": round(overall_raw, 3),
        "words": W,
        "sentences": sentences,
        "per_criteria": per_criteria,
        "semantic_details": semantic_details,
        "debug": {
            "content_subtotal": content_subtotal,
            "speech_pts": sp_pts,
            "lang_pts_total": lang_pts_total,
            "fill_pts": fill_pts,
            "eng_pts": eng_pts
        }
    }
    return output

# Quick manual test
if __name__ == "__main__":
    sample = "Good morning everyone. My name is Aisha, and I am 14 years old. I study in class 9. I love reading and badminton. Thank you."
    print(score_transcript(sample, duration_sec=60, alpha=0.5))

# =====================================================================
# Muraqam — Source-manner clustering + validation
# Goal: discover latent "source" clusters that punctuate differently,
#       then verify (chi-square) that punctuation-class distribution
#       actually differs across clusters. If it doesn't, drop the idea.
#
# Design notes:
#   - Cluster on RAW pre-cleaning text (before normalization strips style).
#   - Three representations compared: style-features, char-tfidf,
#     fine-tuned-model embeddings. The chi-square effect size decides.
#   - HDBSCAN => no preset k, noise points allowed.
# =====================================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
import hdbscan

# ---------------------------------------------------------------------
# CONFIG — adapt to your notebook
# ---------------------------------------------------------------------
RAW_TEXT_COL   = "text"          # raw, PRE-cleaning text per row
LABEL_COL      = "punct_label"   # per-row dominant punct class, or see below
ID_COL         = "id"
MIN_CLUSTER    = 50
MIN_SAMPLES    = 10
SVD_DIM        = 50
RANDOM_STATE   = 0

# ---------------------------------------------------------------------
# 0. Load RAW text (before Claude's cleaning touched it)
# ---------------------------------------------------------------------
# train = pd.read_csv("raw_train.csv")   # <-- your raw file
# texts = train[RAW_TEXT_COL].fillna("").tolist()

# ---------------------------------------------------------------------
# 1. STYLE FEATURES  (interpretable; strong baseline for *manner*)
#    Surface statistics that separate sources regardless of topic.
# ---------------------------------------------------------------------
AR_CONNECTIVES = ["و", "ثم", "أو", "لكن", "بل", "حتى", "إذ", "لأن", "كما"]
DIACRITICS = set("ًٌٍَُِّْ")

def style_features(t: str) -> dict:
    n = max(len(t), 1)
    words = t.split()
    wn = max(len(words), 1)
    # sentence-ish segmentation on whitespace runs / existing breaks
    diac = sum(ch in DIACRITICS for ch in t)
    tatweel = t.count("ـ")
    latin = sum(("a" <= ch.lower() <= "z") for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    conj_starts = sum(w.startswith(tuple(AR_CONNECTIVES)) for w in words)
    return {
        "len_chars":     len(t),
        "n_words":       len(words),
        "avg_word_len":  n / wn,
        "diac_rate":     diac / n,
        "tatweel_rate":  tatweel / n,
        "latin_rate":    latin / n,
        "digit_rate":    digits / n,
        "conj_start_rt": conj_starts / wn,
        "has_honorific": float("ﷺ" in t),
        "ws_rate":       t.count(" ") / n,
    }

def build_style_matrix(texts):
    df = pd.DataFrame([style_features(t) for t in texts])
    return StandardScaler().fit_transform(df.values), df.columns.tolist()

# ---------------------------------------------------------------------
# 2. CHAR-TFIDF  (surface n-grams; good manner signal, topic-light)
# ---------------------------------------------------------------------
def build_char_tfidf(texts, n_components=SVD_DIM):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                            max_features=20000)
    X = normalize(tfidf.fit_transform(texts))
    Xr = TruncatedSVD(n_components=n_components,
                      random_state=RANDOM_STATE).fit_transform(X)
    return Xr

# ---------------------------------------------------------------------
# 3. FINE-TUNED MODEL EMBEDDINGS
#    IMPORTANT: use YOUR trained punctuation model, not vanilla arabert.
#    Vanilla arabert clusters by topic (wrong axis). A model fine-tuned
#    on the punctuation task carries *manner* in its pooled features.
#    Mean-pool over tokens (better than [CLS] for style).
# ---------------------------------------------------------------------
def build_model_embeddings(texts, model, tokenizer, device="cuda",
                           batch_size=32, max_len=448, n_components=SVD_DIM):
    import torch
    model.eval().to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1]                 # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs.append(pooled.cpu().numpy())
    X = normalize(np.concatenate(embs, 0))
    if X.shape[1] > n_components:
        X = TruncatedSVD(n_components=n_components,
                         random_state=RANDOM_STATE).fit_transform(X)
    return X

# ---------------------------------------------------------------------
# 4. CLUSTER  (HDBSCAN — no preset k)
# ---------------------------------------------------------------------
def cluster(X):
    clu = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER,
                          min_samples=MIN_SAMPLES,
                          metric="euclidean")
    labels = clu.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  clusters={n_clusters}  noise={n_noise}/{len(labels)}")
    return labels

# ---------------------------------------------------------------------
# 5. VALIDATE — does punctuation manner actually differ by cluster?
#    This is the decision gate. Big Cramér's V => the trick is real.
# ---------------------------------------------------------------------
def validate_clusters(labels, punct_labels):
    """punct_labels: per-row punctuation class (str/int).
       For multi-punct rows, pass the dominant class per row."""
    mask = labels != -1
    ct = pd.crosstab(np.asarray(labels)[mask],
                     np.asarray(punct_labels)[mask])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    print(f"  chi2={chi2:,.1f}  p={p:.2e}  Cramér's V={cramers_v:.3f}")
    # Row-normalized class distribution per cluster — eyeball the split
    dist = ct.div(ct.sum(1), axis=0).round(3)
    print("  per-cluster class distribution:\n", dist)
    # Which classes differ most across clusters (max-min spread)
    spread = (dist.max(0) - dist.min(0)).sort_values(ascending=False)
    print("\n  classes most source-dependent (spread):\n", spread.head(8))
    return cramers_v, dist

# ---------------------------------------------------------------------
# 6. RUN — compare representations, let chi-square pick the winner
# ---------------------------------------------------------------------
def run(texts, punct_labels, model=None, tokenizer=None):
    results = {}

    print("[STYLE FEATURES]")
    Xs, _ = build_style_matrix(texts)
    ls = cluster(Xs)
    results["style"] = (ls, *validate_clusters(ls, punct_labels))

    print("\n[CHAR-TFIDF]")
    Xt = build_char_tfidf(texts)
    lt = cluster(Xt)
    results["tfidf"] = (lt, *validate_clusters(lt, punct_labels))

    if model is not None:
        print("\n[FINE-TUNED MODEL EMB]")
        Xe = build_model_embeddings(texts, model, tokenizer)
        le = cluster(Xe)
        results["model_emb"] = (le, *validate_clusters(le, punct_labels))

    print("\n=== Cramér's V by representation (higher = stronger source manner) ===")
    for k, v in results.items():
        print(f"  {k:12s}  V={v[1]:.3f}")
    return results

# ---------------------------------------------------------------------
# 7. EXPLOIT — once a winner is chosen, use cluster id downstream
# ---------------------------------------------------------------------
# a) Per-cluster threshold tuning: refit your per-class thresholds
#    within each cluster's validation slice.
# b) Cluster-conditioned input: prepend a special token per cluster,
#    e.g.  text = f"[SRC_{cid}] " + text   (add tokens to tokenizer).
# c) Per-cluster logit calibration (temperature/Platt) before ensembling.
#
# GUARD: verify clusters are stable across train/test. Assign test rows
# to nearest train cluster centroid; if the test mix is different,
# per-cluster priors overfit — fall back to global.
# ---------------------------------------------------------------------
# results = run(texts, punct_labels, model=my_punct_model, tokenizer=tok)

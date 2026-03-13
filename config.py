"""
Configuration centrale du projet ECB Sentiment.
Modifier ici les paramètres globaux.
"""

# ── URLs BCE ──────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.ecb.europa.eu"
PRESSCONF_INDEX = "/press/pressconf/html/index.en.html"

# ── Période d'analyse (papier : juin 1998 – sept 2022) ───────────────────────
DATE_START = "1998-06-09"
DATE_END   = "2022-09-08"

# ── Paramètres du bag-of-words (papier section 3.1) ──────────────────────────
WINDOW_SIZE = 10          # mots avant/après le mot-clé conservés
MIN_WORD_COUNT = 3        # longueur minimale d'un token après stemming

# ── Mots-clés cibles ──────────────────────────────────────────────────────────
TARGET_WORDS = {
    "inflation":    {"type": "unigram"},
    "interest rate": {"type": "bigram", "forms": ["rate", "rates"]},
    "unemployment": {"type": "unigram"},
}

# ── Chemins fichiers ──────────────────────────────────────────────────────────
PATHS = {
    "raw":        "data/raw/pressconf/",
    "processed":  "data/processed/transcripts/",
    "url_list":   "data/processed/pressconf_urls.csv",
    "warin_csv":  "data/raw/Warin_Sanger_ECB.csv",
    "output":     "data/output/",
    "scores":     "data/output/sentiment_scores.csv",
}

# ── Rate limiter (scraping poli) ──────────────────────────────────────────────
RATE_LIMIT = {
    "min_delay": 2.0,   # secondes min entre deux requêtes
    "max_delay": 5.0,   # secondes max entre deux requêtes
    "max_retries": 3,
    "backoff_factor": 2.0,  # délai × 2 à chaque retry
}

# ── Validation (valeurs attendues d'après le papier) ─────────────────────────
EXPECTED_N_CONFERENCES = 266
EXPECTED_MEAN_WORD_COUNT = 5719

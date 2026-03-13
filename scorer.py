"""
Calcul des scores de sentiment hawkish/dovish.
Implémente exactement les formules (1) et (2) du papier (section 3.2).

Formule (1) — score hawkish d'un document :
    t_hawkish = Σ_i Σ_j (s_ij * W_ij) / N_i

Formule (2) — sentiment final :
    t_sentiment = t_hawkish - t_dovish

où :
  - s_ij = 1 si le mot j est hawkish/dovish dans la phrase i
  - W_ij = nombre d'occurrences du mot-clé cible dans la phrase i
  - N_i  = nombre total de mots dans la fenêtre de la phrase i
"""

import pandas as pd
from cleaner import extract_windows
from dictionary import get_words, count_pattern_matches
from logger import setup_logger

logger = setup_logger()


def _score_window(window: dict, return_details: bool = False) -> dict:
    """
    Calcule les scores hawkish et dovish pour une fenêtre autour d'un mot-clé.

    Implémente la formule (1) du papier pour une phrase individuelle.
    """
    keyword  = window["keyword"]
    raw_tokens = window.get("raw_tokens", [])
    n_tokens = window["n_tokens"]

    if n_tokens == 0:
        base = {"hawkish": 0.0, "dovish": 0.0}
        if return_details:
            base["details"] = {
                "keyword": keyword,
                "n_tokens": n_tokens,
                "keyword_count": int(window.get("keyword_count", 1)),
                "W_ij": int(window.get("keyword_count", 1)),
                "hawk_matches": 0,
                "dove_matches": 0,
                "raw_tokens": window.get("raw_tokens", []),
                "filtered_tokens": window.get("tokens", []),
                "hawkish_patterns": [],
                "dovish_patterns": [],
            }
        return base

    hawkish_words = get_words(keyword, "hawkish")
    dovish_words  = get_words(keyword, "dovish")

    # W_ij du papier = nombre d'occurrences du mot-clé dans la phrase i.
    # Ici on génère une fenêtre par occurrence, donc on normalise le poids
    # pour éviter une amplification quadratique (k fenêtres * W_ij=k).
    keyword_count = max(1, int(window.get("keyword_count", 1)))
    occurrences_in_sentence = max(1, int(window.get("occurrences_in_sentence", 1)))
    W_ij = keyword_count / occurrences_in_sentence

    hawk_matches = count_pattern_matches(raw_tokens, hawkish_words)
    dove_matches = count_pattern_matches(raw_tokens, dovish_words)

    score_h = (hawk_matches * W_ij) / n_tokens
    score_d = (dove_matches * W_ij) / n_tokens

    result = {"hawkish": score_h, "dovish": score_d}
    if return_details:
        result["details"] = {
            "keyword": keyword,
            "n_tokens": n_tokens,
            "keyword_count": keyword_count,
            "occurrences_in_sentence": occurrences_in_sentence,
            "W_ij": W_ij,
            "hawk_matches": hawk_matches,
            "dove_matches": dove_matches,
            "raw_tokens": raw_tokens,
            "filtered_tokens": window.get("tokens", []),
            "hawkish_patterns": hawkish_words,
            "dovish_patterns": dovish_words,
            "window_hawkish_score": score_h,
            "window_dovish_score": score_d,
            "window_sentiment": score_h - score_d,
        }
    return result


def score_document(
    text: str,
    return_details: bool = False,
):
    """
    Calcule les scores de sentiment pour l'ensemble d'un document.
    
    Retourne un dict avec :
      - "inflation"     : score sentiment inflation
      - "interest_rate" : score sentiment taux d'intérêt
      - "unemployment"  : score sentiment chômage
      - "all"           : score agrégé (somme des trois)
    
    Score positif → hawkish, score négatif → dovish.
    """
    windows = extract_windows(text)

    # Accumuler les scores par keyword
    scores_by_keyword: dict[str, dict[str, float]] = {
        "inflation":    {"hawkish": 0.0, "dovish": 0.0},
        "interest rate": {"hawkish": 0.0, "dovish": 0.0},
        "unemployment": {"hawkish": 0.0, "dovish": 0.0},
    }

    trace = []

    for window in windows:
        kw = window["keyword"]
        ws = _score_window(window, return_details=return_details)
        scores_by_keyword[kw]["hawkish"] += ws["hawkish"]
        scores_by_keyword[kw]["dovish"]  += ws["dovish"]
        if return_details:
            trace.append(
                {
                    "keyword": kw,
                    "sentence": window.get("sentence", ""),
                    **ws.get("details", {}),
                }
            )

    # Formule (2) : sentiment = hawkish - dovish
    sentiment = {}
    total = 0.0

    for kw, sc in scores_by_keyword.items():
        s = sc["hawkish"] - sc["dovish"]
        key = kw.replace(" ", "_")
        sentiment[key] = s
        total += s

    sentiment["all"] = total

    if not return_details:
        return sentiment

    return {
        "scores": sentiment,
        "trace": trace,
        "by_keyword": scores_by_keyword,
        "n_windows": len(windows),
    }


def score_all_documents(
    documents: list[dict],
) -> pd.DataFrame:
    """
    Calcule les scores pour une liste de documents.
    
    Args:
        documents : liste de dicts {date, text, word_count}
    
    Returns:
        DataFrame avec colonnes :
          date, inflation, interest_rate, unemployment, all
    """
    records = []
    total = len(documents)

    if total == 0:
        logger.warning("Aucun document à scorer. DataFrame vide retourné.")
        return pd.DataFrame(
            columns=["date", "inflation", "interest_rate", "unemployment", "all"]
        )

    for i, doc in enumerate(documents, 1):
        date = doc["date"]
        text = doc.get("text", "")

        if not text:
            logger.warning(f"[{i}/{total}] Texte vide pour {date}")
            records.append({
                "date": date,
                "inflation": 0.0,
                "interest_rate": 0.0,
                "unemployment": 0.0,
                "all": 0.0,
            })
            continue

        logger.debug(f"[{i}/{total}] Score pour {date}")
        scores = score_document(text)
        records.append({"date": date, **scores})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Stats rapides pour validation
    logger.info(f"\n{'='*50}")
    logger.info(f"Scores calculés pour {len(df)} documents")
    logger.info(f"\nMoyennes :")
    for col in ["inflation", "interest_rate", "unemployment", "all"]:
        logger.info(f"  {col:20s}: {df[col].mean():.4f} (std={df[col].std():.4f})")
    logger.info(f"{'='*50}")

    return df

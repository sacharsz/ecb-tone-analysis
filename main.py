"""
Pipeline complet ECB Sentiment.
Lance les 3 étapes séquentiellement :
    1. Chargement des URLs depuis CSV local
    2. Chargement des transcripts depuis CSV local
  3. Calcul des scores de sentiment
"""

import argparse
import pandas as pd
from pathlib import Path

from config import PATHS, EXPECTED_N_CONFERENCES, EXPECTED_MEAN_WORD_COUNT, DATE_START, DATE_END
from src.data.transcript_fetcher import load_urls_from_warin_csv, load_transcripts_from_warin_csv
from src.sentiment.scorer import score_all_documents
from src.utils.logger import setup_logger

logger = setup_logger()


def _find_local_dataset_path() -> Path | None:
    candidates = [
        Path(PATHS.get("warin_csv", "")),
        Path("data/raw/warin_sanger_ECB.csv"),
        Path("data/raw/Warin_Sanger_ECB.csv"),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return None


def _save_url_list(df: pd.DataFrame) -> None:
    path = Path(PATHS["url_list"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"URLs sauvegardées → {path}")


def step1_collect_urls() -> pd.DataFrame:
    """Étape 1 : Chargement des URLs depuis le CSV local."""
    warin_csv_path = _find_local_dataset_path()
    if not warin_csv_path:
        raise FileNotFoundError(
            "CSV source introuvable. Placez Warin_Sanger_ECB.csv dans data/raw/."
        )

    logger.info(f"Dataset local détecté : {warin_csv_path}")
    df = load_urls_from_warin_csv(warin_csv_path)
    _save_url_list(df)

    logger.info(
        f"{len(df)} URLs chargées depuis dataset local (période cible {DATE_START} → {DATE_END})"
    )

    # Validation
    if len(df) < EXPECTED_N_CONFERENCES * 0.7:
        logger.warning(
            f"⚠ Seulement {len(df)} conférences trouvées "
            f"(attendu ~{EXPECTED_N_CONFERENCES}). "
            "Vérifier le fichier CSV source."
        )
    return df


def step2_fetch_transcripts() -> list[dict]:
    """Étape 2 : Chargement des transcripts depuis le CSV local."""
    logger.info("=== ÉTAPE 2 : Chargement des transcripts ===")
    warin_csv_path = _find_local_dataset_path()

    if not warin_csv_path:
        raise FileNotFoundError(
            "CSV source introuvable. Placez Warin_Sanger_ECB.csv dans data/raw/."
        )

    documents = load_transcripts_from_warin_csv(warin_csv_path)

    # Validation : longueur moyenne (papier : 5719 mots)
    wcs = [d["word_count"] for d in documents if d["word_count"] > 0]
    if wcs:
        mean_wc = sum(wcs) / len(wcs)
        logger.info(f"Longueur moyenne : {mean_wc:.0f} mots (papier : {EXPECTED_MEAN_WORD_COUNT})")
        if abs(mean_wc - EXPECTED_MEAN_WORD_COUNT) / EXPECTED_MEAN_WORD_COUNT > 0.3:
            logger.warning(
                "⚠ La longueur moyenne s'écarte de plus de 30% de celle du papier. "
                "Vérifier l'extraction HTML."
            )
    return documents


def step3_compute_scores(
    documents: list[dict],
) -> pd.DataFrame:
    """Étape 3 : Calcul des scores de sentiment."""
    logger.info("=== ÉTAPE 3 : Calcul des scores de sentiment ===")
    scores_df = score_all_documents(documents)

    output_path = Path(PATHS["scores"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    logger.info(f"Scores sauvegardés → {output_path}")

    return scores_df


def run_pipeline():
    """Pipeline complet."""
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║     ECB SENTIMENT PIPELINE           ║")
    logger.info("╚══════════════════════════════════════╝")

    step1_collect_urls()
    documents = step2_fetch_transcripts()
    scores_df = step3_compute_scores(documents)

    logger.info("\n✓ Pipeline terminé avec succès.")
    logger.info(f"  Fichier de sortie : {PATHS['scores']}")
    logger.info(f"  Prochaine étape : ouvrir notebooks/02_sentiment_validation.ipynb")
    return scores_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECB Sentiment Pipeline")
    _ = parser.parse_args()

    run_pipeline()

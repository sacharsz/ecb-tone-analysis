"""
Étape 2 : Chargement des transcripts BCE depuis le CSV local Warin & Sanger.
"""

import pandas as pd
from pathlib import Path

from config import DATE_START, DATE_END
from logger import setup_logger

logger = setup_logger()


def _read_warin_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = None
    last_error = None
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc, engine="python")
            logger.info(f"CSV lu avec encodage {enc}")
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        raise UnicodeDecodeError(
            "csv",
            b"",
            0,
            1,
            f"Impossible de lire {csv_path} avec utf-8/cp1252/latin1: {last_error}",
        )
    return df


def load_urls_from_warin_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Extrait un DataFrame d'URLs (date, url) depuis le CSV Warin & Sanger.
    """
    logger.info(f"Chargement URLs depuis dataset local: {csv_path}")
    df = _read_warin_csv(csv_path)

    required_cols = ["date", "link"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {csv_path}: {missing}. Colonnes disponibles: {list(df.columns)}"
        )

    url_df = df[["date", "link"]].rename(columns={"link": "url"}).copy()
    url_df["date"] = pd.to_datetime(url_df["date"], errors="coerce")
    url_df["url"] = url_df["url"].astype(str).str.strip()
    url_df = url_df.dropna(subset=["date"])
    url_df = url_df[url_df["url"] != ""]

    mask = (url_df["date"] >= DATE_START) & (url_df["date"] <= DATE_END)
    url_df = url_df[mask].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return url_df


def load_transcripts_from_warin_csv(csv_path: str | Path) -> list[dict]:
    """
    Charge les transcripts depuis le dataset Warin & Sanger déjà présent en local.

    Le fichier attendu est un CSV séparé par ';' avec au minimum :
      - date
      - firstPart
      - secondPart
    et idéalement :
      - firstPartVp
      - secondPartExternalIntervention
      - link
    """
    logger.info(f"Chargement dataset local: {csv_path}")
    df = _read_warin_csv(csv_path)

    expected_cols = ["date", "firstPart", "secondPart"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {csv_path}: {missing}. Colonnes disponibles: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mask = (df["date"] >= DATE_START) & (df["date"] <= DATE_END)
    df = df.dropna(subset=["date"])
    df = df[mask].sort_values("date").reset_index(drop=True)

    def _safe_text(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    documents = []
    for _, row in df.iterrows():
        parts = [
            _safe_text(row.get("firstPart")),
            _safe_text(row.get("firstPartVp")),
            _safe_text(row.get("secondPart")),
            _safe_text(row.get("secondPartExternalIntervention")),
        ]
        text = " ".join([part for part in parts if part]).strip()
        documents.append(
            {
                "date": row["date"],
                "text": text,
                "word_count": len(text.split()),
                "url": _safe_text(row.get("link")),
                "source": "warin_sanger_csv",
            }
        )

    success = sum(1 for doc in documents if doc["word_count"] > 0)
    logger.info(f"Dataset local chargé: {success}/{len(documents)} textes non vides")
    return documents

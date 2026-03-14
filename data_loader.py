# data_loader.py
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "eonia": "EONIARATE",
    "estr":  "ECBESTRVOLWGTTRMDMNRT",
    "mro":   "ECBMRRFR",
}


def download_series(series_id: str, filename: str) -> pd.DataFrame:
    url = f"{FRED_BASE}?id={series_id}"
    print(f"Téléchargement {series_id}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    path = DATA_DIR / filename
    path.write_bytes(r.content)

    # Lire sans supposer le nom de colonne
    df = pd.read_csv(path)
    print(f"  Colonnes détectées : {list(df.columns)}")

    # Renommer la première colonne en 'date' quelle que soit sa casse
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    print(f"  ✓ {len(df)} observations → {path}")
    return df


def build_eonia_series() -> pd.DataFrame:
    """Concatène EONIA (jusqu'à 2021) et €STR (depuis 2022)."""
    eonia = download_series("EONIARATE", "eonia_raw.csv")
    estr  = download_series("ECBESTRVOLWGTTRMDMNRT", "estr_raw.csv")

    eonia.columns = ["rate"]
    estr.columns  = ["rate"]

    combined = pd.concat([eonia, estr]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.to_csv(DATA_DIR / "eonia_combined.csv")
    print(f"  ✓ Série combinée → data/raw/eonia_combined.csv")
    return combined


def build_mro_series() -> pd.DataFrame:
    path = Path("data/raw/mro_ecb.csv")
    
    df = pd.read_csv(path)
    print(f"Colonnes : {list(df.columns)}")
    
    # Garder seulement DATETIME + la valeur (3ème colonne)
    df = df.iloc[:, [0, 2]]
    df.columns = ["date", "mro"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    # Supprimer les doublons (même taux répété chaque jour)
    # et garder uniquement les jours où le taux change
    df = df[df["mro"] != df["mro"].shift(1)]
    
    df.to_csv("data/raw/mro_combined.csv")
    print(f"✓ {len(df)} changements de taux chargés")
    print(df.head(10))
    return df


if __name__ == "__main__":
    build_eonia_series()
    build_mro_series()
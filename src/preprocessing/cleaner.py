"""
Nettoyage et preprocessing des textes des conférences de presse BCE.
Implémente les étapes décrites en section 3.1 du papier :
  - Suppression ponctuation, capitalisation, stop words
  - Remplacement des caractères spéciaux (ex: \u0080 → Euro)
  - Stemming (Porter)
  - Fenêtrage autour des mots-clés (±10 mots)
"""

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

from config import WINDOW_SIZE, TARGET_WORDS

# Télécharger les ressources NLTK si nécessaire
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()

# Corrections de caractères spéciaux mentionnées dans le papier (section 3.1)
CHAR_REPLACEMENTS = {
    "\u0080": "euro",
    "\u20ac": "euro",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
}


def fix_special_chars(text: str) -> str:
    for char, replacement in CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def clean_text(text: str) -> str:
    """Nettoyage basique : lowercase, ponctuation, caractères spéciaux."""
    text = fix_special_chars(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)   # espaces multiples
    text = text.strip()
    return text


def tokenize_sentence(sentence: str) -> list[str]:
    """
    Tokenise une phrase et applique :
    - suppression ponctuation
    - suppression stop words
    - stemming (Porter)
    """
    tokens = word_tokenize(sentence)
    cleaned = []
    for tok in tokens:
        tok = tok.lower().strip(string.punctuation)
        if not tok or tok in STOP_WORDS:
            continue
        if not re.match(r"^[a-z]+$", tok):   # garder seulement lettres
            continue
        stemmed = STEMMER.stem(tok)
        cleaned.append(stemmed)
    return cleaned


def stem_keyword(keyword: str) -> str:
    """Stemme un mot-clé pour la comparaison."""
    return STEMMER.stem(keyword.lower())


# Mots-clés stemmés pour la détection rapide
STEMMED_TARGETS = {
    keyword: stem_keyword(keyword.split()[0])  # premier mot du bigram
    for keyword in TARGET_WORDS
}


def _raw_tokens(sentence: str) -> list[str]:
    """
    Tokenise une phrase en tokens bruts (lowercase, lettres seulement).
    PAS de suppression stop words, PAS de stemming.
    Utilisé pour localiser la position du mot-clé avant fenêtrage.
    """
    tokens = word_tokenize(sentence)
    return [
        tok.lower().strip(string.punctuation)
        for tok in tokens
        if re.match(r"^[a-zA-Z]+$", tok.strip(string.punctuation))
    ]


def _filter_window(raw_window: list[str]) -> list[str]:
    """
    Applique suppression stop words + stemming à une fenêtre de tokens bruts.
    C'est ce qui donne les tokens finaux comptés dans N_i.
    """
    result = []
    for tok in raw_window:
        if tok in STOP_WORDS:
            continue
        stemmed = STEMMER.stem(tok)
        result.append(stemmed)
    return result


def _filter_window_raw(raw_window: list[str]) -> list[str]:
    """
    Fenêtre filtrée des stop words sans stemming.
    Sert au matching wildcard du dictionnaire.
    """
    result = []
    for tok in raw_window:
        if tok in STOP_WORDS:
            continue
        result.append(tok)
    return result


def extract_windows(text: str) -> list[dict]:
    """
    Pour chaque occurrence d'un mot-clé cible dans le texte,
    extrait une fenêtre de ±WINDOW_SIZE mots autour.

    Ordre correct (conforme au papier section 3.1) :
      1. Tokeniser en tokens BRUTS (pas de filtrage encore)
      2. Localiser le mot-clé dans les tokens bruts
      3. Extraire la fenêtre ±WINDOW_SIZE sur les tokens BRUTS
      4. Supprimer les stop words et stemmer dans la fenêtre → tokens finaux
      5. N_i = len(tokens finaux)  ← c'est ce que le papier appelle N_i

    Vérification sur l'exemple du papier (section 3.2) :
      Phrase : "Inflation remains very low in the context of weak demand
                and significant slack in labour and product markets."
      Fenêtre brute autour de "inflation" (10 mots après) :
        [inflation, remains, very, low, in, the, context, of, weak, demand]
      Après suppression stop words :
        [inflation, remains, low, context, weak, demand]  → N_i = 6 ✓
      Mots dovish dans la fenêtre : low, weak → score = 2/6 = 0.33 ✓

    Retourne une liste de dicts :
      {
        "keyword"  : str,        # mot-clé trouvé ("inflation", etc.)
        "sentence" : str,        # phrase originale
        "tokens"   : list[str],  # tokens stemmés dans la fenêtre (après filtrage)
        "n_tokens" : int,        # N_i du papier
      }
    """
    text = clean_text(text)
    sentences = sent_tokenize(text)
    results = []

    for sent in sentences:
        raw = _raw_tokens(sent)   # tokens bruts, pas de filtrage
        if not raw:
            continue

        for keyword, info in TARGET_WORDS.items():

            # ── Trouver les positions du mot-clé dans les tokens BRUTS ──────
            kw_first = keyword.split()[0].lower()   # "inflation" ou "interest"
            positions = []

            if info["type"] == "unigram":
                positions = [i for i, t in enumerate(raw) if t == kw_first]

            elif info["type"] == "bigram":
                default_second = keyword.split()[1].lower()
                second_forms = [w.lower() for w in info.get("forms", [default_second])]
                for i in range(len(raw) - 1):
                    if raw[i] == kw_first and raw[i + 1] in second_forms:
                        positions.append(i)

            # ── Pour chaque occurrence : fenêtre brute → filtrage ────────────
            if info["type"] == "unigram":
                keyword_count_in_sentence = sum(1 for t in raw if t == kw_first)
            else:
                default_second = keyword.split()[1].lower()
                second_forms = [w.lower() for w in info.get("forms", [default_second])]
                keyword_count_in_sentence = sum(
                    1
                    for i in range(len(raw) - 1)
                    if raw[i] == kw_first and raw[i + 1] in second_forms
                )

            n_occurrences = len(positions)

            spans = [
                (max(0, pos - WINDOW_SIZE), min(len(raw), pos + WINDOW_SIZE + 1), pos)
                for pos in positions
            ]

            for start, end, _ in spans:
                raw_window = raw[start:end]
                filtered_tokens = _filter_window(raw_window)
                filtered_raw_tokens = _filter_window_raw(raw_window)

                results.append({
                    "keyword": keyword,
                    "sentence": sent,
                    "tokens": filtered_tokens,
                    "raw_tokens": filtered_raw_tokens,
                    "n_tokens": len(filtered_tokens),
                    "keyword_count": keyword_count_in_sentence,
                    "occurrences_in_sentence": n_occurrences,
                })

    return results
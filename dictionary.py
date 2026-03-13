"""
Dictionnaire hawkish/dovish.
Transcription exacte de la Table A1 du papier (Czudaj & Nguyen, 2025).

Note sur les wildcards :
    Les mots du dictionnaire contiennent des troncatures (ris*, increas*, etc.).
    Le matching est fait sur tokens bruts (préfixe) pour respecter ces patterns,
    puis avec fallback stemmé pour les entrées sans wildcard.
"""

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def _normalize_pattern(word: str) -> str:
        return word.lower().strip()


def _token_matches_pattern(token_raw: str, pattern: str) -> bool:
        """
        Règle de matching :
            - pattern wildcard (ex: ris*)   -> token.startswith('ris')
            - pattern exact (ex: rose, fell)-> comparaison exacte OU stemmée
        """
        token = token_raw.lower().strip()
        pat = _normalize_pattern(pattern)

        if not token or not pat:
                return False

        if pat.endswith("*"):
                prefix = pat[:-1]
                return bool(prefix) and token.startswith(prefix)

        return token == pat or _stemmer.stem(token) == _stemmer.stem(pat)


# ── Dictionnaire brut (Table A1 du papier) ────────────────────────────────────

_RAW_DICTIONARY = {
    "inflation": {
        "hawkish": [
            "ris*", "rose", "favourable", "favorable",
            "increas*", "high*", "fast*", "buoyant",
            "strong*", "grow*", "grew", "worse*",
        ],
        "dovish": [
            "declin*", "decreas*", "low*", "fall*", "fell",
            "slow*", "weak*",
        ],
    },
    "interest rate": {
        "hawkish": [
            "ris*", "rose", "favourable", "favorable",
            "increas*", "high*", "fast*", "buoyant",
            "strong*", "grow*", "grew", "worse*",
        ],
        "dovish": [
            "declin*", "decreas*", "low*", "fall*", "fell",
            "slow*", "weak*",
        ],
    },
    # Pour unemployment : hawkish et dovish sont inversés par rapport
    # aux autres keywords (chômage qui BAISSE = hawkish)
    "unemployment": {
        "hawkish": [
            "declin*", "decreas*", "low*", "fall*", "fell",
            "slow*", "weak*", "shrink", "shrank",
        ],
        "dovish": [
            "ris*", "rose", "favourable", "favorable",
            "increas*", "high*", "fast*", "strong*",
            "grow*", "grew", "down", "modest",
        ],
    },
}

DICTIONARY: dict[str, dict[str, list[str]]] = {
    keyword: {
        "hawkish": [_normalize_pattern(w) for w in words["hawkish"]],
        "dovish": [_normalize_pattern(w) for w in words["dovish"]],
    }
    for keyword, words in _RAW_DICTIONARY.items()
}


def get_words(keyword: str, tone: str) -> list[str]:
    """
    Retourne la liste des mots/patterns hawkish ou dovish
    pour un mot-clé donné.

    Args:
        keyword : "inflation", "interest rate", ou "unemployment"
        tone    : "hawkish" ou "dovish"
    """
    if keyword not in DICTIONARY:
        raise KeyError(f"Keyword inconnu : {keyword}. Attendu : {list(DICTIONARY.keys())}")
    if tone not in ("hawkish", "dovish"):
        raise ValueError(f"Tone inconnu : {tone}. Attendu : 'hawkish' ou 'dovish'")
    return DICTIONARY[keyword][tone]


def count_pattern_matches(tokens_raw: list[str], patterns: list[str]) -> int:
    """Compte le nombre de tokens qui matchent au moins un pattern."""
    total = 0
    for token in tokens_raw:
        if any(_token_matches_pattern(token, pattern) for pattern in patterns):
            total += 1
    return total

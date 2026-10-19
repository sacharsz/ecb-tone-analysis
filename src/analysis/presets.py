EONIA_PRESET = {
    'target_csv_path': 'data/raw/eonia_combined.csv',
    'target_col': 'rate',
    'target_name': 'rate',
    'target_previous_day': False,
    'controls': [
        {
            'csv_path': 'data/raw/mro_combined.csv',
            'col': 'mro',
            'name': 'mro',
            'previous_day': False,
            'fill_method': 'bfill',
        }
    ],
    'regression': {
        'y_col': 'rate',
        'x_cols': ['{sentiment_lag}', 'mro', 'rate_lag_1'],
        'include_y_lag_1': True,
        'hac_maxlags': 4,
    },
    'graph_focus_feature': 'sentiment_lag',
    'graph_max_k': 12,
    'graph_keywords': ['inflation', 'interest_rate', 'unemployment', 'all'],
}

EONIA_MRO_FOCUS_PRESET = {
    **EONIA_PRESET,
    'graph_focus_feature': 'mro',
}

EONIA_PAPER_STEP2_PRESET = {
    'target_csv_path': 'data/raw/eonia_combined.csv',
    'target_col': 'rate',
    'target_name': 'rate',
    'target_previous_day': False,
    'controls': [
        {
            'csv_path': 'data/raw/mro_combined.csv',
            'col': 'mro',
            'name': 'mro',
            'previous_day': False,
            'fill_method': 'bfill',
        }
    ],
    'regression': {
        'y_col': 'rate',
        'x_cols': ['{sentiment_lag}', 'mro'],
        'include_y_lag_1': False,
        'hac_maxlags': 4,
    },
    'graph_focus_feature': 'sentiment_lag',
    'graph_min_k': 0,
    'graph_max_k': 12,
    'graph_keywords': ['inflation', 'interest_rate', 'unemployment', 'all'],
    'graph_ci_style': 'errorbar',
}

BUND10_PRESET = {
    'target_csv_path': 'data/raw/bund10_combined.csv',
    'target_col': 'rate',
    'target_name': 'bund10',
    'target_previous_day': False,
    'controls': [
        {
            'csv_path': 'data/raw/mro_combined.csv',
            'col': 'mro',
            'name': 'mro',
            'previous_day': False,
            'fill_method': 'bfill',
        }
    ],
    'regression': {
        'y_col': 'bund10',
        'x_cols': ['{sentiment_lag}', 'mro', 'bund10_lag_1'],
        'include_y_lag_1': True,
        'hac_maxlags': 4,
    },
    'graph_focus_feature': 'sentiment_lag',
    'graph_max_k': 12,
}

PRESETS = {
    'eonia': EONIA_PRESET,
    'eonia_mro_focus': EONIA_MRO_FOCUS_PRESET,
    'eonia_paper_step2': EONIA_PAPER_STEP2_PRESET,
    'bund10': BUND10_PRESET,
}


def get_preset(name: str) -> dict:
    key = name.lower().strip()
    if key not in PRESETS:
        raise ValueError(f"Preset inconnu: {name}. Disponibles: {list(PRESETS.keys())}")
    return PRESETS[key]

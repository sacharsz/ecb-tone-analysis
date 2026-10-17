import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence

import statsmodels.api as sm

from src.utils.logger import setup_logger
from .presets import get_preset

logger = setup_logger()


def run_regression_from_xy(
    y: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    focus_feature: str,
    hac_maxlags: int = 4,
) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X et y doivent avoir le même nombre de lignes.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names doit avoir la même longueur que le nombre de colonnes de X.")
    if focus_feature not in feature_names:
        raise ValueError(f"focus_feature='{focus_feature}' absent de feature_names={list(feature_names)}")

    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta_hat = np.linalg.inv(X_with_const.T @ X_with_const) @ (X_with_const.T @ y)

    y_sm = pd.Series(y.flatten(), name='y')
    X_sm = pd.DataFrame(X, columns=list(feature_names))
    X_sm = sm.add_constant(X_sm)
    model = sm.OLS(y_sm, X_sm).fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})

    coef = float(model.params[focus_feature])
    se = float(model.bse[focus_feature])
    lower = coef - 1.96 * se
    upper = coef + 1.96 * se

    hac_result = {
        'coef': coef,
        'se': se,
        'lower': float(lower),
        'upper': float(upper),
        'pvalue': float(model.pvalues[focus_feature]),
    }

    return {
        'beta_hat': beta_hat,
        'hac': hac_result,
        'model': model,
        'feature_names': ['const', *feature_names],
    }


def _load_series(csv_path: str | Path, value_col: str | None = None, output_name: str = 'value') -> pd.DataFrame:
    series = pd.read_csv(csv_path, parse_dates=['date'])
    series = series.dropna(subset=['date']).sort_values('date')

    if value_col is None:
        non_date_cols = [column for column in series.columns if column != 'date']
        if not non_date_cols:
            raise ValueError(f"Aucune colonne de valeur trouvée dans {csv_path}")
        value_col = non_date_cols[0]

    if value_col not in series.columns:
        raise ValueError(f"Colonne '{value_col}' introuvable dans {csv_path}")

    series = series[['date', value_col]].dropna().sort_values('date')
    series = series.rename(columns={'date': 'source_date', value_col: output_name})
    return series


def step4_first_regression(
    regression_df: pd.DataFrame | None = None,
    k: int = 1,
    keyword: str = "all",
    y_col: str = 'rate',
    x_cols: list[str] | None = None,
    include_y_lag_1: bool = True,
    y: np.ndarray | None = None,
    X: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    focus_feature: str | None = None,
    hac_maxlags: int = 4,
) -> dict:
    logger.info("=== ÉTAPE 4 : Première régression linéaire ===")
    if y is not None and X is not None:
        if feature_names is None:
            raise ValueError("feature_names est requis si X/y sont passés directement.")
        if focus_feature is None:
            focus_feature = feature_names[0]
        result = run_regression_from_xy(
            y=y,
            X=X,
            feature_names=feature_names,
            focus_feature=focus_feature,
            hac_maxlags=hac_maxlags,
        )
        logger.info(
            f"HAC (maxlags={hac_maxlags}) {focus_feature} : "
            f"coef={result['hac']['coef']:.6f}, se={result['hac']['se']:.6f}, "
            f"p-value={result['hac']['pvalue']:.6f}, "
            f"IC95%=[{result['hac']['lower']:.6f}, {result['hac']['upper']:.6f}]"
        )
        return result

    if regression_df is None:
        raise ValueError("regression_df est requis si X/y ne sont pas passés directement.")

    regression_data = regression_df.copy()
    sentiment_col = f"sentiment_lag_{k}"
    regression_data[sentiment_col] = regression_data[keyword].shift(k)

    if include_y_lag_1:
        y_lag_col = f"{y_col}_lag_1"
        regression_data[y_lag_col] = regression_data[y_col].shift(1)
    else:
        y_lag_col = None

    if x_cols is None:
        x_cols = [sentiment_col, 'mro']
        if y_lag_col:
            x_cols.append(y_lag_col)
    else:
        x_cols = [
            (sentiment_col if col in {'sentiment_lag', '{sentiment_lag}'} else col)
            for col in x_cols
        ]

    drop_cols = [y_col, *x_cols]
    regression_ready = regression_data.dropna(subset=drop_cols).copy()

    y_arr = regression_ready[y_col].to_numpy(dtype=float)
    X_arr = regression_ready[x_cols].to_numpy(dtype=float)

    focus = focus_feature or sentiment_col
    if focus in {'sentiment_lag', '{sentiment_lag}'}:
        focus = sentiment_col
    if focus == sentiment_col:
        feature_names = [
            ('sentiment_lag' if col == sentiment_col else col)
            for col in x_cols
        ]
        focus = 'sentiment_lag'
    else:
        feature_names = list(x_cols)

    result = run_regression_from_xy(
        y=y_arr,
        X=X_arr,
        feature_names=feature_names,
        focus_feature=focus,
        hac_maxlags=hac_maxlags,
    )

    logger.info(
        f"HAC (maxlags={hac_maxlags}) {focus} : "
        f"coef={result['hac']['coef']:.6f}, se={result['hac']['se']:.6f}, "
        f"p-value={result['hac']['pvalue']:.6f}, "
        f"IC95%=[{result['hac']['lower']:.6f}, {result['hac']['upper']:.6f}]"
    )

    return result


def build_regression(
    scores_df: pd.DataFrame,
    target_csv_path: str | Path = 'data/raw/eonia_combined.csv',
    target_col: str = 'rate',
    target_name: str = 'rate',
    target_previous_day: bool = True,
    controls: list[dict] | None = None,
) -> pd.DataFrame:
    if controls is None:
        controls = [
            {
                'csv_path': 'data/raw/mro_combined.csv',
                'col': 'mro',
                'name': 'mro',
                'previous_day': False,
                'fill_method': 'bfill',
            }
        ]

    scores = scores_df.copy()
    scores['date'] = pd.to_datetime(scores['date'], errors='coerce')
    scores = scores.dropna(subset=['date']).sort_values('date')
    conference_dates = scores[['date']].drop_duplicates().sort_values('date')

    score_cols = [
        column for column in ['inflation', 'interest_rate', 'unemployment', 'all']
        if column in scores.columns
    ]

    scores_aligned = conference_dates.merge(
        scores[['date', *score_cols]].sort_values('date'),
        on='date',
        how='left',
    )

    target_series = _load_series(target_csv_path, value_col=target_col, output_name=target_name)
    target_series = target_series.rename(columns={'source_date': f'{target_name}_source_date'})
    target_aligned = pd.merge_asof(
        conference_dates,
        target_series,
        left_on='date',
        right_on=f'{target_name}_source_date',
        direction='backward',
        allow_exact_matches=not target_previous_day,
    )

    aligned_controls: list[pd.DataFrame] = []
    for control in controls:
        control_name = control.get('name')
        control_series = _load_series(
            control['csv_path'],
            value_col=control.get('col'),
            output_name=control_name,
        )
        control_series = control_series.rename(columns={'source_date': f'{control_name}_source_date'})

        control_aligned = pd.merge_asof(
            conference_dates,
            control_series,
            left_on='date',
            right_on=f'{control_name}_source_date',
            direction='backward',
            allow_exact_matches=not control.get('previous_day', False),
        )

        fill_method = control.get('fill_method')
        if fill_method:
            if fill_method == 'bfill':
                control_aligned[control_name] = control_aligned[control_name].bfill()
            elif fill_method == 'ffill':
                control_aligned[control_name] = control_aligned[control_name].ffill()
            else:
                control_aligned[control_name] = control_aligned[control_name].fillna(fill_method)

        aligned_controls.append(control_aligned[['date', control_name]])

    regression_df = (
        conference_dates
        .merge(target_aligned, on='date', how='left')
        .merge(scores_aligned, on='date', how='left')
    )
    for control_aligned in aligned_controls:
        regression_df = regression_df.merge(control_aligned, on='date', how='left')

    logger.info(f"Aperçu alignement {target_name} (conférence -> valeur utilisée) :")
    logger.info(
        "\n" + regression_df[['date', f'{target_name}_source_date', target_name]].head(8).to_string(index=False)
    )
    return regression_df


def graph_beta_over_lags(
    regression_df: pd.DataFrame,
    min_k: int = 1,
    max_k: int = 12,
    regression_params: dict | None = None,
    focus_feature: str = 'sentiment_lag',
    keywords: list[str] | None = None,
    ci_style: str = 'band',
) -> None:
    import matplotlib.pyplot as plt

    params = dict(regression_params or {})
    y_col = params.get('y_col', 'rate')
    if focus_feature in {'{y_lag_1}', 'y_lag_1'}:
        focus_feature_display = f'{y_col}_lag_1'
    elif focus_feature in {'sentiment_lag', '{sentiment_lag}'}:
        focus_feature_display = 'sentiment_lag_k'
    else:
        focus_feature_display = focus_feature

    if keywords is None:
        keywords = ['inflation', 'interest_rate', 'unemployment', 'all']

    results = {
        keyword: {'coef': [], 'lower': [], 'upper': [], 'pvalue': []}
        for keyword in keywords
    }

    for keyword in keywords:
        for lag in range(min_k, max_k + 1):
            current_focus = focus_feature
            if focus_feature in {'{y_lag_1}', 'y_lag_1'}:
                current_focus = f'{y_col}_lag_1'

            regression_result = step4_first_regression(
                regression_df=regression_df,
                k=lag,
                keyword=keyword,
                focus_feature=current_focus,
                **params,
            )
            hac = regression_result['hac']
            results[keyword]['coef'].append(hac['coef'])
            results[keyword]['lower'].append(hac['lower'])
            results[keyword]['upper'].append(hac['upper'])
            results[keyword]['pvalue'].append(hac['pvalue'])

    pvalues_records = []
    for keyword in keywords:
        for index, lag in enumerate(range(min_k, max_k + 1)):
            pvalues_records.append({
                'keyword': keyword,
                'lag_k': lag,
                'coef_hac': results[keyword]['coef'][index],
                'pvalue_hac': results[keyword]['pvalue'][index],
                'ci95_lower': results[keyword]['lower'][index],
                'ci95_upper': results[keyword]['upper'][index],
            })

    pvalues_df = pd.DataFrame(pvalues_records)
    pvalues_path = Path('data/output/hac_pvalues_by_lag.csv')
    pvalues_path.parent.mkdir(parents=True, exist_ok=True)
    pvalues_df.to_csv(pvalues_path, index=False)
    logger.info(f"P-values HAC sauvegardées -> {pvalues_path}")

    n_keywords = len(keywords)
    if n_keywords == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
        axes = [axes]
    elif n_keywords == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        axes = list(axes)
    else:
        n_rows = int(np.ceil(n_keywords / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True)
        axes = np.array(axes).reshape(-1).tolist()

    lags = list(range(min_k, max_k + 1))

    for ax, keyword in zip(axes, keywords):
        coef_values = np.asarray(results[keyword]['coef'], dtype=float)
        lower_values = np.asarray(results[keyword]['lower'], dtype=float)
        upper_values = np.asarray(results[keyword]['upper'], dtype=float)

        if ci_style == 'errorbar':
            lower_err = coef_values - lower_values
            upper_err = upper_values - coef_values
            ax.errorbar(
                lags,
                coef_values,
                yerr=[lower_err, upper_err],
                fmt='o',
                capsize=3,
                label=f"β {focus_feature_display}",
            )
        else:
            ax.plot(lags, coef_values, marker='o', label=f"β {focus_feature_display}")
            ax.fill_between(
                lags,
                lower_values,
                upper_values,
                alpha=0.2,
                label='IC95% HAC',
            )
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

        finite_values = np.concatenate([
            coef_values[np.isfinite(coef_values)],
            lower_values[np.isfinite(lower_values)],
            upper_values[np.isfinite(upper_values)],
        ])
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_max), 1.0)
        pad = max(0.1 * span, 0.02)

        low_lim = y_min - pad
        high_lim = y_max + pad

        # Inclure 0 uniquement s'il est proche de la zone affichée,
        # pour éviter une échelle écrasée quand les coefficients sont loin de 0.
        zero_threshold = 0.75 * span
        if low_lim <= 0 <= high_lim:
            pass
        elif abs(0 - low_lim) <= zero_threshold:
            low_lim = 0
        elif abs(high_lim - 0) <= zero_threshold:
            high_lim = 0

        ax.set_ylim(low_lim, high_lim)

        ax.set_title(keyword)
        ax.set_xlabel('Lag k')
        ax.set_ylabel(f'β {focus_feature_display} (HAC)')
        ax.set_xticks(lags)
        ax.set_xticklabels(lags)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    for ax in axes[len(keywords):]:
        ax.set_visible(False)

    fig.suptitle(f'Coefficient {focus_feature_display} en fonction du lag k', fontsize=13)
    plt.tight_layout()
    fig.savefig(pvalues_path.parent / f'coef_{focus_feature_display}_by_lag.png', dpi=150)
    plt.show()


def run_regression_preset(scores_df: pd.DataFrame, preset_name: str = 'eonia') -> dict:
    preset = get_preset(preset_name)

    regression_df = build_regression(
        scores_df=scores_df,
        target_csv_path=preset['target_csv_path'],
        target_col=preset['target_col'],
        target_name=preset['target_name'],
        target_previous_day=preset['target_previous_day'],
        controls=preset['controls'],
    )

    regression_result = step4_first_regression(
        regression_df=regression_df,
        **preset['regression'],
    )

    graph_beta_over_lags(
        regression_df=regression_df,
        min_k=preset.get('graph_min_k', 1),
        max_k=preset.get('graph_max_k', 12),
        regression_params=preset['regression'],
        focus_feature=preset.get('graph_focus_feature', 'sentiment_lag'),
        keywords=preset.get('graph_keywords'),
        ci_style=preset.get('graph_ci_style', 'band'),
    )
    return regression_result


def run_eonia_example(scores_df: pd.DataFrame) -> dict:
    return run_regression_preset(scores_df, preset_name='eonia')

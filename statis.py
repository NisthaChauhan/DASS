import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene, f_oneway, kruskal
from config import ALPHA

def run_statistical_tests(merged_df):
    results = []
    warnings = []

    for aspect in ['Depression', 'Anxiety', 'Stress']:
        _, p_norm = shapiro(merged_df[f'{aspect}_change'])
        t_stat, p_val = ttest_rel(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])

        if p_norm < ALPHA:
            _, p_val = wilcoxon(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 'Wilcoxon'
            t_stat = np.nan
            warnings.append(aspect)
        else:
            test_type = 't-test'

        results.append({
            'Aspect': aspect,
            'Test': test_type,
            'Statistic': t_stat,
            'p-value': p_val
        })

    return results, warnings

def run_anova(merged_df):
    melted = merged_df.melt(
        value_vars=['Depression_change', 'Anxiety_change', 'Stress_change'],
        var_name='Aspect', value_name='Change'
    )

    groups = [melted[melted['Aspect'] == name]['Change'] for name in melted['Aspect'].unique()]
    _, levene_p = levene(*groups)

    if all(shapiro(g)[1] > ALPHA for g in groups) and levene_p > ALPHA:
        f_stat, p_val = f_oneway(*groups)
        test = 'ANOVA'
    else:
        f_stat, p_val = kruskal(*groups)
        test = 'Kruskal-Wallis'

    return {
        'Aspect': 'All',
        'Test': test,
        'Statistic': f_stat,
        'p-value': p_val
    }

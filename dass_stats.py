# dass_stats.py

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def run_paired_ttests(df):
    results = []

    for var in ['Depression', 'Anxiety', 'Stress']:
        before = df[f'{var}_Before']
        after = df[f'{var}_After']
        t_stat, p_val = stats.ttest_rel(before, after)

        results.append({
            'Variable': var,
            'Test': 'Paired t-test',
            't-statistic': round(t_stat, 4),
            'p-value': round(p_val, 4),
            'Significant at α=0.05': 'Yes' if p_val < 0.05 else 'No'
        })
    
    return pd.DataFrame(results)

def run_anova(df, dependent_vars, group_col='Group'):
    results = []

    for var in dependent_vars:
        groups = [group[var].dropna() for name, group in df.groupby(group_col)]
        f_stat, p_val = stats.f_oneway(*groups)

        results.append({
            'Variable': var,
            'Test': 'One-way ANOVA',
            'F-statistic': round(f_stat, 4),
            'p-value': round(p_val, 4),
            'Significant at α=0.05': 'Yes' if p_val < 0.05 else 'No'
        })

    return pd.DataFrame(results)

def plot_before_after(df, variable):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[[f'{variable}_Before', f'{variable}_After']])
    plt.title(f'{variable} Scores: Before vs After')
    plt.ylabel('Score')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

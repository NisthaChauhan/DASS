# enhanced_stats.py
# enhanced_statistics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene, f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from config import ALPHA, DASS_TAGS

def calculate_cohens_d(data1, data2):
    """Calculate Cohen's d effect size for paired samples"""
    diff = data1 - data2
    d = np.mean(diff) / np.std(diff, ddof=1)
    return abs(d)  # We're interested in the magnitude, not direction

def run_statistical_tests(merged_df):
    """
    Run comprehensive statistical tests on DASS data
    """
    # Store all statistical results
    paired_results = []
    
    # Run paired tests for each DASS subscale
    for aspect in ['Depression', 'Anxiety', 'Stress']:
        # Check normality of difference scores
        diff = merged_df[f'{aspect}_post'] - merged_df[f'{aspect}_pre']
        shapiro_stat, p_norm = shapiro(diff)
        
        # Run appropriate test based on normality
        if p_norm < ALPHA:
            # Non-parametric Wilcoxon test
            stat, p_val = wilcoxon(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 'Wilcoxon'
            effect_size = calculate_cohens_d(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            dof = 'N/A'
        else:
            # Parametric t-test
            stat, p_val = ttest_rel(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 't-test'
            effect_size = calculate_cohens_d(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            dof = len(merged_df) - 1
        
        # Calculate means and standard deviations
        pre_mean = merged_df[f'{aspect}_pre'].mean()
        pre_std = merged_df[f'{aspect}_pre'].std()
        post_mean = merged_df[f'{aspect}_post'].mean()
        post_std = merged_df[f'{aspect}_post'].std()
        mean_diff = pre_mean - post_mean
        
        # Determine effect size interpretation
        if effect_size < 0.2:
            effect_interp = "Negligible"
        elif effect_size < 0.5:
            effect_interp = "Small"
        elif effect_size < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"
        
        # Add results to list
        paired_results.append({
            'Aspect': aspect,
            'Test': test_type,
            'Pre Mean': pre_mean,
            'Pre SD': pre_std,
            'Post Mean': post_mean,
            'Post SD': post_std,
            'Mean Diff': mean_diff,
            'Statistic': stat,
            'df': dof,
            'p-value': p_val,
            'Significant': p_val < ALPHA,
            'Effect Size': effect_size,
            'Effect Interpretation': effect_interp,
            'Normality p': p_norm,
            'Normal': p_norm >= ALPHA
        })
    
    # Run ANOVA on change scores
    # Create a data frame for ANOVA analysis
    anova_data = pd.DataFrame({
        'Aspect': np.repeat(['Depression', 'Anxiety', 'Stress'], len(merged_df)),
        'Change': np.concatenate([
            merged_df['Depression_change'],
            merged_df['Anxiety_change'],
            merged_df['Stress_change']
        ]),
        'Subject': np.tile(np.arange(len(merged_df)), 3)
    })
    
    # Check ANOVA assumptions - normality and homogeneity of variance
    aspects_normal = True
    aspects_variance_equal = True
    
    # Check normality for each group
    for aspect in ['Depression', 'Anxiety', 'Stress']:
        _, p = shapiro(merged_df[f'{aspect}_change'])
        if p < ALPHA:
            aspects_normal = False
    
    # Check homogeneity of variance
    _, p_levene = levene(
        merged_df['Depression_change'],
        merged_df['Anxiety_change'],
        merged_df['Stress_change']
    )
    if p_levene < ALPHA:
        aspects_variance_equal = False
    
    # Run appropriate test based on assumptions
    if aspects_normal and aspects_variance_equal:
        # One-way ANOVA
        model = ols('Change ~ Aspect', data=anova_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        test_type = 'ANOVA'
        f_stat = anova_table['F'][0]
        p_val = anova_table['PR(>F)'][0]
        df_model = int(anova_table['df'][0])
        df_resid = int(anova_table['df'][1])
    else:
        # Non-parametric Kruskal-Wallis
        h_stat, p_val = kruskal(
            merged_df['Depression_change'],
            merged_df['Anxiety_change'],
            merged_df['Stress_change']
        )
        test_type = 'Kruskal-Wallis'
        f_stat = h_stat
        df_model = 2  # k-1 where k is number of groups
        df_resid = 3 * len(merged_df) - 3  # N-k
    
    # Create ANOVA result dictionary
    anova_result = {
        'Test': test_type,
        'Statistic': f_stat,
        'df_between': df_model,
        'df_within': df_resid,
        'p-value': p_val,
        'Significant': p_val < ALPHA,
        'Normality Assumption Met': aspects_normal,
        'Variance Assumption Met': aspects_variance_equal
    }
    
    return paired_results, anova_result

def display_statistical_results(merged_df):
    """
    Display comprehensive statistical results in both console output and visualizations
    """
    paired_results, anova_result = run_statistical_tests(merged_df)
    
    # Print console output for paired tests
    print("\n" + "="*80)
    print(f"STATISTICAL ANALYSIS RESULTS (α = {ALPHA})")
    print("="*80)
    
    print("\nPaired Tests (Pre vs Post):")
    print("-"*80)
    print(f"{'Aspect':<12} {'Test':<10} {'Pre':<15} {'Post':<15} {'Diff':<10} {'Statistic':<12} {'p-value':<10} {'Sig.':<5} {'Effect Size':<10} {'Interpretation'}")
    print("-"*80)
    
    for res in paired_results:
        print(f"{res['Aspect']:<12} "
              f"{res['Test']:<10} "
              f"{res['Pre Mean']:.2f}±{res['Pre SD']:.2f} "
              f"{res['Post Mean']:.2f}±{res['Post SD']:.2f} "
              f"{res['Mean Diff']:.2f} "
              f"{res['Statistic']:.3f} "
              f"{res['p-value']:.4f} "
              f"{'*' if res['Significant'] else '-':<5} "
              f"{res['Effect Size']:.2f} "
              f"{res['Effect Interpretation']}")
    
    # Print ANOVA results
    print("\nComparison Between Aspects (ANOVA/Kruskal-Wallis):")
    print("-"*80)
    print(f"Test: {anova_result['Test']}")
    print(f"Statistic: {anova_result['Statistic']:.3f}, df_between={anova_result['df_between']}, df_within={anova_result['df_within']}")
    print(f"p-value: {anova_result['p-value']:.4f} {'(Significant)' if anova_result['Significant'] else '(Not Significant)'}")
    print(f"Assumptions: Normality {'Met' if anova_result['Normality Assumption Met'] else 'Violated'}, "
          f"Equal Variance {'Met' if anova_result['Variance Assumption Met'] else 'Violated'}")
    
    # Create a visual table of results
    create_results_table(paired_results, anova_result)
    
    return paired_results, anova_result

def create_results_table(paired_results, anova_result):
    """
    Create a visual table of statistical results
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create data for the paired tests table
    table_data = []
    for res in paired_results:
        row = [
            res['Aspect'],
            res['Test'],
            f"{res['Pre Mean']:.2f} ± {res['Pre SD']:.2f}",
            f"{res['Post Mean']:.2f} ± {res['Post SD']:.2f}",
            f"{res['Mean Diff']:.2f}",
            f"{res['Statistic']:.3f}",
            f"{res['p-value']:.4f}",
            '*' if res['Significant'] else '-',
            f"{res['Effect Size']:.2f}",
            res['Effect Interpretation']
        ]
        table_data.append(row)
    
    # Add a row for the ANOVA/Kruskal-Wallis result
    table_data.append([
        'All Aspects', 
        anova_result['Test'],
        '-',
        '-',
        '-',
        f"{anova_result['Statistic']:.3f}",
        f"{anova_result['p-value']:.4f}",
        '*' if anova_result['Significant'] else '-',
        '-',
        '-'
    ])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=['Aspect', 'Test', 'Pre (M±SD)', 'Post (M±SD)', 'Difference', 
                  'Statistic', 'p-value', 'Sig.', 'Effect Size', 'Interpretation'],
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
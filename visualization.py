# Modified visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene, f_oneway, kruskal
from config import ALPHA, DASS_TAGS

def add_description(fig, text):
    """
    Add a separate description box below the main axes without using tight_layout.
    Uses a different approach to avoid incompatibility warnings.
    """
    # First adjust the main subplot to make room at the bottom
    plt.subplots_adjust(bottom=0.2)
    
    # Add text at the bottom with a bounding box
    fig.text(0.5, 0.05, text, ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))


def run_visualizations(merged_df):
    """
    Run all visualizations for the DASS pre/post analysis
    """
    plot_paired_bars_with_significance(merged_df)
    plot_boxplots(merged_df)
    plot_violin_plots(merged_df)
    plot_individual_changes(merged_df)
    plot_statistical_results(merged_df)
    plot_clinical_categories(merged_df)
    plot_difference_bars(merged_df)



def plot_difference_bars(merged_df):
    """
    Bar chart with pre, post, and difference bars
    """
    aspects = ['Depression', 'Anxiety', 'Stress']
    fig, ax = plt.subplots(figsize=(14, 9))
    positions = np.arange(len(aspects))
    bar_width = 0.28
    colors = ['#4e79a7', '#e15759', '#59a14f']

    # Compute statistics
    pre_means = [merged_df[f'{a}_pre'].mean() for a in aspects]
    post_means = [merged_df[f'{a}_post'].mean() for a in aspects]
    diffs = [post - pre for pre, post in zip(pre_means, post_means)]
    pre_sem = [merged_df[f'{a}_pre'].sem() for a in aspects]
    post_sem = [merged_df[f'{a}_post'].sem() for a in aspects]
    diff_sem = [np.sqrt(pre**2 + post**2) for pre, post in zip(pre_sem, post_sem)]

    # p-values
    p_vals = []
    for a in aspects:
        _, p_norm = shapiro(merged_df[f'{a}_change'])
        if p_norm < ALPHA:
            _, pval = wilcoxon(merged_df[f'{a}_pre'], merged_df[f'{a}_post'])
        else:
            _, pval = ttest_rel(merged_df[f'{a}_pre'], merged_df[f'{a}_post'])
        p_vals.append(pval)

    # Plot bars
    ax.bar(positions - bar_width, pre_means, bar_width, color=colors[0], yerr=pre_sem, capsize=5, label='Pre')
    ax.bar(positions, post_means, bar_width, color=colors[1], yerr=post_sem, capsize=5, label='Post')
    # Modified: plotting the difference bars in the negative direction
    ax.bar(positions + bar_width, diffs, bar_width, color=colors[2], yerr=diff_sem, capsize=5, label='Difference', bottom=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(aspects, fontsize=12)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('DASS-21 Scores: Pre, Post, and Differences', fontsize=16)

    # Labels
    for i, (pre, post, diff) in enumerate(zip(pre_means, post_means, diffs)):
        ax.text(i - bar_width, pre + 0.5, f'{pre:.1f}', ha='center')
        ax.text(i, post + 0.5, f'{post:.1f}', ha='center')
        # Adjusted label positioning for difference bars
        ax.text(i + bar_width, diff + 0.5 if diff > 0 else diff - 1.5, f'{diff:.1f}', ha='center')

    # Significance
    for i, (p, d) in enumerate(zip(p_vals, diffs)):
        # Adjusted positioning for significance indicators
        y0 = d - max(diff_sem[i], 2) if d > 0 else d + max(diff_sem[i], 2)
        sym = get_sig_symbol(p)
        ax.text(i + bar_width, y0, f'{sym}\n(p={p:.3f})', ha='center', va='center')

    ax.legend(loc='upper right')
    # Don't use tight_layout as it's incompatible with manually added text  # Adjusted for description space below

    desc = (
        "Pre (blue) & Post (red) scores with SEM. Green bars = difference (post-pre).\n"
        "Significance: * p<.05, ** p<.01, *** p<.001."
    )
    # Using the modified add_description function
    add_description(fig, desc)
    plt.savefig('enhanced_difference_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_sig_symbol(p):
    """Helper function for significance symbols"""
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

    

def plot_paired_bars_with_significance(merged_df):
    """
    Create paired bar charts with error bars and significance indicators
    """
    
    aspects = ['Depression', 'Anxiety', 'Stress']
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set width of bars
    bar_width = 0.35
    positions = np.arange(len(aspects))
    
    # Calculate means and standard errors
    pre_means = [merged_df[f'{aspect}_pre'].mean() for aspect in aspects]
    post_means = [merged_df[f'{aspect}_post'].mean() for aspect in aspects]
    pre_errors = [merged_df[f'{aspect}_pre'].std() / np.sqrt(len(merged_df)) for aspect in aspects]
    post_errors = [merged_df[f'{aspect}_post'].std() / np.sqrt(len(merged_df)) for aspect in aspects]
    
    # Create bars
    pre_bars = ax.bar(positions - bar_width/2, pre_means, bar_width, label='Pre-Test', 
                      color='skyblue', yerr=pre_errors, capsize=5)
    post_bars = ax.bar(positions + bar_width/2, post_means, bar_width, label='Post-Test', 
                       color='lightcoral', yerr=post_errors, capsize=5)
    
    # Add labels and title
    ax.set_xlabel('DASS Subscales', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Pre vs Post DASS Scores with Standard Error', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(aspects)
    ax.legend()
    
    # Add significance indicators
    for i, aspect in enumerate(aspects):
        # Run statistical test
        _, p_norm = shapiro(merged_df[f'{aspect}_change'])
        
        if p_norm < ALPHA:
            # Use non-parametric Wilcoxon if not normal
            _, pval = wilcoxon(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 'Wilcoxon'
        else:
            # Use parametric t-test if normal
            _, pval = ttest_rel(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 't-test'
        
        # Add significance asterisks
        if pval < 0.001:
            sig_text = '***'
        elif pval < 0.01:
            sig_text = '**'
        elif pval < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
            
        # Position the significance indicator
        max_height = max(pre_means[i], post_means[i])
        y_pos = max_height + max(pre_errors[i], post_errors[i]) + 0.5
        
        ax.annotate(sig_text, xy=(positions[i], y_pos), 
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=14)
        
        # Add test type and p-value as text below
        ax.annotate(f"{test_type}, p={pval:.3f}", xy=(positions[i], y_pos-1.5),
                   ha='center', va='bottom', fontsize=9)
    
    # Add significance legend
    ax.annotate('* p<0.05, ** p<0.01, *** p<0.001, ns: not significant', 
               xy=(1, -0.1), xycoords='axes fraction', ha='right', fontsize=10)
    
    # Replaced figtext with add_description
    description = (
        "Paired comparison showing pre-post scores with standard error bars.\n"
        "Significance indicators show results of appropriate parametric/non-parametric tests.\n"
        "Error bars represent standard error of the mean. Dashed lines connect matched pairs."
    )
    add_description(fig, description)
    
    # Don't use tight_layout as it's incompatible with manually added text
    plt.savefig('paired_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplots(merged_df):
    """
    Create boxplots showing distributions and outliers
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Reshape data for seaborn
    plot_data = pd.melt(merged_df, 
                        value_vars=['Depression_pre', 'Depression_post',
                                   'Anxiety_pre', 'Anxiety_post',
                                   'Stress_pre', 'Stress_post'],
                        var_name='Metric', value_name='Score')
    
    # Create custom labels
    plot_data['Category'] = plot_data['Metric'].apply(lambda x: x.split('_')[0])
    plot_data['Time'] = plot_data['Metric'].apply(lambda x: x.split('_')[1])
    
    # Create boxplot with seaborn
    sns.boxplot(x='Category', y='Score', hue='Time', data=plot_data, palette='pastel', ax=ax)
    
    ax.set_title('DASS Subscale Distributions and Outliers', fontsize=14)
    ax.set_xlabel('DASS Subscale', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(title='Time Point')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    description = (
        "Boxplots showing score distributions with median lines, quartile boxes, and outliers.\n"
        "Whiskers extend to 1.5×IQR. Circles show individual outliers beyond whiskers."
    )
    
    # Use add_description instead of figtext
    add_description(fig, description)
    
    # Don't use tight_layout as it's incompatible with manually added text
    plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_violin_plots(merged_df):
    """
    Create violin plots to visualize the full distribution shapes
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, aspect in enumerate(['Depression', 'Anxiety', 'Stress']):
        # Create dataframe for this aspect
        aspect_data = pd.DataFrame({
            'Pre-Test': merged_df[f'{aspect}_pre'],
            'Post-Test': merged_df[f'{aspect}_post']
        })
        
        # Create the violin plot
        sns.violinplot(data=aspect_data, ax=axes[i], palette='muted', inner='quartile')
        
        # Add a title and customize
        axes[i].set_title(f'{aspect} Score Distribution', fontsize=14)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Calculate and display means with text
        pre_mean = merged_df[f'{aspect}_pre'].mean()
        post_mean = merged_df[f'{aspect}_post'].mean()
        
        axes[i].annotate(f'Mean: {pre_mean:.2f}', xy=(0, pre_mean), 
                       xytext=(0.2, pre_mean+0.5), color='darkblue', fontsize=10)
        axes[i].annotate(f'Mean: {post_mean:.2f}', xy=(1, post_mean), 
                       xytext=(0.8, post_mean+0.5), color='darkblue', fontsize=10)
    
    description = (
        "Violin plots show probability density of scores. Wider sections = higher probability.\n"
        "White dots mark medians, black bars show IQR, thin lines extend to 1.5×IQR."
    )
    
    # Use add_description instead of figtext
    add_description(fig, description)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_individual_changes(merged_df):
    """
    Create line plots showing individual participant changes
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, aspect in enumerate(['Depression', 'Anxiety', 'Stress']):
        # Plot each participant's change as a line
        for idx, row in merged_df.iterrows():
            axes[i].plot([0, 1], [row[f'{aspect}_pre'], row[f'{aspect}_post']], 
                       'o-', alpha=0.3, color='gray')
        
        # Calculate and plot the mean change as a thick line
        pre_mean = merged_df[f'{aspect}_pre'].mean()
        post_mean = merged_df[f'{aspect}_post'].mean()
        axes[i].plot([0, 1], [pre_mean, post_mean], 'o-', linewidth=3, color='blue',
                   label='Mean')
        
        # Customize the plot
        axes[i].set_title(f'Individual {aspect} Score Changes', fontsize=14)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Pre-Test', 'Post-Test'])
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Calculate and display percentage of improvement
        improved = sum(merged_df[f'{aspect}_post'] < merged_df[f'{aspect}_pre'])
        pct_improved = improved / len(merged_df) * 100
        axes[i].annotate(f'{pct_improved:.1f}% Improved', xy=(0.5, axes[i].get_ylim()[1] * 0.9),
                       ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                         fc="white", ec="gray", alpha=0.8))
    
    description = (
        "Parallel coordinates plot showing individual participant trajectories.\n"
        "Gray lines: individual changes. Blue line: mean change trajectory. "
        "Positive slopes indicate improvement."
    )
    
    # Use add_description instead of figtext
    add_description(fig, description)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('individual_changes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_results(merged_df):
    """
    Create a visual table summarizing statistical test results
    """
    # Run statistical tests
    results = []
    
    for aspect in ['Depression', 'Anxiety', 'Stress']:
        # Normality test
        _, p_norm = shapiro(merged_df[f'{aspect}_change'])
        
        # Choose appropriate test
        if p_norm < ALPHA:
            # Non-parametric
            stat, p_val = wilcoxon(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 'Wilcoxon'
            stat_formatted = f"W={stat:.2f}"
        else:
            # Parametric
            stat, p_val = ttest_rel(merged_df[f'{aspect}_pre'], merged_df[f'{aspect}_post'])
            test_type = 't-test'
            stat_formatted = f"t={stat:.2f}"
        
        # Calculate effect size (Cohen's d)
        mean_diff = merged_df[f'{aspect}_change'].mean()
        std_diff = merged_df[f'{aspect}_change'].std()
        cohen_d = abs(mean_diff) / std_diff if std_diff != 0 else 0
        
        # Determine effect size interpretation
        if cohen_d < 0.2:
            effect = "Negligible"
        elif cohen_d < 0.5:
            effect = "Small"
        elif cohen_d < 0.8:
            effect = "Medium"
        else:
            effect = "Large"
        
        # Add to results
        results.append({
            'Aspect': aspect,
            'Test': test_type,
            'Statistic': stat_formatted,
            'p-value': p_val,
            'Cohen\'s d': cohen_d,
            'Effect': effect,
            'Significant': p_val < ALPHA
        })
    
    # ANOVA test on change scores
    aspects = ['Depression', 'Anxiety', 'Stress']
    anova_data = [merged_df[f'{aspect}_change'] for aspect in aspects]
    
    # Check ANOVA assumptions
    assumption_violations = []
    for i, aspect in enumerate(aspects):
        if shapiro(merged_df[f'{aspect}_change'])[1] < ALPHA:
            assumption_violations.append(f"{aspect} not normal")
    
    if len(assumption_violations) > 0:
        # Use non-parametric Kruskal-Wallis
        stat, p_val = kruskal(*anova_data)
        test_type = 'Kruskal-Wallis'
        stat_formatted = f"H={stat:.2f}"
    else:
        # Use parametric ANOVA
        stat, p_val = f_oneway(*anova_data)
        test_type = 'ANOVA'
        stat_formatted = f"F={stat:.2f}"
    
    # Add ANOVA/Kruskal result
    results.append({
        'Aspect': 'All (between aspects)',
        'Test': test_type,
        'Statistic': stat_formatted,
        'p-value': p_val,
        'Cohen\'s d': None,
        'Effect': None,
        'Significant': p_val < ALPHA
    })
    
    # Create a visual table
    fig, ax = plt.subplots(figsize=(12, len(results) + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for res in results:
        row = [
            res['Aspect'],
            res['Test'],
            res['Statistic'],
            f"{res['p-value']:.4f}" + (' *' if res['Significant'] else ''),
        ]
        
        # Add effect size info if available
        if res['Cohen\'s d'] is not None:
            row.append(f"{res['Cohen\'s d']:.2f}")
            row.append(res['Effect'])
        else:
            row.append('N/A')
            row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Aspect', 'Test', 'Statistic', 'p-value', 'Cohen\'s d', 'Effect Size'],
        loc='center',
        cellLoc='center',
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color significant results
    for i, res in enumerate(results):
        if res['Significant']:
            for j in range(6):
                cell = table[(i+1, j)]
                cell.set_facecolor('#d8f3dc')  # Light green
    
    # Add title
    plt.title('Statistical Test Results', fontweight='bold', fontsize=14, pad=20)
    
    # Add significance note using add_description
    description = '* Significant at α = 0.05 level'
    add_description(fig, description)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('statistical_results_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_clinical_categories(merged_df):
    """
    Visualize shifts in clinical categories from pre to post
    """
    # Define classification functions
    def classify_depression(score):
        if score <= 9: return 'Normal'
        elif score <= 13: return 'Mild'
        elif score <= 20: return 'Moderate'
        elif score <= 27: return 'Severe'
        else: return 'Extremely Severe'

    def classify_anxiety(score):
        if score <= 7: return 'Normal'
        elif score <= 9: return 'Mild'
        elif score <= 14: return 'Moderate'
        elif score <= 19: return 'Severe'
        else: return 'Extremely Severe'

    def classify_stress(score):
        if score <= 14: return 'Normal'
        elif score <= 18: return 'Mild'
        elif score <= 25: return 'Moderate'
        elif score <= 33: return 'Severe'
        else: return 'Extremely Severe'
    
    # Map scores to categories
    classifiers = {
        'Depression': classify_depression,
        'Anxiety': classify_anxiety,
        'Stress': classify_stress
    }
    
    # Calculate categories
    for aspect, classifier in classifiers.items():
        merged_df[f'{aspect}_pre_cat'] = merged_df[f'{aspect}_pre'].apply(classifier)
        merged_df[f'{aspect}_post_cat'] = merged_df[f'{aspect}_post'].apply(classifier)
    
    # Set up plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    categories = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    colors = ['#b5e48c', '#99d98c', '#52b69a', '#168aad', '#184e77']
    
    for i, aspect in enumerate(['Depression', 'Anxiety', 'Stress']):
        # Count frequencies
        pre_counts = merged_df[f'{aspect}_pre_cat'].value_counts().reindex(categories, fill_value=0)
        post_counts = merged_df[f'{aspect}_post_cat'].value_counts().reindex(categories, fill_value=0)
        
        # Calculate percentages
        total = len(merged_df)
        pre_pcts = (pre_counts / total) * 100
        post_pcts = (post_counts / total) * 100
        
        # X positions
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        axes[i].bar(x - width/2, pre_pcts, width, label='Pre-Test', color='skyblue')
        axes[i].bar(x + width/2, post_pcts, width, label='Post-Test', color='lightcoral')
        
        # Customize plot
        axes[i].set_title(f'{aspect} Clinical Categories', fontsize=14)
        axes[i].set_ylabel('Percentage of Participants (%)', fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(categories, rotation=45, ha='right')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend()
        
        # Add percentage values
        for j, (pre, post) in enumerate(zip(pre_pcts, post_pcts)):
            if pre > 0:
                axes[i].text(j - width/2, pre + 1, f'{pre:.1f}%', ha='center', va='bottom')
            if post > 0:
                axes[i].text(j + width/2, post + 1, f'{post:.1f}%', ha='center', va='bottom')
    
    description = (
        "Distribution of participants across clinical severity categories.\n"
        "Bars show percentage in each category before and after intervention."
    )
    add_description(fig, description)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('clinical_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
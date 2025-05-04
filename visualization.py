# enhanced_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure Seaborn style
sns.set_theme(style="whitegrid", palette="muted")

def plot_scores(merged_df):
    """Boxplot visualization with Seaborn styling"""
    plt.figure(figsize=(12, 6))
    df_melt = merged_df.melt(value_vars=['Depression_pre', 'Depression_post',
                                        'Anxiety_pre', 'Anxiety_post',
                                        'Stress_pre', 'Stress_post'],
                            var_name='Metric', value_name='Score')
    
    # Create custom ordering for the plot
    order = ['Depression_pre', 'Depression_post',
             'Anxiety_pre', 'Anxiety_post',
             'Stress_pre', 'Stress_post']
    
    ax = sns.boxplot(x='Metric', y='Score', data=df_melt, order=order,
                    hue='Metric', dodge=False, width=0.7)
    
    # Custom x-axis labels
    labels = [item.get_text().replace('_pre', '\n(Pre)').replace('_post', '\n(Post)') 
              for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    
    plt.title('Mental Health Scores Before and After Intervention\n(Boxplot Comparison)')
    plt.xlabel('')
    plt.ylabel('DASS Score')
    sns.despine(left=True)
    plt.tight_layout()

def plot_mean_differences(merged_df):
    """Pointplot showing mean differences with confidence intervals"""
    plt.figure(figsize=(10, 6))
    
    # Calculate mean differences
    aspects = ['Depression', 'Anxiety', 'Stress']
    diffs = []
    for aspect in aspects:
        diffs.append({
            'Aspect': aspect,
            'Mean Difference': merged_df[f'{aspect}_post'].mean() - merged_df[f'{aspect}_pre'].mean(),
            'CI_low': merged_df[f'{aspect}_post'].mean() - 1.96*merged_df[f'{aspect}_post'].std()/np.sqrt(len(merged_df)),
            'CI_high': merged_df[f'{aspect}_post'].mean() + 1.96*merged_df[f'{aspect}_post'].std()/np.sqrt(len(merged_df))
        })
    
    diff_df = pd.DataFrame(diffs)
    
    ax = sns.pointplot(x='Aspect', y='Mean Difference', data=diff_df,
                       join=False, scale=0.7, errwidth=2, capsize=0.1,
                       color='#2ecc71')
    
    # Add confidence intervals
    for i, row in diff_df.iterrows():
        plt.plot([i, i], [row['CI_low'], row['CI_high']], color='#2ecc71', lw=2)
    
    plt.axhline(0, color='#e74c3c', linestyle='--', alpha=0.7)
    plt.title('Mean Differences with 95% Confidence Intervals')
    plt.ylabel('Post - Pre Difference')
    sns.despine()
    plt.tight_layout()

def plot_distribution_shifts(merged_df):
    """Violin plots showing distribution changes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    aspects = ['Depression', 'Anxiety', 'Stress']
    
    for i, aspect in enumerate(aspects):
        df = pd.melt(merged_df[[f'{aspect}_pre', f'{aspect}_post']], 
                    var_name='Period', value_name='Score')
        df['Period'] = df['Period'].str.replace('_pre', 'Pre').replace('_post', 'Post')
        
        sns.violinplot(x='Period', y='Score', data=df, ax=axes[i],
                       palette='pastel', inner='quartile', cut=0)
        axes[i].set_title(f'{aspect} Distribution Shift')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score' if i == 0 else '')
    
    plt.suptitle('Pre/Post Intervention Distribution Changes')
    sns.despine()
    plt.tight_layout()

def plot_improvement_heatmap(merged_df):
    """Heatmap showing individual participant improvements"""
    plt.figure(figsize=(15, 8))
    
    # Calculate percentage changes
    change_df = merged_df[['Depression_change', 'Anxiety_change', 'Stress_change']]
    change_df = (change_df / merged_df[['Depression_pre', 'Anxiety_pre', 'Stress_pre']].values) * 100
    
    # Cluster participants by improvement patterns
    g = sns.clustermap(change_df.T, cmap='coolwarm', 
                       metric='correlation', standard_scale=1,
                       figsize=(15, 8), dendrogram_ratio=0.1)
    g.ax_heatmap.set_xlabel('Participants')
    g.ax_heatmap.set_ylabel('Aspects')
    g.ax_heatmap.set_title('Participant Improvement Patterns\n(% Change from Baseline)')
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.tight_layout()

def plot_severity_shifts(merged_df):
    """Visualization of severity category shifts using Seaborn"""
    aspects = ['Depression', 'Anxiety', 'Stress']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Process data for seaborn
    severity_data = []
    for aspect in aspects:
        for idx, row in merged_df.iterrows():
            severity_data.append({
                'Aspect': aspect,
                'Period': 'Pre',
                'Severity': classify_severity(row[f'{aspect}_pre'], aspect)
            })
            severity_data.append({
                'Aspect': aspect,
                'Period': 'Post',
                'Severity': classify_severity(row[f'{aspect}_post'], aspect)
            })
    
    severity_df = pd.DataFrame(severity_data)
    
    # Plot using countplot
    for i, aspect in enumerate(aspects):
        ax = axes[i]
        sns.countplot(x='Period', hue='Severity', 
                      data=severity_df[severity_df['Aspect'] == aspect],
                      ax=ax, palette='viridis', hue_order=[
                          'Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
        ax.set_title(f'{aspect} Severity Distribution')
        ax.set_xlabel('')
        ax.set_ylabel('Count' if i == 0 else '')
        ax.legend().set_visible(False)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Severity', 
               bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.suptitle('Severity Category Shifts')
    sns.despine()
    plt.tight_layout()

def create_compact_dashboard(merged_df):
    """Compact dashboard using Seaborn"""
    plt.figure(figsize=(18, 12))
    
    # Boxplots
    plt.subplot(2, 2, 1)
    plot_scores(merged_df)
    plt.title('Score Distributions', fontsize=12)
    
    # Mean differences
    plt.subplot(2, 2, 2)
    plot_mean_differences(merged_df)
    plt.title('Intervention Effects', fontsize=12)
    
    # Severity shifts
    plt.subplot(2, 2, 3)
    plot_severity_shifts(merged_df)
    plt.title('Severity Category Changes', fontsize=12)
    
    # Distribution shifts
    plt.subplot(2, 2, 4)
    plot_distribution_shifts(merged_df)
    plt.title('Distribution Changes', fontsize=12)
    
    plt.suptitle('Comprehensive Mental Health Intervention Analysis', fontsize=16)
    plt.tight_layout()

def run_visualizations(merged_df):
    """Run all Seaborn-based visualizations"""
    plot_scores(merged_df)
    plt.show()
    
    plot_mean_differences(merged_df)
    plt.show()
    
    plot_distribution_shifts(merged_df)
    plt.show()
    
    plot_improvement_heatmap(merged_df)
    plt.show()
    
    plot_severity_shifts(merged_df)
    plt.show()
    
    create_compact_dashboard(merged_df)
    plt.show()

# Helper function remains unchanged
def classify_severity(score, aspect):
    """Classify DASS scores into severity categories"""
    cutoffs = {
        'Depression': [9, 13, 20, 27],
        'Anxiety': [7, 9, 14, 19],
        'Stress': [14, 18, 25, 33]
    }
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    for i, cutoff in enumerate(cutoffs[aspect]):
        if score <= cutoff:
            return labels[i]
    return labels[-1]
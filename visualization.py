import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="pastel")

def classify_severity(score, aspect):
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

def plot_scores(df):
    melted = df.melt(value_vars=[
        'Depression_pre', 'Depression_post',
        'Anxiety_pre', 'Anxiety_post',
        'Stress_pre', 'Stress_post'
    ], var_name='Metric', value_name='Score')

    order = ['Depression_pre', 'Depression_post',
             'Anxiety_pre', 'Anxiety_post',
             'Stress_pre', 'Stress_post']

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Metric', y='Score', data=melted, order=order)
    ax.set_xticklabels([
        label.replace('_pre', '\n(Pre)').replace('_post', '\n(Post)')
        for label in order
    ])
    plt.title("Scores Before and After Intervention")
    plt.tight_layout()

def plot_mean_differences(df):
    aspects = ['Depression', 'Anxiety', 'Stress']
    records = []

    for asp in aspects:
        pre = df[f'{asp}_pre']
        post = df[f'{asp}_post']
        diff = post.mean() - pre.mean()
        ci = 1.96 * post.std() / np.sqrt(len(post))
        records.append({'Aspect': asp, 'Mean Difference': diff, 'CI': ci})

    diff_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    ax = sns.pointplot(x='Aspect', y='Mean Difference', data=diff_df,
                       errorbar='sd', color='green', markers='o', linestyles='none')
    for i, row in diff_df.iterrows():
        plt.plot([i, i], [row['Mean Difference'] - row['CI'], row['Mean Difference'] + row['CI']],
                 color='green', linewidth=2)
    plt.axhline(0, ls='--', color='red')
    plt.title("Mean Score Differences (Post - Pre)")
    plt.tight_layout()

def plot_distribution_shifts(df):
    aspects = ['Depression', 'Anxiety', 'Stress']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, aspect in enumerate(aspects):
        melted = df[[f'{aspect}_pre', f'{aspect}_post']].melt(var_name='Period', value_name='Score')
        melted['Period'] = melted['Period'].str.replace('_pre', 'Pre').replace('_post', 'Post')

        sns.violinplot(x='Period', y='Score', data=melted, ax=axes[i], inner='quartile', cut=0)
        axes[i].set_title(f'{aspect} Distribution')

    plt.suptitle("Distribution Shifts")
    plt.tight_layout()


def plot_severity_shifts(df):
    aspects = ['Depression', 'Anxiety', 'Stress']
    records = []
    for asp in aspects:
        for _, row in df.iterrows():
            records.append({'Aspect': asp, 'Period': 'Pre', 'Severity': classify_severity(row[f'{asp}_pre'], asp)})
            records.append({'Aspect': asp, 'Period': 'Post', 'Severity': classify_severity(row[f'{asp}_post'], asp)})

    sev_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, asp in enumerate(aspects):
        sns.countplot(data=sev_df[sev_df['Aspect'] == asp],
                      x='Period', hue='Severity',
                      palette='viridis', ax=axes[i],
                      hue_order=['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'])
        axes[i].set_title(f"{asp} Severity Shift")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Severity", bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()

def plot_distplot(df):
    aspects = ['Depression', 'Anxiety', 'Stress']

    for aspect in aspects:
        change_col = f'{aspect}_change'
        df['Depression_change'] = df['Depression_post'] - df['Depression_pre']
        df['Anxiety_change'] = df['Anxiety_post'] - df['Anxiety_pre']
        df['Stress_change'] = df['Stress_post'] - df['Stress_pre']

        if change_col in df.columns:
            if df[change_col].isnull().sum() == 0:  # Check if there are no NaN values
                sns.histplot(df[change_col], kde=True, bins=15, color='skyblue')
                plt.title(f'{aspect} Change Score Distribution')
                plt.xlabel('Change Score')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Warning: {change_col} contains NaN values.")
        else:
            print(f"Warning: {change_col} column not found.")


def run_visualizations(df):
    def save_fig(name):
        fig = plt.gcf()
        fig.set_size_inches(10, 8)  # 4:3 aspect ratio
        plt.tight_layout()
        fig.savefig(r"visualisations\{name}.png".format(name=name), dpi=300)
        plt.show()

    plot_scores(df)
    save_fig("plot_scores")

    plot_mean_differences(df)
    save_fig("plot_mean_differences")

    plot_distribution_shifts(df)
    save_fig("plot_distribution_shifts")

    plot_severity_shifts(df)
    save_fig("plot_severity_shifts")

    plot_distplot(df)
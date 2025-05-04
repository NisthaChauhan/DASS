# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_scores(merged_df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df[['Depression_pre', 'Depression_post', 
                                'Anxiety_pre', 'Anxiety_post',
                                'Stress_pre', 'Stress_post']])
    plt.xticks(ticks=[0,1,2,3,4,5], 
               labels=['Depression (Pre)', 'Depression (Post)',
                       'Anxiety (Pre)', 'Anxiety (Post)',
                       'Stress (Pre)', 'Stress (Post)'])
    plt.title('Mental Health Scores Before and After Intervention')
    plt.ylabel('Score')
    plt.show()

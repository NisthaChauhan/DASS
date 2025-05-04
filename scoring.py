# scoring.py

def compute_change_scores(merged_df):
    for aspect in ['Depression', 'Anxiety', 'Stress']:
        merged_df[f'{aspect}_change'] = merged_df[f'{aspect}_post'] - merged_df[f'{aspect}_pre']
    return merged_df

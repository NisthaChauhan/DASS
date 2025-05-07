# data_processing.py
import pandas as pd
from config import DASS_TAGS

def process_dass_data(filepath, sheet_name):
    # Read Excel sheet and clean data
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Clean column names and remove scores that will be recalculated
    df = df.rename(columns={'Participant_ID': 'Name'})
    df = df.drop(columns=[
        'Depression_Score', 
        'Anxiety_Score', 
        'Stress_Score',
        'Comments'
    ], errors='ignore')
    
    # Create new column names structure
    new_columns = ['Name', 'Age', 'Gender']
    for i in range(1, 22):
        prefix = None
        for category, indices in DASS_TAGS.items():
            if i in indices:
                prefix = category[0]
                break
        new_columns.append(f'{prefix}{i}' if prefix else f'Q{i}')
    
    # Verify column count matches
    if len(df.columns) != len(new_columns):
        raise ValueError(f"Column count mismatch. Expected {len(new_columns)} columns, got {len(df.columns)}")
    
    df.columns = new_columns
    
    # Calculate fresh scores
    for category in DASS_TAGS:
        cols = [f'{category[0]}{i}' for i in DASS_TAGS[category]]
        df[category] = df[cols].sum(axis=1)
    
    return df[['Name', 'Age', 'Gender', 'Depression', 'Anxiety', 'Stress']]
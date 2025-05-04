# data_processing.py

import pandas as pd
from config import DASS_TAGS

def process_dass_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Comments'])
    
    new_columns = ['Name']
    for i in range(1, 22):
        for key, indices in DASS_TAGS.items():
            if i in indices:
                new_columns.append(f"{key[0]}{i}")
                break
        else:
            new_columns.append(f'Q{i}')
    df.columns = new_columns

    df['Depression'] = df[[f'D{i}' for i in DASS_TAGS['Depression']]].sum(axis=1)
    df['Anxiety'] = df[[f'A{i}' for i in DASS_TAGS['Anxiety']]].sum(axis=1)
    df['Stress'] = df[[f'S{i}' for i in DASS_TAGS['Stress']]].sum(axis=1)

    return df[['Name', 'Depression', 'Anxiety', 'Stress']]

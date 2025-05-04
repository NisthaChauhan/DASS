'''
This script implements the DASS (Depression, Anxiety, and Stress Scale) questionnaire.

Depression: 3, 5, 10, 13, 16, 17, 21
Anxiety:2, 4, 7, 9, 15, 19, 20
Stress: 1, 6, 8, 11, 12, 14, 18

After summing the scores:
                    Depression  Anxiety     Stress
Normal              0-9         0-7         0-14
Mild                10-13       8-9         15-18
Moderate            14-20       10-14       19-25
Severe              21-27       15-19       26-33
Extremely Severe    28+         20+         34+
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dass=pd.read_csv(r"C:\Nistha\DASS\DASS.csv",encoding='unicode_escape')
dass=dass.drop(columns=['Name','Comments'],axis=1)

dass_tags = {
    'Depression': [3, 5, 10, 13, 16, 17, 21],
    'Anxiety': [2, 4, 7, 9, 15, 19, 20],
    'Stress': [1, 6, 8, 11, 12, 14, 18]
}

'''
    Rename columns
'''
new_column_names = []
for i in range(1, 22):
    if i in dass_tags['Depression']:
        new_column_names.append(f'D{i}')
    elif i in dass_tags['Anxiety']:
        new_column_names.append(f'A{i}')
    elif i in dass_tags['Stress']:
        new_column_names.append(f'S{i}')
    else:
        new_column_names.append(f'Q{i}')  
dass.columns = new_column_names
#print(dass.columns)

'''
    Dataframes for each category
'''

depression_frames =dass[[f'D{q}' for q in dass_tags['Depression']]]
anxiety_frames = dass[[f'A{q}' for q in dass_tags['Anxiety']]]
stress_frames = dass[[f'S{q}' for q in dass_tags['Stress']]]

'''
    Plotting
'''
def plot_category_frequency(df, cols, category_name):
    plt.figure(figsize=(10, 6))
    
    for response in range(4):
        counts = [df[col].value_counts().get(response, 0) for col in cols]
        x = range(len(cols))
        plt.bar([i + 0.2*response for i in x], counts, width=0.2, 
                label=f'Response {response}')
    
    plt.title(f'{category_name} Response Frequency by Question')
    plt.ylabel('Frequency')
    plt.xlabel('Question')
    plt.xticks([i + 0.3 for i in range(len(cols))], 
               [col.replace(category_name[0], '') for col in cols])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_category_stacked(df, cols, category_name):
    plt.figure(figsize=(10, 6))   
    data = {}
    for col in cols:
        data[col] = df[col].value_counts().sort_index()
    counts_df = pd.DataFrame(data)    
    counts_df.T.plot(kind='bar', stacked=True)    
    plt.title(f'{category_name} Response Distribution')
    plt.ylabel('Frequency')
    plt.xlabel('Question')
    plt.xticks(rotation=45)
    plt.legend(title='Response Value', labels=['0', '1', '2', '3'])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

'''
plot_category_frequency(dass, depression_frames, "Depression")
plot_category_frequency(dass, anxiety_frames, "Anxiety")
plot_category_frequency(dass, stress_frames, "Stress")

#STACKED
plot_category_stacked(dass, depression_frames, "Depression")
plot_category_stacked(dass, anxiety_frames, "Anxiety")
plot_category_stacked(dass, stress_frames, "Stress")
'''

'''
    Classification
'''
desc_depression = depression_frames.describe().T
desc_depression['Category'] = 'Depression'
desc_anxiety = anxiety_frames.describe().T
desc_anxiety['Category'] = 'Anxiety'
desc_stress = stress_frames.describe().T
desc_stress['Category'] = 'Stress'
full_description = pd.concat([desc_depression, desc_anxiety, desc_stress])
full_description = full_description[['Category'] + [col for col in full_description.columns if col != 'Category']]

dass['Depression_Score'] = depression_frames.sum(axis=1)
dass['Anxiety_Score'] = anxiety_frames.sum(axis=1)
dass['Stress_Score'] = stress_frames.sum(axis=1)

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

dass['Depression_Level'] = dass['Depression_Score'].apply(classify_depression)
dass['Anxiety_Level'] = dass['Anxiety_Score'].apply(classify_anxiety)
dass['Stress_Level'] = dass['Stress_Score'].apply(classify_stress)

'''
    Saving
'''
with pd.ExcelWriter(r"C:\Nistha\DASS\DASS1.xlsx") as writer:
    dass.to_excel(writer, sheet_name='OriginalDataWithScores', index=False)
    full_description.to_excel(writer, sheet_name='Category_Wise_DescriptiveStats', index_label='Question')    
    dass.to_excel(writer, sheet_name='FinalScores_Classified', index=False)
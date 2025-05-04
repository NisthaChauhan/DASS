# main.py

from data_processing import process_dass_data
from scoring import compute_change_scores
from statis import run_statistical_tests, run_anova

from visualization import plot_scores
import pandas as pd
from config import ALPHA

def main():
    pre = process_dass_data('PreTest.csv')
    post = process_dass_data('PostTest.csv')
    
    merged = pd.merge(pre, post, on='Name', suffixes=('_pre', '_post'))
    merged = compute_change_scores(merged)

    results, normality_warnings = run_statistical_tests(merged)
    results.append(run_anova(merged))

    plot_scores(merged)

    results_df = pd.DataFrame(results)
    print("\nStatistical Results:")
    print(results_df.to_string(index=False))

    print(f"\nSignificance level: α = {ALPHA}")
    for res in results:
        if res['Aspect'] == 'All':
            print(f"\nANOVA/Kruskal: {'Significant' if res['p-value'] < ALPHA else 'No significant'} difference (p={res['p-value']:.4f})")
        else:
            print(f"{res['Aspect']}: {'Reject H₀' if res['p-value'] < ALPHA else 'Fail to reject H₀'} ({res['Test']} p={res['p-value']:.4f})")
    
    if normality_warnings:
        print("\nNon-parametric tests used for:", ", ".join(normality_warnings))

if __name__ == "__main__":
    main()

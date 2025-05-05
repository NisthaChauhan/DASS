# main.py
#    pre = process_dass_data(r"C:\Nistha\DASS\datasets\PreTest.csv")
#    post = process_dass_data(r"C:\Nistha\DASS\datasets\PostTest.csv")
# updated_main.py

from data_processing import process_dass_data
from scoring import compute_change_scores
from statis import run_statistical_tests, run_anova
from visualization import run_visualizations
'''(
    plot_scores, 
    plot_mean_scores, 
    plot_individual_trajectories,
    plot_distribution_comparison,
    plot_correlation_heatmap,
    plot_change_histograms,
    plot_severity_shifts,
    plot_radar_chart,
    create_summary_dashboard
)'''

import pandas as pd
import matplotlib.pyplot as plt
from config import ALPHA

def main():
    # Process data
    pre = process_dass_data(r"C:\Nistha\DASS-1\datasets\PreTest.csv")
    post = process_dass_data(r"C:\Nistha\DASS-1\datasets\PostTest.csv")
    
    # Merge datasets
    merged = pd.merge(pre, post, on='Name', suffixes=('_pre', '_post'))
    merged = compute_change_scores(merged)

    # Run statistical tests
    results, normality_warnings = run_statistical_tests(merged)
    results.append(run_anova(merged))

    # Display statistical results
    # results_df = pd.DataFrame(results)
    # print("\nStatistical Results:")
    # print(results_df.to_string(index=False))

    # #print("*****************\nresult:\n",results,"\n*****************")
    # print(f"\nSignificance level: α = {ALPHA}")
    # for res in results:
    #     if res['Aspect'] == 'All':
    #         print(f"\nANOVA/Kruskal: {'Significant' if res['p-value'] < ALPHA else 'No significant'} difference (p={res['p-value']:.4f})")
    #     else:
    #         print(f"{res['Aspect']}: {'Reject H₀' if res['p-value'] < ALPHA else 'Fail to reject H₀'} ({res['Test']} p={res['p-value']:.4f})")

    # Display statistical results
    print("\n================ Paired Tests (t-test or Wilcoxon) per Aspect ================")
    print("Significance level: α = {:.2f}".format(ALPHA))
    print("{:<12} {:<10} {:>10} {:>12} {:>25}".format("Aspect", "Test", "Statistic", "p-value", "Decision"))
    print("-" * 75)

    for res in results:
        if res["Aspect"] != "All":
            stat = f"{res['Statistic']:.4f}" if not pd.isna(res['Statistic']) else "N/A"
            pval = f"{res['p-value']:.4f}"
            decision = "Reject H₀" if res['p-value'] < ALPHA else "Fail to reject H₀"
            print("{:<12} {:<10} {:>10} {:>12} {:>25}".format(res['Aspect'], res['Test'], stat, pval, decision))

    if normality_warnings:
        print("\nNote: Non-parametric Wilcoxon test used for:", ", ".join(normality_warnings))

    # Extract and display ANOVA/Kruskal result
    anova_result = next(res for res in results if res["Aspect"] == "All")

    print("\n================ Overall Comparison (ANOVA or Kruskal-Wallis) ================")
    print("{:<10} {:<20} {:>10} {:>12} {:>25}".format("Aspect", "Test", "Statistic", "p-value", "Decision"))
    print("-" * 75)
    stat = f"{anova_result['Statistic']:.4f}"
    pval = f"{anova_result['p-value']:.4f}"
    decision = "Significant difference" if anova_result['p-value'] < ALPHA else "No significant difference"
    print("{:<10} {:<20} {:>10} {:>12} {:>25}".format("All", anova_result['Test'], stat, pval, decision))

    if normality_warnings:
        print("\nNote: Non-parametric tests (Wilcoxon/Kruskal) used due to non-normality in:", ", ".join(normality_warnings))

    # Create visualizations
    print("\n================Generating visualizations")
    run_visualizations(merged)
    
if __name__ == "__main__":
    main()
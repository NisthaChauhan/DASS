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
    pre = process_dass_data(r"C:\Nistha\DASS\datasets\PreTest.csv")
    post = process_dass_data(r"C:\Nistha\DASS\datasets\PostTest.csv")
    
    # Merge datasets
    merged = pd.merge(pre, post, on='Name', suffixes=('_pre', '_post'))
    merged = compute_change_scores(merged)

    # Run statistical tests
    results, normality_warnings = run_statistical_tests(merged)
    results.append(run_anova(merged))

    # Display statistical results
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

    # Create visualizations
    print("\nGenerating visualizations...")
    run_visualizations(merged)
    plt.show()
    '''
    # Original boxplot
    plt.figure(figsize=(12, 6))
    plot_scores(merged)
    #plt.savefig("plots/boxplot_comparison.png")
    
    # Mean scores bar chart
    plot_mean_scores(merged)
    #plt.savefig("plots/mean_scores.png")

    
    # Distribution comparison
    plot_distribution_comparison(merged)
    #plt.savefig("plots/distribution_comparison.png")
    
    # Correlation heatmap
    plot_correlation_heatmap(merged)
    #plt.savefig("plots/correlation_heatmap.png")
    
    # Change score histograms
    plot_change_histograms(merged)
    #plt.savefig("plots/change_histograms.png")
    
    # Severity classification shifts
    plot_severity_shifts(merged)
    #plt.savefig("plots/severity_shifts.png")
    
    # Radar chart
    plot_radar_chart(merged)
    #plt.savefig("plots/radar_chart.png")
    
    # Summary dashboard
    create_summary_dashboard(merged)
    #plt.savefig("plots/summary_dashboard.png")
    plt.show()'''

if __name__ == "__main__":
    main()
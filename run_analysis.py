# run_analysis.py

from dass_stats import load_data, run_paired_ttests, run_anova, plot_before_after

# Load the post-intervention dataset
df = load_data("PostTest.csv")

# Run Paired t-tests
ttest_results = run_paired_ttests(df)
print("\n--- Paired t-test Results ---")
print(ttest_results)

# Optional: Run ANOVA if group column exists
if 'Group' in df.columns:
    anova_results = run_anova(df, ['Depression_After', 'Anxiety_After', 'Stress_After'])
    print("\n--- ANOVA Results ---")
    print(anova_results)

# Visualizations
for var in ['Depression', 'Anxiety', 'Stress']:
    plot_before_after(df, var)

# Save results
ttest_results.to_csv("PairedTTest_Results.csv", index=False)
if 'Group' in df.columns:
    anova_results.to_csv("ANOVA_Results.csv", index=False)

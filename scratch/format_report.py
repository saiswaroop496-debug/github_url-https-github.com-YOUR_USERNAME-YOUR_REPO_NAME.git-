import pandas as pd
import os

df = pd.read_csv('scratch/sim_vs_real_comparison.csv')

def get_outcome(score_str):
    if type(score_str) != str:
        return 'N/A'
    base_score = str(score_str).split(' ')[0]
    h, a = map(int, base_score.split('-'))
    if h > a: return 'Home Win'
    elif a > h: return 'Away Win'
    else: return 'Draw'

df['Original Outcome'] = df['Actual Score'].apply(get_outcome)
df['Predicted Outcome'] = df['Simulated Score'].apply(get_outcome)

# Reorder columns
df = df[['Date', 'Match', 'Actual Score', 'Simulated Score', 'Original Outcome', 'Predicted Outcome', 'Lam_H', 'Lam_A']]

# Accuracy
exact_scores = (df['Actual Score'] == df['Simulated Score'].apply(lambda x: x.split(' ')[0])).sum()
correct_outcomes = (df['Original Outcome'] == df['Predicted Outcome']).sum()
total = len(df)

report = f"""# 2026 World Cup: Simulation vs Reality Report

### Performance Summary
- **Total Matches Simulated:** {total}
- **Correct W/D/L Outcome Predictions:** {correct_outcomes} / {total} ({(correct_outcomes/total)*100:.1f}%)
- **Exact 90-Min Scoreline Matches:** {exact_scores} / {total} ({(exact_scores/total)*100:.1f}%)

> [!NOTE]
> The simulation was run as a single-pass Monte Carlo utilizing Bivariate Poisson parameters (Lam_H, Lam_A). Extra Time (AET) and Penalties (PEN) were mathematically simulated for knockout stage matches if tied at 90 minutes.

### Detailed Tabular Results

| Date | Match | Original Score | Predicted Score | Original Outcome | Predicted Outcome | $\lambda_H$ | $\lambda_A$ |
|------|-------|----------------|-----------------|------------------|-------------------|-------------|-------------|
"""

for idx, row in df.iterrows():
    report += f"| {row['Date']} | {row['Match']} | **{row['Actual Score']}** | **{row['Simulated Score']}** | {row['Original Outcome']} | {row['Predicted Outcome']} | {row['Lam_H']:.2f} | {row['Lam_A']:.2f} |\n"

with open('C:/Users/MMS Mandapeta/.gemini/antigravity/brain/3b87df0b-690d-4948-9190-44b802ca1365/simulation_accuracy_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print('Done!')

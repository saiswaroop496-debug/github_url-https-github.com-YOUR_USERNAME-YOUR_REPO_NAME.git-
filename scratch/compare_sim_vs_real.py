import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from inference import run_inference, _ensure_loaded
from models.poisson_dixon_coles import score_probability_matrix

warnings.filterwarnings('ignore')

# Load real results
df = pd.read_csv('data/international_results.csv')
wc_2026 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'] >= '2026-06-01')].copy()
wc_2026 = wc_2026.dropna(subset=['home_score', 'away_score'])
wc_2026 = wc_2026.sort_values('date')

results = []
_ensure_loaded(is_live=False)

for i, row in wc_2026.iterrows():
    home = row['home_team']
    away = row['away_team']
    real_h = int(row['home_score'])
    real_a = int(row['away_score'])
    
    # Run Inference
    res = run_inference(home, away)
    mc_params = res.get('mc_params', {'lam_h': 1.4, 'lam_a': 1.0, 'rho': 0.0})
    lam_h, lam_a, rho = mc_params['lam_h'], mc_params['lam_a'], mc_params['rho']
    
    # Bivariate Poisson Goals (90 min)
    matrix = score_probability_matrix(lam_h, lam_a, rho)
    flat_probs = matrix.flatten()
    flat_probs /= flat_probs.sum()
    idx = np.random.choice(len(flat_probs), p=flat_probs)
    sim_h, sim_a = divmod(idx, matrix.shape[1])
    
    sim_str = f"{sim_h}-{sim_a}"
    real_str = f"{real_h}-{real_a}"
    
    # Determine Knockout
    # Let's assume matches from June 26th onwards are knockouts (or last 16)
    is_knockout = row['date'] >= '2026-06-26'
    
    if is_knockout and sim_h == sim_a:
        # Extra Time
        et_lam_h = (lam_h / 3) * 0.85
        et_lam_a = (lam_a / 3) * 0.85
        et_matrix = score_probability_matrix(et_lam_h, et_lam_a, rho)
        et_flat = et_matrix.flatten()
        et_flat /= et_flat.sum()
        et_idx = np.random.choice(len(et_flat), p=et_flat)
        et_h, et_a = divmod(et_idx, et_matrix.shape[1])
        
        sim_h += et_h
        sim_a += et_a
        sim_str = f"{sim_h}-{sim_a} (AET)"
        
        if sim_h == sim_a:
            # Penalties
            prob_h_pen, prob_a_pen = 0.75, 0.75
            h_pen = np.random.binomial(5, prob_h_pen)
            a_pen = np.random.binomial(5, prob_a_pen)
            
            while h_pen == a_pen:
                h_pen += np.random.binomial(1, prob_h_pen)
                a_pen += np.random.binomial(1, prob_a_pen)
            
            sim_str = f"{sim_h}-{sim_a} (AET) [{h_pen}-{a_pen} PEN]"

    results.append({
        "Date": row['date'],
        "Match": f"{home} vs {away}",
        "Original Score": real_str,
        "Predicted Score": sim_str,
        "Lam_H": round(lam_h, 2),
        "Lam_A": round(lam_a, 2)
    })

res_df = pd.DataFrame(results)

def get_outcome(score_str):
    if type(score_str) != str:
        return 'N/A'
    base_score = str(score_str).split(' ')[0]
    h, a = map(int, base_score.split('-'))
    if h > a: return 'Home Win'
    elif a > h: return 'Away Win'
    else: return 'Draw'

res_df['Original Outcome'] = res_df['Original Score'].apply(get_outcome)
res_df['Predicted Outcome'] = res_df['Predicted Score'].apply(get_outcome)
res_df = res_df[['Date', 'Match', 'Original Score', 'Predicted Score', 'Original Outcome', 'Predicted Outcome', 'Lam_H', 'Lam_A']]

# Create a detailed markdown artifact
import os
artifact_dir = "C:/Users/MMS Mandapeta/.gemini/antigravity/brain/3b87df0b-690d-4948-9190-44b802ca1365"
with open(os.path.join(artifact_dir, 'simulation_accuracy_report.md'), 'w', encoding='utf-8') as f:
    f.write("# 2026 World Cup: Simulation vs Reality Report\n\n")
    
    # Calculate accuracy
    exact_match = 0
    correct_result = 0
    for idx, r in res_df.iterrows():
        act_h, act_a = map(int, r['Original Score'].split('-'))
        
        sim_base = r['Predicted Score'].split(' ')[0]
        sim_h, sim_a = map(int, sim_base.split('-'))
        
        if act_h == sim_h and act_a == sim_a:
            exact_match += 1
            
        if r['Original Outcome'] == r['Predicted Outcome']:
            correct_result += 1
            
    f.write(f"- **Total Matches Evaluated:** {len(res_df)}\n")
    f.write(f"- **Correct Match Outcome (W/D/L):** {correct_result} / {len(res_df)} ({(correct_result/len(res_df))*100:.1f}%)\n")
    f.write(f"- **Exact 90-Min Scoreline Match:** {exact_match} / {len(res_df)} ({(exact_match/len(res_df))*100:.1f}%)\n\n")
    f.write("### Comparison Table\n\n")
    
    try:
        f.write(res_df.to_markdown(index=False))
    except ImportError:
        for col in res_df.columns:
            f.write(f"| {col} ")
        f.write("|\n")
        f.write("|---" * len(res_df.columns) + "|\n")
        for idx, row in res_df.iterrows():
            for col in res_df.columns:
                f.write(f"| {row[col]} ")
            f.write("|\n")

print("Completed!")

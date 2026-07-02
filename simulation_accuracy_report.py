# simulation_accuracy_report.py
"""
Validate the minute-by-minute simulator against real WC match data.
Run: python simulation_accuracy_report.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from models.minute_simulation import run_simulation, SimulationConfig
from models.poisson_dixon_coles import score_probability_matrix, outcome_probs


def run_accuracy_report(data_path: str = "data/worldcup_matches.csv",
                         dc_params_path: str = "model_versions/latest/dc_params.joblib",
                         n_sims: int = 1000) -> dict:

    df = pd.read_csv(data_path)
    
    # dc_params is a joblib file now from V7.2
    import joblib
    dc_params = joblib.load(dc_params_path)

    # Use last 20% as test set (chronological)
    test_df = df.tail(int(len(df) * 0.20)).reset_index(drop=True)
    print(f"\n📊 Running simulation accuracy report on {len(test_df)} test matches...")

    records = []
    for _, row in test_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        attack = dc_params.get('attack', {})
        defense = dc_params.get('defense', {})
        home_adv = dc_params.get('home_adv', 0.3)
        
        lam_h = np.exp(attack.get(home_team, 0.3) - defense.get(away_team, -0.5) + home_adv)
        lam_a = np.exp(attack.get(away_team, 0.0) - defense.get(home_team, -0.5))

        config = SimulationConfig(
            lam_h=lam_h, lam_a=lam_a,
            rho=dc_params.get('rho', -0.13),
            n_simulations=n_sims
        )
        result = run_simulation(config)

        actual_goals_h = float(row.get('home_goals', 0) or 0)
        actual_goals_a = float(row.get('away_goals', 0) or 0)

        records.append({
            'home_team':        home_team,
            'away_team':        away_team,
            'actual_home':      actual_goals_h,
            'actual_away':      actual_goals_a,
            'sim_home':         result['expected']['goals_home'],
            'sim_away':         result['expected']['goals_away'],
            'actual_yellows':   float(row.get('home_yellow', 0) or 0) + float(row.get('away_yellow', 0) or 0),
            'sim_yellows':      result['expected']['yellows_home'] + result['expected']['yellows_away'],
            'actual_corners':   float(row.get('home_corners', 0) or 0) + float(row.get('away_corners', 0) or 0),
            'sim_corners':      result['expected']['corners_home'] + result['expected']['corners_away'],
            'most_likely_score': result['most_likely_score'],
            'sim_home_win':     result['home_win_prob'],
            'sim_draw':         result['draw_prob'],
            'sim_away_win':     result['away_win_prob'],
        })

    results_df = pd.DataFrame(records)

    # Compute accuracy metrics
    goal_mae_h = np.mean(np.abs(results_df['actual_home'] - results_df['sim_home']))
    goal_mae_a = np.mean(np.abs(results_df['actual_away'] - results_df['sim_away']))
    yellow_mae = np.mean(np.abs(results_df['actual_yellows'] - results_df['sim_yellows']))
    corner_mae = np.mean(np.abs(results_df['actual_corners'] - results_df['sim_corners']))

    # Outcome accuracy
    actual_outcomes = []
    for _, row in results_df.iterrows():
        if row['actual_home'] > row['actual_away']:
            actual_outcomes.append('Home Win')
        elif row['actual_home'] == row['actual_away']:
            actual_outcomes.append('Draw')
        else:
            actual_outcomes.append('Away Win')

    pred_outcomes = []
    for _, row in results_df.iterrows():
        if row['sim_home_win'] > max(row['sim_draw'], row['sim_away_win']):
            pred_outcomes.append('Home Win')
        elif row['sim_draw'] > row['sim_away_win']:
            pred_outcomes.append('Draw')
        else:
            pred_outcomes.append('Away Win')

    outcome_acc = np.mean([a == p for a, p in zip(actual_outcomes, pred_outcomes)])

    print("\n══════════════════════════════════════════════════════")
    print("  SIMULATION ACCURACY REPORT")
    print("══════════════════════════════════════════════════════")
    print(f"  Test matches:          {len(results_df)}")
    print(f"  Simulations per match: {n_sims:,}")
    print(f"\n  Goals MAE (home):      {goal_mae_h:.3f}  (target: <0.5)")
    print(f"  Goals MAE (away):      {goal_mae_a:.3f}  (target: <0.5)")
    print(f"  Yellows MAE:           {yellow_mae:.3f}  (target: <1.5)")
    print(f"  Corners MAE:           {corner_mae:.3f}  (target: <2.0)")
    print(f"\n  Outcome Accuracy:      {outcome_acc:.1%}")
    print(f"\n  Expected goals/match:  {results_df['sim_home'].mean():.2f} - {results_df['sim_away'].mean():.2f}")
    print(f"  Actual goals/match:    {results_df['actual_home'].mean():.2f} - {results_df['actual_away'].mean():.2f}")
    print("══════════════════════════════════════════════════════")

    # Save report
    results_df.to_csv("simulation_accuracy_report.csv", index=False)
    print(f"\n  Saved: simulation_accuracy_report.csv")
    return {'goal_mae_h': goal_mae_h, 'outcome_accuracy': outcome_acc}


def _get_lambda(team: str, dc_params: dict, param: str) -> float:
    """Get attack strength from DC params with fallback."""
    team_dict = dc_params.get(param, {})
    if isinstance(team_dict, dict) and team in team_dict:
        return float(np.exp(team_dict[team]))
    return 1.2   # neutral fallback


if __name__ == "__main__":
    run_accuracy_report()

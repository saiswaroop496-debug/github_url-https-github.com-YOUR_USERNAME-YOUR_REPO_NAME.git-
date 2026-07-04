import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from inference import run_inference, _ensure_loaded
from features.arbitrage_scanner import scan_pure_arbitrage, scan_statistical_arbitrage
from models.poisson_dixon_coles import score_probability_matrix
from features.yolo_vision import YOLOv8VisualModel

def generate_schedule(num_matches=80):
    """Generate a realistic schedule of 2026 World Cup matches completed till date."""
    top_teams = [
        "Argentina", "France", "Brazil", "England", "Belgium", "Netherlands",
        "Portugal", "Spain", "Italy", "Croatia", "Uruguay", "Colombia",
        "Morocco", "USA", "Mexico", "Germany", "Senegal", "Japan",
        "Switzerland", "Iran", "Denmark", "Korea Republic", "Australia", "Ecuador"
    ]
    
    start_date = datetime.datetime(2026, 6, 11)
    schedule = []
    
    for i in range(num_matches):
        match_date = start_date + datetime.timedelta(days=i // 4) # ~4 matches a day
        home = np.random.choice(top_teams)
        away = np.random.choice([t for t in top_teams if t != home])
        schedule.append({
            "match_id": f"2026_M{i+1}",
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "stage": "group" if i < 72 else "knockout"
        })
    return schedule

def simulate_2026_tournament():
    print("=" * 60)
    print(" 2026 WORLD CUP SIMULATION (Arbitrage + YOLOv8) - Till Date")
    print("=" * 60)
    
    _ensure_loaded(is_live=False)
    yolo = YOLOv8VisualModel()
    
    # 1. Generate Schedule
    schedule = generate_schedule(80) # 80 matches completed till July 4, 2026
    print(f"[*] Generated {len(schedule)} completed matches up to 2026-07-04.")
    
    bankroll_inr = 1000.0
    ledger = []
    bankroll_history = [bankroll_inr]
    
    for i, match in enumerate(schedule):
        home, away = match['home_team'], match['away_team']
        print(f"\n--- {match['date']} | {home} vs {away} ---")
        
        # A. YOLOv8 Visual Inference
        vision_feats = yolo.process_match_frames(match['match_id'])
        print(f"[YOLOv8] Detected Home Def Line: {vision_feats['tactical_visuals']['home_avg_defensive_line_m']}m, Pressing: {vision_feats['tactical_visuals']['global_pressing_intensity']}")
        
        # B. Run V7.5 Inference
        try:
            res = run_inference(home, away, stage=match['stage'])
        except Exception as e:
            print(f"[!] Inference failed for {home} vs {away}: {e}")
            continue
            
        prob_h = res.get('home_win_prob', 0.33)
        prob_d = res.get('draw_prob', 0.33)
        prob_a = res.get('away_win_prob', 0.34)
        
        # C. Generate Synthetic Market Odds (Introduce Vig + Random Noise + Rare Arbs)
        # We simulate market odds. Usually market is efficient, but sometimes it deviates.
        true_odds_h, true_odds_d, true_odds_a = 1/max(prob_h, 0.01), 1/max(prob_d, 0.01), 1/max(prob_a, 0.01)
        
        # Add bookmaker vig (105% overround typical) + noise
        noise_h = np.random.normal(1.0, 0.1)
        noise_d = np.random.normal(1.0, 0.1)
        noise_a = np.random.normal(1.0, 0.1)
        
        market_h = true_odds_h * 0.95 * noise_h
        market_d = true_odds_d * 0.95 * noise_d
        market_a = true_odds_a * 0.95 * noise_a
        
        # Inject Pure Arbitrage randomly (2% chance)
        if np.random.rand() < 0.02:
            market_h = market_h * 1.15
            market_d = market_d * 1.15
            market_a = market_a * 1.15
            
        df_match = pd.DataFrame([{
            'max_h': market_h, 'max_d': market_d, 'max_a': market_a
        }])
        
        # D. Arbitrage Scanner
        df_match = scan_pure_arbitrage(df_match)
        df_match = scan_statistical_arbitrage(df_match, pd.Series([prob_h]), pd.Series([prob_d]), pd.Series([prob_a]), bankroll=bankroll_inr)
        
        row = df_match.iloc[0]
        
        # Get MC params from inference
        mc_params = res.get('mc_params', {'lam_h': 1.4, 'lam_a': 1.0, 'rho': 0.0})
        lam_h, lam_a, rho = mc_params['lam_h'], mc_params['lam_a'], mc_params['rho']
        
        # Bivariate Poisson Goal Simulation (90 mins)
        matrix = score_probability_matrix(lam_h, lam_a, rho)
        flat_probs = matrix.flatten()
        flat_probs /= flat_probs.sum()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        h_goals, a_goals = divmod(idx, matrix.shape[1])
        
        # 1X2 market settles at 90 mins
        if h_goals > a_goals: outcome = 'H'
        elif h_goals < a_goals: outcome = 'A'
        else: outcome = 'D'
        
        match_str = f"{h_goals}-{a_goals}"
        
        # Knockout Logic: Extra Time & Penalties
        stage = 'group' if i < 48 else 'knockout'
        if stage == 'knockout' and outcome == 'D':
            # Extra Time (30 mins, 15% fatigue penalty)
            et_lam_h = (lam_h / 3) * 0.85
            et_lam_a = (lam_a / 3) * 0.85
            et_matrix = score_probability_matrix(et_lam_h, et_lam_a, rho)
            et_flat = et_matrix.flatten()
            et_flat /= et_flat.sum()
            et_idx = np.random.choice(len(et_flat), p=et_flat)
            et_h, et_a = divmod(et_idx, et_matrix.shape[1])
            
            total_h = h_goals + et_h
            total_a = a_goals + et_a
            match_str += f" ({total_h}-{total_a} AET)"
            
            if total_h == total_a:
                # Penalties
                prob_h_pen, prob_a_pen = 0.75, 0.75
                h_pen = np.random.binomial(5, prob_h_pen)
                a_pen = np.random.binomial(5, prob_a_pen)
                
                while h_pen == a_pen:
                    h_pen += np.random.binomial(1, prob_h_pen)
                    a_pen += np.random.binomial(1, prob_a_pen)
                
                match_str += f" [{h_pen}-{a_pen} PEN]"
        
        print(f"[Match Result] {outcome} won! (Score: {match_str})")
        
        # Calculate Betting PnL (Settles at 90 mins)
        pnl = 0.0
        
        # 1. Pure Arbitrage Execution
        if row['is_pure_arb']:
            print(f"[$$$] Pure Arbitrage Found! Implied: {row['arb_implied_prob']:.2%} | ROI: {row['pure_arb_roi_pct']:.2f}%")
            # Proportional Arb Staking logic from scanner (arb_stake_h_pct etc)
            arb_stake_h = bankroll_inr * row.get('arb_stake_h_pct', 0.0)
            arb_stake_d = bankroll_inr * row.get('arb_stake_d_pct', 0.0)
            arb_stake_a = bankroll_inr * row.get('arb_stake_a_pct', 0.0)
            
            # Arb settlement based on outcome
            arb_cost = arb_stake_h + arb_stake_d + arb_stake_a
            if outcome == 'H': arb_payout = arb_stake_h * market_h
            elif outcome == 'D': arb_payout = arb_stake_d * market_d
            else: arb_payout = arb_stake_a * market_a
            
            pnl += (arb_payout - arb_cost)
            
        # 2. Stat Arb Execution
        stake_h = row.get('stake_h', 0.0)
        stake_d = row.get('stake_d', 0.0)
        stake_a = row.get('stake_a', 0.0)
        
        if stake_h > 0:
            print(f"[*] Stat Edge Home! Stake: INR {stake_h:.2f} @ {market_h:.2f}")
            if outcome == 'H': pnl += stake_h * (market_h - 1)
            else: pnl -= stake_h
            
        if stake_d > 0:
            print(f"[*] Stat Edge Draw! Stake: INR {stake_d:.2f} @ {market_d:.2f}")
            if outcome == 'D': pnl += stake_d * (market_d - 1)
            else: pnl -= stake_d
            
        if stake_a > 0:
            print(f"[*] Stat Edge Away! Stake: INR {stake_a:.2f} @ {market_a:.2f}")
            if outcome == 'A': pnl += stake_a * (market_a - 1)
            else: pnl -= stake_a
            
        bankroll_inr += pnl
        bankroll_history.append(bankroll_inr)
        
        # Update Ledger
        ledger.append({
            "Match_ID": match['match_id'],
            "Date": match['date'],
            "Match": f"{home} vs {away}",
            "Score": match_str,
            "Model_Prob_H": round(prob_h, 3),
            "Model_Prob_D": round(prob_d, 3),
            "Model_Prob_A": round(prob_a, 3),
            "Market_Odds_H": round(market_h, 2),
            "Market_Odds_D": round(market_d, 2),
            "Market_Odds_A": round(market_a, 2),
            "YOLO_Pressing": vision_feats['tactical_visuals']['global_pressing_intensity'],
            "Result": outcome,
            "Pure_Arb": row['is_pure_arb'],
            "Stat_Stake_H": round(stake_h, 2),
            "Stat_Stake_D": round(stake_d, 2),
            "Stat_Stake_A": round(stake_a, 2),
            "PnL_INR": round(pnl, 2),
            "Balance_INR": round(bankroll_inr, 2)
        })
        
        print(f"[Balance] INR {bankroll_inr:.2f} (PnL: INR {pnl:.2f})")

    # Save Balance Sheet
    df_ledger = pd.DataFrame(ledger)
    df_ledger.to_csv("scratch/balance_sheet.csv", index=False)
    print("\n[✔] Saved Balance Sheet to scratch/balance_sheet.csv")
    
    # Plot Bankroll
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(bankroll_history)), bankroll_history, marker='o', color='green' if bankroll_history[-1] >= 1000 else 'red')
    plt.title("2026 World Cup Arbitrage Bankroll Growth (INR)")
    plt.xlabel("Matches Completed")
    plt.ylabel("Bankroll (₹)")
    plt.grid(True)
    plt.axhline(1000, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig("scratch/bankroll_chart.png")
    print("[✔] Saved Bankroll Chart to scratch/bankroll_chart.png")

if __name__ == "__main__":
    simulate_2026_tournament()

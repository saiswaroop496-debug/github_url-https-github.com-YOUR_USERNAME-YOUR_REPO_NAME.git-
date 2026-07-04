import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from data.scraper import load_and_enrich_dataset
from features.arbitrage_scanner import scan_pure_arbitrage, scan_statistical_arbitrage

def test_historical_arbitrage():
    print("=" * 60)
    print("  STATISTICAL ARBITRAGE BACKTEST SCANNER")
    print("=" * 60)

    # 1. Load Data with Odds
    print("Loading historical data & odds...")
    dc_df, df = load_and_enrich_dataset("data/worldcup_matches.csv")
    
    # 2. Add odds columns directly for testing if none found (Synthetic Injection)
    if 'max_h' not in df.columns or df['max_h'].isna().all():
        print("[!] No real odds found in data/football_data_odds/. Injecting synthetic odds for test...")
        # Inject synthetic odds where sum < 1.0 (Pure Arb) on row 0
        df['max_h'] = 2.50
        df['max_d'] = 3.50
        df['max_a'] = 3.80 # 1/2.5 + 1/3.5 + 1/3.8 = 0.4 + 0.285 + 0.263 = 0.948 < 1.0 (Arbitrage!)

        # Row 1: Stat Arb Edge
        # Let's say model thinks H is 60%, but market pays 2.50 (40%). 
        df.loc[1, 'max_h'] = 2.50
        df.loc[1, 'max_d'] = 3.00
        df.loc[1, 'max_a'] = 2.80

    # 3. Scan Pure Arbitrage
    print("\nScanning for Pure Arbitrage...")
    df = scan_pure_arbitrage(df)
    pure_arbs = df[df['is_pure_arb']]
    print(f"Found {len(pure_arbs)} pure arbitrage opportunities.")
    if not pure_arbs.empty:
        for idx, row in pure_arbs.head(5).iterrows():
            print(f"  [{row['date']}] {row['home_team']} vs {row['away_team']}")
            print(f"    Max Odds: H:{row['max_h']:.2f} D:{row['max_d']:.2f} A:{row['max_a']:.2f}")
            print(f"    Implied Prob: {row['arb_implied_prob']:.2%} | GUARANTEED ROI: {row['pure_arb_roi_pct']:.2f}%")

    # 4. Load Models & Scan Stat Arbitrage
    print("\nLoading Meta-Learner for Stat Arbitrage...")
    # In production `inference.py`, the real outputs are used.
    
    print("\nScanning for Statistical Arbitrage (Value Bets) using Quarter-Kelly...")
    # Simulated model probabilities (in real life, these come from `predict_proba`)
    preds_h = pd.Series([0.60 if i == 1 else 0.45 for i in range(len(df))])
    preds_d = pd.Series([0.25 if i == 1 else 0.30 for i in range(len(df))])
    preds_a = pd.Series([0.15 if i == 1 else 0.25 for i in range(len(df))])

    df = scan_statistical_arbitrage(df, preds_h, preds_d, preds_a, bankroll=1000.0)
    
    stat_arbs = df[df['total_stat_stake'] > 0]
    print(f"Found {len(stat_arbs)} Statistical Arbitrage (Value Edge) opportunities.")
    if not stat_arbs.empty:
        for idx, row in stat_arbs.head(5).iterrows():
            print(f"  [{row['date']}] {row['home_team']} vs {row['away_team']}")
            print(f"    Max Odds: H:{row['max_h']:.2f} D:{row['max_d']:.2f} A:{row['max_a']:.2f}")
            print(f"    Model Prob: H:{preds_h[idx]:.2%} D:{preds_d[idx]:.2%} A:{preds_a[idx]:.2%}")
            
            if row['stake_h'] > 0:
                print(f"    => Edge H: {row['stat_edge_h']:.2%} | Stake: ${row['stake_h']:.2f}")
            if row['stake_d'] > 0:
                print(f"    => Edge D: {row['stat_edge_d']:.2%} | Stake: ${row['stake_d']:.2f}")
            if row['stake_a'] > 0:
                print(f"    => Edge A: {row['stat_edge_a']:.2%} | Stake: ${row['stake_a']:.2f}")

    print("\n[✔] Arbitrage Scanner Backtest Complete.")

if __name__ == "__main__":
    test_historical_arbitrage()

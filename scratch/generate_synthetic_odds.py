import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from inference import run_inference, _ensure_loaded

def generate_odds():
    _ensure_loaded(is_live=False)
    df = pd.read_csv('data/international_results.csv')
    wc_2026 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'] >= '2026-06-01')].copy()
    wc_2026 = wc_2026.dropna(subset=['home_score', 'away_score']).sort_values('date')
    
    odds_data = []
    for i, row in wc_2026.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        
        # Get true model prob
        res = run_inference(home, away)
        ph, pd_draw, pa = res['home_win_prob'], res['draw_prob'], res['away_win_prob']
        
        # Generate 3 bookmakers with noise
        np.random.seed(hash(home + away) % 10000)
        
        bookies = []
        for b in range(3):
            # Bookies have up to 15% error to create statistical edges and arbitrage
            nh = ph * max(0.5, np.random.normal(1.0, 0.15))
            nd = pd_draw * max(0.5, np.random.normal(1.0, 0.15))
            na = pa * max(0.5, np.random.normal(1.0, 0.15))
            
            # Normalize
            total = nh + nd + na
            nh, nd, na = nh/total, nd/total, na/total
            
            # Add bias based on bookie index to ensure divergence
            bias = 0.05 if b == 0 else (-0.05 if b == 1 else 0)
            nh = max(0.05, nh + bias)
            nd = max(0.05, nd - bias*0.5)
            na = max(0.05, na - bias*0.5)
            
            total = nh + nd + na
            nh, nd, na = nh/total, nd/total, na/total
            
            # Convert to odds with 3% vig (sharper market, easier arbitrage)
            vig = 1.03
            bookies.append({
                'h': 1 / nh / vig,
                'd': 1 / nd / vig,
                'a': 1 / na / vig
            })
            
        max_h = max([b['h'] for b in bookies])
        max_d = max([b['d'] for b in bookies])
        max_a = max([b['a'] for b in bookies])
        
        odds_data.append({
            'date': date,
            'home_team': home,
            'away_team': away,
            'market_h': max(1.01, max_h),
            'market_d': max(1.01, max_d),
            'market_a': max(1.01, max_a)
        })
        
    pd.DataFrame(odds_data).to_csv('scratch/historical_odds_baseline.csv', index=False)
    print("Generated synthetic odds!")

if __name__ == '__main__':
    generate_odds()

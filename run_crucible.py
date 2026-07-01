import numpy as np
import pandas as pd
import random
import time
from models.poisson_dixon_coles import live_in_play_predict
from sklearn.metrics import brier_score_loss, log_loss

random.seed(42)
np.random.seed(42)

def generate_crucible_timeline(home_goals, away_goals):
    """
    Generates a high-frequency minute-by-minute (0 to 90) timeline of events.
    """
    events = {}
    
    # Place goals
    for _ in range(home_goals):
        events[random.randint(1, 90)] = {'type': 'Goal', 'team': 'home'}
    for _ in range(away_goals):
        events[random.randint(1, 90)] = {'type': 'Goal', 'team': 'away'}
        
    # Place random red cards (rare)
    if random.random() < 0.15:
        events[random.randint(10, 85)] = {'type': 'Red Card', 'team': 'home'}
    if random.random() < 0.15:
        events[random.randint(10, 85)] = {'type': 'Red Card', 'team': 'away'}
        
    # Generate minute-by-minute state
    timeline = []
    current_h = 0
    current_a = 0
    red_h = 0
    red_a = 0
    
    for minute in range(0, 91):
        if minute in events:
            ev = events[minute]
            if ev['type'] == 'Goal':
                if ev['team'] == 'home': current_h += 1
                else: current_a += 1
            elif ev['type'] == 'Red Card':
                if ev['team'] == 'home': red_h += 1
                else: red_a += 1
                
        timeline.append({
            'minute': minute,
            'home_goals': current_h,
            'away_goals': current_a,
            'red_cards': {'home': red_h, 'away': red_a}
        })
        
    return timeline

def run_crucible(num_matches=75):
    print("======================================================")
    print(f"  V7 CRUCIBLE: {num_matches}-MATCH HIGH-FREQUENCY SIMULATION")
    print("======================================================")
    
    df = pd.read_csv("data/international_results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['tournament'] == 'FIFA World Cup'].copy()
    df = df.dropna(subset=['home_score', 'away_score'])
    
    matches = df.sample(n=num_matches, random_state=42).to_dict('records')
    
    y_true_all = {m: [] for m in range(0, 91)}
    y_prob_all = {m: [] for m in range(0, 91)}
    
    start_time = time.time()
    
    for idx, match in enumerate(matches):
        h_score = int(match['home_score'])
        a_score = int(match['away_score'])
        
        true_class = 0 if h_score > a_score else (1 if h_score == a_score else 2)
        timeline = generate_crucible_timeline(h_score, a_score)
        
        for state in timeline:
            m = state['minute']
            
            # Use baseline lambdas for backtest
            lam_h = 1.4
            lam_a = 1.0
            
            probs = live_in_play_predict(
                lam_h_prematch=lam_h,
                lam_a_prematch=lam_a,
                elapsed=m,
                home_goals=state['home_goals'],
                away_goals=state['away_goals'],
                red_cards=state['red_cards']
            )
            
            prob_arr = [probs['home_win_prob'], probs['draw_prob'], probs['away_win_prob']]
            
            y_true_all[m].append(true_class)
            y_prob_all[m].append(prob_arr)
            
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
    print("Calculating minute-by-minute metrics...")
    
    with open("crucible_log.txt", "w") as f:
        f.write("Minute | Log-Loss | Brier Score\n")
        f.write("-" * 35 + "\n")
        for m in range(0, 91, 10): # Print every 10 mins to console, but calculate all
            y_t = y_true_all[m]
            y_p = np.array(y_prob_all[m])
            
            # Brier Score across 3 classes
            y_one_hot = np.zeros((len(y_t), 3))
            y_one_hot[np.arange(len(y_t)), y_t] = 1
            
            brier = np.mean(np.sum((y_p - y_one_hot)**2, axis=1))
            try:
                ll = log_loss(y_t, y_p, labels=[0, 1, 2])
            except ValueError:
                ll = 0.0
                
            f.write(f"  {m:02d}'  |  {ll:.4f}  |    {brier:.4f}\n")
            print(f"  {m:02d}'  |  Log-Loss: {ll:.4f}  |  Brier: {brier:.4f}")
            
    print("Crucible log written to crucible_log.txt")

if __name__ == "__main__":
    run_crucible()

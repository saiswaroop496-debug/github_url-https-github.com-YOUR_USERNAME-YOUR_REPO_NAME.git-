import numpy as np
import pandas as pd
import random
import time
from models.poisson_dixon_coles import live_in_play_predict
from sklearn.metrics import log_loss

random.seed(101)
np.random.seed(101)

def generate_detailed_timeline(home_goals, away_goals):
    events = {}
    for _ in range(home_goals):
        events.setdefault(random.randint(1, 90), []).append({'type': 'Goal', 'team': 'home'})
    for _ in range(away_goals):
        events.setdefault(random.randint(1, 90), []).append({'type': 'Goal', 'team': 'away'})
        
    for _ in range(random.randint(0, 6)):
        events.setdefault(random.randint(5, 85), []).append({'type': 'Yellow Card', 'team': random.choice(['home', 'away'])})
        
    if random.random() < 0.25:
        events.setdefault(random.randint(20, 80), []).append({'type': 'Red Card', 'team': random.choice(['home', 'away'])})

    for _ in range(random.randint(2, 10)):
        events.setdefault(random.randint(45, 85), []).append({'type': 'Substitution', 'team': random.choice(['home', 'away'])})
        
    timeline = []
    current_h, current_a, red_h, red_a = 0, 0, 0, 0
    momentum_h, momentum_a = 50.0, 50.0 
    
    for minute in range(0, 91):
        if minute in events:
            for ev in events[minute]:
                if ev['type'] == 'Goal':
                    if ev['team'] == 'home': current_h += 1; momentum_h += 20
                    else: current_a += 1; momentum_a += 20
                elif ev['type'] == 'Red Card':
                    if ev['team'] == 'home': red_h += 1; momentum_h -= 30
                    else: red_a += 1; momentum_a -= 30
                elif ev['type'] == 'Yellow Card':
                    if ev['team'] == 'home': momentum_h -= 5
                    else: momentum_a -= 5
                elif ev['type'] == 'Substitution':
                    if ev['team'] == 'home': momentum_h += 10
                    else: momentum_a += 10
                    
        momentum_h = momentum_h + (50 - momentum_h) * 0.1
        momentum_a = momentum_a + (50 - momentum_a) * 0.1
        
        timeline.append({
            'minute': minute, 'home_goals': current_h, 'away_goals': current_a,
            'red_cards': {'home': red_h, 'away': red_a},
            'live_state': {'momentum_index': (momentum_h - momentum_a) / 100.0, 'home_possession': 50 + (momentum_h - momentum_a)/2}
        })
    return timeline

def run_detailed_crucible(num_matches=75):
    print(f"Running 75-Match Detailed Simulation... Catching Vulnerabilities...")
    
    df = pd.read_csv("data/international_results.csv")
    df = df[df['tournament'] == 'FIFA World Cup'].copy()
    df = df.dropna(subset=['home_score', 'away_score'])
    matches = df.sample(n=num_matches, random_state=42).to_dict('records')
    
    vulnerabilities = []
    
    # Store probability arrays for metrics
    y_true_all = {m: [] for m in range(0, 91)}
    y_prob_all = {m: [] for m in range(0, 91)}
    
    for match_idx, match in enumerate(matches):
        h_score = int(match['home_score'])
        a_score = int(match['away_score'])
        true_class = 0 if h_score > a_score else (1 if h_score == a_score else 2)
        timeline = generate_detailed_timeline(h_score, a_score)
        
        lam_h, lam_a = 1.45, 1.30 
        
        for state in timeline:
            m = state['minute']
            try:
                probs = live_in_play_predict(
                    lam_h_prematch=lam_h, lam_a_prematch=lam_a,
                    elapsed=m, home_goals=state['home_goals'], away_goals=state['away_goals'],
                    red_cards=state['red_cards'], live_state=state['live_state']
                )
                
                # Check Simplex Vulnerability
                prob_sum = probs['home_win_prob'] + probs['draw_prob'] + probs['away_win_prob']
                if not np.isclose(prob_sum, 1.0, atol=1e-3):
                    vulnerabilities.append(f"Simplex Violation at match {match_idx} min {m}: Sum={prob_sum}")
                    
                # Check Edge Case Negatives
                if any(p < 0 for p in [probs['home_win_prob'], probs['draw_prob'], probs['away_win_prob']]):
                    vulnerabilities.append(f"Negative Probability at match {match_idx} min {m}")
                    
                prob_arr = [probs['home_win_prob'], probs['draw_prob'], probs['away_win_prob']]
                y_true_all[m].append(true_class)
                y_prob_all[m].append(prob_arr)
                
            except Exception as e:
                vulnerabilities.append(f"Crash at match {match_idx} min {m}: {str(e)}")

    print(f"Total Vulnerabilities Caught: {len(vulnerabilities)}")
    for v in set(vulnerabilities):
        print("VULNERABILITY:", v)

    # Dump a quick summary file
    with open("detailed_75_results.txt", "w") as f:
        f.write(f"VULNERABILITIES CAUGHT: {len(vulnerabilities)}\n")
        if not vulnerabilities:
            f.write("System passed 100% mathematically stable under heavy chaos injection.\n")
        
        f.write("\nAggregate Metrics across 75 matches (6750+ chaotic states):\n")
        for m in range(0, 91, 15):
            y_t = y_true_all[m]
            y_p = np.array(y_prob_all[m])
            # log_loss throws error if a class is entirely missing in small samples, so try/except it
            try:
                ll = log_loss(y_t, y_p, labels=[0, 1, 2])
            except ValueError:
                ll = float('nan')
            
            y_one_hot = np.zeros((len(y_t), 3))
            y_one_hot[np.arange(len(y_t)), y_t] = 1
            brier = np.mean(np.sum((y_p - y_one_hot)**2, axis=1))
            f.write(f"Min {m:02d}: LogLoss={ll:.4f}, Brier={brier:.4f}\n")

if __name__ == "__main__":
    run_detailed_crucible()

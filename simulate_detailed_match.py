import numpy as np
import pandas as pd
import random
import time
from models.poisson_dixon_coles import live_in_play_predict

random.seed(99)
np.random.seed(99)

def generate_detailed_timeline(home_goals, away_goals):
    """
    Generates a 0-90 timeline including goals, red cards, yellow cards, and substitutions.
    """
    events = {}
    
    # Place goals
    for _ in range(home_goals):
        events.setdefault(random.randint(1, 90), []).append({'type': 'Goal', 'team': 'home', 'desc': 'Home Goal!'})
    for _ in range(away_goals):
        events.setdefault(random.randint(1, 90), []).append({'type': 'Goal', 'team': 'away', 'desc': 'Away Goal!'})
        
    # Place cards
    for _ in range(random.randint(2, 5)):
        team = random.choice(['home', 'away'])
        events.setdefault(random.randint(5, 85), []).append({'type': 'Yellow Card', 'team': team, 'desc': f'{team.title()} Yellow'})
        
    if random.random() < 0.20:
        team = random.choice(['home', 'away'])
        events.setdefault(random.randint(20, 80), []).append({'type': 'Red Card', 'team': team, 'desc': f'{team.title()} RED CARD!'})

    # Place subs
    for _ in range(random.randint(4, 8)):
        team = random.choice(['home', 'away'])
        events.setdefault(random.randint(45, 85), []).append({'type': 'Substitution', 'team': team, 'desc': f'{team.title()} Sub'})
        
    timeline = []
    current_h, current_a = 0, 0
    red_h, red_a = 0, 0
    momentum_h, momentum_a = 50.0, 50.0 # Base momentum
    
    for minute in range(0, 91):
        minute_events = []
        if minute in events:
            for ev in events[minute]:
                minute_events.append(ev['desc'])
                if ev['type'] == 'Goal':
                    if ev['team'] == 'home': 
                        current_h += 1
                        momentum_h += 20
                    else: 
                        current_a += 1
                        momentum_a += 20
                elif ev['type'] == 'Red Card':
                    if ev['team'] == 'home': 
                        red_h += 1
                        momentum_h -= 30
                    else: 
                        red_a += 1
                        momentum_a -= 30
                elif ev['type'] == 'Yellow Card':
                    if ev['team'] == 'home': momentum_h -= 5
                    else: momentum_a -= 5
                elif ev['type'] == 'Substitution':
                    if ev['team'] == 'home': momentum_h += 10
                    else: momentum_a += 10
                    
        # Momentum decay back to 50
        momentum_h = momentum_h + (50 - momentum_h) * 0.1
        momentum_a = momentum_a + (50 - momentum_a) * 0.1
        
        timeline.append({
            'minute': minute,
            'home_goals': current_h,
            'away_goals': current_a,
            'red_cards': {'home': red_h, 'away': red_a},
            'events': ", ".join(minute_events),
            'live_state': {
                'momentum_index': (momentum_h - momentum_a) / 100.0,
                'home_possession': 50 + (momentum_h - momentum_a)/2
            }
        })
        
    return timeline

def simulate_detailed():
    print("Running 5x Minute-by-Minute Detailed Simulation...")
    
    # Simulate a blockbuster match (e.g. Argentina vs France 3-3)
    timeline = generate_detailed_timeline(3, 3)
    lam_h = 1.45
    lam_a = 1.40
    
    report_lines = []
    report_lines.append("| Minute | Score | Events | P(Home) | P(Draw) | P(Away) | Momentum |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for state in timeline:
        m = state['minute']
        if m % 5 == 0 or state['events'] != "":
            probs = live_in_play_predict(
                lam_h_prematch=lam_h,
                lam_a_prematch=lam_a,
                elapsed=m,
                home_goals=state['home_goals'],
                away_goals=state['away_goals'],
                red_cards=state['red_cards'],
                live_state=state['live_state']
            )
            
            p_h = probs['home_win_prob']
            p_d = probs['draw_prob']
            p_a = probs['away_win_prob']
            
            score = f"{state['home_goals']}-{state['away_goals']}"
            evs = state['events'] if state['events'] else "-"
            mom = f"{state['live_state']['momentum_index']:+.2f}"
            
            if evs != "-" or m % 5 == 0:
                report_lines.append(f"| {m:02d}' | {score} | {evs} | {p_h:.3f} | {p_d:.3f} | {p_a:.3f} | {mom} |")
            
    with open("detailed_sim_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("Report generated at detailed_sim_report.md")

if __name__ == "__main__":
    simulate_detailed()

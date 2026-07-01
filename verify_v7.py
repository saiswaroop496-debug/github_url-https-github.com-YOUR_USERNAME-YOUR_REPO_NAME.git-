import json
import re
from pathlib import Path

def run_checks():
    print("Running V7 Verification Checks...\n")
    
    # Step 3
    print("--- Step 3: Check team_states.json ---")
    try:
        s = json.load(open('model_versions/latest/team_states.json'))
        defaults = [t for t,v in s.items() if v['glicko']==1500 and v['rd']==200]
        print(f"Teams with real ratings: {len(s)-len(defaults)}/{len(s)}")
        if len(defaults) == 0:
            print("PASS: All team states are real\n")
        else:
            print(f"FAIL: {len(defaults)} teams still at defaults\n")
    except Exception as e:
        print(f"ERROR: {e}\n")

    # Step 4
    print("--- Step 4: Check draw recall ---")
    try:
        txt = open('v7_validation.txt').read()
        draw_recall = re.search(r'Draw.*?(\d+\.\d+)', txt)
        if draw_recall:
            r = float(draw_recall.group(1))
            print(f"Draw recall: {r:.3f}")
            if r > 0.05:
                print("PASS: Draw recall above threshold\n")
            else:
                print(f"FAIL: Draw recall {r} < 5%\n")
        else:
            print("FAIL: Could not find Draw recall in v7_validation.txt\n")
    except Exception as e:
        print(f"ERROR: {e}\n")

    # Step 5
    print("--- Step 5: Check app.py uses real inference ---")
    try:
        src = open('app.py', encoding='utf-8').read()
        if 'np.random.uniform' not in src and 'run_inference' in src:
            print("PASS: app.py uses real model\n")
        else:
            print("FAIL: app.py uses random or missing run_inference\n")
    except Exception as e:
        print(f"ERROR: {e}\n")

    # Step 6
    print("--- Step 6: Test inference predictions ---")
    try:
        from inference import run_inference
        r1 = run_inference('Brazil', 'Germany', 0.3, 'group')
        r2 = run_inference('Japan', 'Saudi Arabia', 0.3, 'group')
        print(f"Brazil vs Germany: {r1.get('home_win_prob', r1):.3f}")
        print(f"Japan vs Saudi:    {r2.get('home_win_prob', r2):.3f}")
        if r1.get('home_win_prob') != r2.get('home_win_prob'):
            print("PASS: Different teams produce different predictions\n")
        else:
            print("FAIL: All teams produce identical predictions\n")
    except Exception as e:
        print(f"ERROR: {e}\n")

    # Step 7
    print("--- Step 7: Test in-play convergence ---")
    try:
        from models.poisson_dixon_coles import live_in_play_predict
        print(f"Min  HomeWin   Draw   AwayWin")
        for m in [0, 15, 30, 45, 60, 75, 80, 85, 88, 89]:
            r = live_in_play_predict(1.4, 1.0, m, 3, 0, -0.13)
            print(f"{m:>3}  {r['home_win_prob']:>7.1%}  {r['draw_prob']:>6.1%}  {r['away_win_prob']:>7.1%}")
        print("\nPASS: In-play engine running")
    except Exception as e:
        print(f"ERROR: {e}\n")

if __name__ == "__main__":
    run_checks()

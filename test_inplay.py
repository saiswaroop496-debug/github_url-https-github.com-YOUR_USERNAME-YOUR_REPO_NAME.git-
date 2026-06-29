# test_inplay.py — Paste in terminal to verify convergence

from models.poisson_dixon_coles import live_in_play_predict

print("Simulating match: Home leading 3-0")
print(f"{'Minute':>8} | {'Home Win':>10} | {'Draw':>8} | {'Away Win':>10}")
print("-" * 45)

for minute in [0, 15, 30, 45, 60, 75, 80, 85, 88, 89]:
    r = live_in_play_predict(
        lam_h_prematch=1.4,   # typical home expected goals
        lam_a_prematch=1.0,
        elapsed=minute,
        home_goals=3,
        away_goals=0,
        rho=-0.13
    )
    print(f"{minute:>8}' | {r['home_win_prob']:>9.1%} | "
          f"{r['draw_prob']:>7.1%} | {r['away_win_prob']:>9.1%}")

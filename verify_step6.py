from betting.stat_arb import compute_cross_market_arb
import numpy as np

# Build a simple DC score matrix for test
from models.poisson_dixon_coles import score_probability_matrix
matrix = score_probability_matrix(1.4, 1.0, -0.13)

result = compute_cross_market_arb(
    odds_h=2.10, odds_d=3.40, odds_a=3.20,
    ah_home_odds=1.95, ah_handicap=-0.5,
    over_odds=1.85, under_odds=2.05,
    score_matrix=matrix
)
print('Stat-Arb result:')
for k, v in result.items():
    print(f'  {k}: {v}')
assert 'best_bet' in result, 'FAIL: best_bet missing from stat_arb output'
print('✅ Stat-Arb working')

from betting.information_ratio_kelly import ir_adjusted_kelly
kelly = ir_adjusted_kelly(
    model_prob=0.52,
    decimal_odds=2.10,
    historical_edges=[0.02, 0.025, 0.03, 0.018, 0.022, 0.028, 0.015, 0.032, 0.019, 0.024],
    ece=0.08,
    fold_std=0.03
)
print(f'IR Kelly: {kelly}')
assert kelly['stake_fraction'] > 0, 'FAIL: IR Kelly returned 0 stake on positive edge'
print('✅ IR Kelly working')

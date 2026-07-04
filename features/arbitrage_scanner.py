import pandas as pd
import numpy as np
from betting.kelly_criterion import sharpe_kelly, evaluate_bet

def scan_pure_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects pure arbitrage opportunities (risk-free) by finding the max odds
    across all available bookmakers for Home, Draw, Away.
    Requires max_h, max_d, max_a columns to be present.
    """
    if 'max_h' not in df.columns or 'max_d' not in df.columns or 'max_a' not in df.columns:
        return df

    # Implied probability of the best synthetic market
    df['arb_implied_prob'] = (1 / df['max_h']) + (1 / df['max_d']) + (1 / df['max_a'])
    
    # Is it an arbitrage? (Sum < 1.0)
    df['is_pure_arb'] = df['arb_implied_prob'] < 1.0

    df['pure_arb_roi_pct'] = np.where(
        df['is_pure_arb'],
        ((1 / df['arb_implied_prob']) - 1) * 100,
        0.0
    )

    # Pure Arb Staking Logic (Proportional to odds to guarantee equal profit)
    # Payout = Stake_H * max_h = Stake_D * max_d = Stake_A * max_a = Target_Return
    # Total_Stake = Stake_H + Stake_D + Stake_A
    # Stake_i = (Total_Stake / arb_implied_prob) * (1 / Odds_i)
    # We will assume a total budget of 10% of bankroll per arb event.
    
    df['arb_stake_h_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_h']), 0.0)
    df['arb_stake_d_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_d']), 0.0)
    df['arb_stake_a_pct'] = np.where(df['is_pure_arb'], (0.10 / df['arb_implied_prob']) * (1 / df['max_a']), 0.0)

    return df

def scan_statistical_arbitrage(df: pd.DataFrame, model_preds_h: pd.Series, model_preds_d: pd.Series, model_preds_a: pd.Series, bankroll: float = 1000.0) -> pd.DataFrame:
    """
    Compares the model's predicted probabilities against the BEST available market odds.
    If the model's probability > implied probability of the odds, there is a statistical edge (value).
    Integrates advanced Kelly sizing (Sharpe Kelly, ECE penalties) and mathematically sound 
    Simultaneous Kelly for mutually exclusive outcomes.
    """
    if 'max_h' not in df.columns:
        return df

    df['stat_edge_h'] = model_preds_h - (1 / df['max_h'])
    df['stat_edge_d'] = model_preds_d - (1 / df['max_d'])
    df['stat_edge_a'] = model_preds_a - (1 / df['max_a'])
    
    stakes_h, stakes_d, stakes_a = [], [], []
    
    for i in range(len(df)):
        ph, pd, pa = model_preds_h.iloc[i], model_preds_d.iloc[i], model_preds_a.iloc[i]
        oh, od, oa = df['max_h'].iloc[i], df['max_d'].iloc[i], df['max_a'].iloc[i]
        
        sh, sd, sa = 0.0, 0.0, 0.0
        
        valid_bets = []
        for outcome, p, o in [('H', ph, oh), ('D', pd, od), ('A', pa, oa)]:
            if np.isnan(p) or np.isnan(o) or o <= 1.0:
                continue
                
            # Use 1/o as a base novig_prob for edge calculation
            res = evaluate_bet(market_type=outcome, model_prob=p, decimal_odds=o, novig_prob=1/o, bankroll=bankroll, min_edge=0.0)
            if res.get('action') == 'BET':
                valid_bets.append({
                    'outcome': outcome,
                    'p': p,
                    'o': o,
                    'res': res
                })
        
        if len(valid_bets) > 0:
            # Sort by expected return (p * o) descending to build optimal simultaneous set
            valid_bets.sort(key=lambda x: x['p'] * x['o'], reverse=True)
            
            simul_bets = []
            for b in valid_bets:
                temp = simul_bets + [b]
                R = 1.0 - sum(1/x['o'] for x in temp)
                sum_p = sum(x['p'] for x in temp)
                
                if R > 0:
                    f_new = b['p'] - ((1 - sum_p) / (b['o'] * R))
                    if f_new > 0:
                        simul_bets.append(b)
                else:
                    # If R <= 0, we cannot add this bet into the mutually exclusive set sensibly
                    pass
            
            # Now calculate final stakes for the selected simultaneous bets
            if len(simul_bets) > 1:
                R_final = 1.0 - sum(1/x['o'] for x in simul_bets)
                sum_p_final = sum(x['p'] for x in simul_bets)
                
                for b in simul_bets:
                    # Mathematically sound raw simultaneous Kelly fraction
                    f_simul = b['p'] - ((1 - sum_p_final) / (b['o'] * R_final))
                    
                    # Apply Sharpe and Penalties in the same way sharpe_kelly does
                    f_base = min(f_simul, b['res']['sharpe_kelly']) / 4.0
                    f_penalized = f_base * b['res']['ece_multiplier'] * b['res']['stab_multiplier']
                    f_capped = min(max(0.0, f_penalized), 0.05) # Max 5% of bankroll
                    stake = f_capped * bankroll
                    
                    if b['outcome'] == 'H': sh = stake
                    elif b['outcome'] == 'D': sd = stake
                    elif b['outcome'] == 'A': sa = stake
            elif len(simul_bets) == 1:
                # Single bet, just use the stake from evaluate_bet
                b = simul_bets[0]
                stake = b['res']['stake']
                if b['outcome'] == 'H': sh = stake
                elif b['outcome'] == 'D': sd = stake
                elif b['outcome'] == 'A': sa = stake
                
        stakes_h.append(sh)
        stakes_d.append(sd)
        stakes_a.append(sa)
        
    df['stat_stake_h'] = stakes_h
    df['stat_stake_d'] = stakes_d
    df['stat_stake_a'] = stakes_a

    # Calculate stakes (Quarter Kelly)
    df['stake_h'] = df['stat_stake_h']
    df['stake_d'] = df['stat_stake_d']
    df['stake_a'] = df['stat_stake_a']

    return df

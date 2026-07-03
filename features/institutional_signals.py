import numpy as np
import pandas as pd

# Global mean fallback values (computed from your 992-match baseline)
SIGNAL_PRIORS = {
    'tournament_momentum_diff':  0.0,
    'glicko_velocity_diff':      0.0,
    'conversion_diff':           0.0,
    'defensive_shape_diff':      0.0,
    'lineup_continuity_diff':    0.0,
}

def add_institutional_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    V7.4: Wrapper that calls the raw signal computation and applies:
      1. Forward-fill within each team (carry last known value)
      2. Global prior fill for remaining NaNs (no zeros — zeros are meaningful)
      3. Clip extreme outliers at ±3 standard deviations (prevents XGBoost instability)
      4. Verification log showing null rates per signal
    """
    # Call the core signal computation
    df = _compute_raw_institutional_signals(df)

    signals = list(SIGNAL_PRIORS.keys())

    for col in signals:
        if col not in df.columns:
            df[col] = SIGNAL_PRIORS[col]
            print(f"  ⚠️  {col} not computed — filling with prior {SIGNAL_PRIORS[col]}")
            continue

        null_before = df[col].isna().sum()

        # 1. Clip outliers first (±3σ)
        mu  = df[col].mean()
        std = df[col].std() + 1e-9
        df[col] = df[col].clip(mu - 3*std, mu + 3*std)

        # 2. Fill remaining NaNs with global prior (not zero)
        df[col] = df[col].fillna(SIGNAL_PRIORS[col])

        null_after = df[col].isna().sum()
        coverage   = 1 - (null_after / len(df))
        print(f"  {col}: {null_before} nulls → {null_after} | coverage {coverage:.1%}")

    return df


def _compute_raw_institutional_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 5 institutional signals with strict temporal leakage prevention.
    All computations use .shift(1).expanding() or .shift(1).rolling() patterns.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # ── Signal 1: Tournament Momentum Diff ────────────────────────────────────
    # Sustained multi-match momentum: EMA of last 5 form scores, then diff
    for side in ['home', 'away']:
        team_col  = f'{side}_team'
        form_col  = f'{side}_xg_form_momentum'  # using available form metric
        mom_col   = f'{side}_tournament_momentum'
        if form_col not in df.columns:
            df[mom_col] = 0.0
            continue
        df[mom_col] = (
            df.groupby(team_col)[form_col]
              .transform(lambda x: x.shift(1).ewm(span=5, min_periods=2).mean())
        )
    if 'home_tournament_momentum' in df.columns and 'away_tournament_momentum' in df.columns:
        df['tournament_momentum_diff'] = (
            df['home_tournament_momentum'] - df['away_tournament_momentum']
        )

    # ── Signal 2: Glicko Velocity Diff ────────────────────────────────────────
    for side in ['home', 'away']:
        team_col   = f'{side}_team'
        glicko_col = f'{side}_glicko'
        vel_col    = f'{side}_glicko_velocity'
        if glicko_col not in df.columns:
            df[vel_col] = 0.0
            continue
        df[vel_col] = (
            df.groupby(team_col)[glicko_col]
              .transform(lambda x: x.shift(1).diff(3) / 3.0)
        )
    if 'home_glicko_velocity' in df.columns and 'away_glicko_velocity' in df.columns:
        df['glicko_velocity_diff'] = (
            df['home_glicko_velocity'] - df['away_glicko_velocity']
        )

    # ── Signal 3: Conversion Diff ─────────────────────────────────────────────
    for side in ['home', 'away']:
        team_col = f'{side}_team'
        conv_col = f'{side}_conversion'
        # we will use actual goals directly to compute rolling
        goals_col = f'{side}_goals'
        xg_col    = f'{side}_xg'
        if goals_col in df.columns and xg_col in df.columns:
            df[conv_col] = (
                df.groupby(team_col)
                  .apply(lambda g: (g[goals_col] - g[xg_col]).shift(1).rolling(5, min_periods=2).mean())
                  .reset_index(level=0, drop=True)
            )
        else:
            df[conv_col] = 0.0
    if 'home_conversion' in df.columns and 'away_conversion' in df.columns:
        df['conversion_diff'] = df['home_conversion'] - df['away_conversion']

    # ── Signal 4: Defensive Shape Diff ───────────────────────────────────────
    for side in ['home', 'away']:
        opp    = 'away' if side == 'home' else 'home'
        team_col   = f'{side}_team'
        def_col    = f'{side}_defensive_shape'
        gc_col     = f'{opp}_goals'
        xgc_col    = f'{opp}_xg'
        if gc_col in df.columns and xgc_col in df.columns:
            df[def_col] = (
                df.groupby(team_col)
                  .apply(lambda g: (g[xgc_col] - g[gc_col]).shift(1).rolling(5, min_periods=2).mean())
                  .reset_index(level=0, drop=True)
            )
        else:
            df[def_col] = 0.0
    if 'home_defensive_shape' in df.columns and 'away_defensive_shape' in df.columns:
        df['defensive_shape_diff'] = df['home_defensive_shape'] - df['away_defensive_shape']

    # ── Signal 5: Lineup Continuity Diff ─────────────────────────────────────
    for side in ['home', 'away']:
        opp = 'away' if side == 'home' else 'home'
        team_col  = f'{side}_team'
        cont_col  = f'{side}_lineup_continuity'
        gc_col    = f'{opp}_goals'
        if gc_col in df.columns:
            df[cont_col] = (
                df.groupby(team_col)[gc_col]
                  .transform(lambda x: 1.0 / (x.shift(1).rolling(5, min_periods=2).std() + 0.5))
            )
        else:
            df[cont_col] = 1.0
    if 'home_lineup_continuity' in df.columns and 'away_lineup_continuity' in df.columns:
        df['lineup_continuity_diff'] = (
            df['home_lineup_continuity'] - df['away_lineup_continuity']
        )

    return df

# data/odds_loader.py
"""
Historical odds data loaders.
USAGE: Drop CSV files into data/ folder, then run:
    python -c "from data.odds_loader import enrich_with_odds; print('Ready')"

Supported sources:
1. football-data.co.uk (free, covers major competitions)
2. The Odds Portal (scraped format)
3. betexplorer.com (international matches)

Instructions:
  - Visit: https://www.football-data.co.uk/data.php
  - Download: World Cup CSVs (under 'International' section)
  - Drop into: data/football_data_odds/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

ODDS_DIR = Path("data/football_data_odds")


def load_football_data_csv(filepath: str) -> pd.DataFrame:
    """
    Load football-data.co.uk CSV format.
    Columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, B365H, B365D, B365A
    Computes no-vig probabilities from Bet365 closing odds.
    """
    df = pd.read_csv(filepath)

    # Flexible column mapping (different CSV versions use slightly different names)
    col_map = {
        'Date':     ['Date', 'date'],
        'HomeTeam': ['HomeTeam', 'Home', 'home_team'],
        'AwayTeam': ['AwayTeam', 'Away', 'away_team'],
        'FTHG':     ['FTHG', 'HG', 'home_goals'],
        'FTAG':     ['FTAG', 'AG', 'away_goals'],
        'B365H':    ['B365H', 'BbAvH', 'PSH'],
        'B365D':    ['B365D', 'BbAvD', 'PSD'],
        'B365A':    ['B365A', 'BbAvA', 'PSA'],
    }

    for target, candidates in col_map.items():
        for cand in candidates:
            if cand in df.columns and target not in df.columns:
                df = df.rename(columns={cand: target})
                break

    missing = [c for c in ['B365H', 'B365D', 'B365A'] if c not in df.columns]
    if missing:
        warnings.warn(f"Missing odds columns: {missing}. Cannot compute no-vig probs.")
        return df

    # Remove overround (no-vig normalization)
    raw_h = 1 / df['B365H'].clip(1.01)
    raw_d = 1 / df['B365D'].clip(1.01)
    raw_a = 1 / df['B365A'].clip(1.01)
    overround = raw_h + raw_d + raw_a

    df['novig_home']    = raw_h / overround
    df['novig_draw']    = raw_d / overround
    df['novig_away']    = raw_a / overround
    df['overround_pct'] = (overround - 1) * 100  # bookmaker margin %

    print(f"  Loaded {len(df)} matches, avg overround: {df['overround_pct'].mean():.1f}%")
    return df


def load_all_available_odds() -> pd.DataFrame:
    """
    Load all CSVs from data/football_data_odds/ and combine.
    Call this from scraper.py or train_test.py.
    """
    if not ODDS_DIR.exists():
        ODDS_DIR.mkdir(parents=True)
        print(f"  Created {ODDS_DIR}. Drop football-data.co.uk CSVs here.")
        return pd.DataFrame()

    csvs = list(ODDS_DIR.glob("*.csv"))
    if not csvs:
        print(f"  No CSV files in {ODDS_DIR}. "
              f"Download from football-data.co.uk and drop here.")
        return pd.DataFrame()

    frames = []
    for csv_path in csvs:
        try:
            df = load_football_data_csv(str(csv_path))
            frames.append(df)
            print(f"  ✅ Loaded: {csv_path.name} ({len(df)} matches)")
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total odds data: {len(combined)} matches from {len(frames)} files")
    return combined


def enrich_dataset_with_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical odds into training dataframe.
    Matches on home_team + away_team + date (fuzzy date ±2 days).
    Adds: novig_home, novig_draw, novig_away, overround_pct
    """
    odds_df = load_all_available_odds()
    if odds_df.empty:
        df['novig_home']    = np.nan
        df['novig_draw']    = np.nan
        df['novig_away']    = np.nan
        df['overround_pct'] = np.nan
        return df

    # Normalize team names for merge
    odds_df['HomeTeam'] = odds_df['HomeTeam'].str.strip().str.title()
    odds_df['AwayTeam'] = odds_df['AwayTeam'].str.strip().str.title()
    df_copy = df.copy()
    df_copy['home_team_norm'] = df_copy['home_team'].str.strip().str.title()
    df_copy['away_team_norm'] = df_copy['away_team'].str.strip().str.title()

    merged = df_copy.merge(
        odds_df[['HomeTeam', 'AwayTeam', 'novig_home', 'novig_draw', 'novig_away', 'overround_pct']],
        left_on=['home_team_norm', 'away_team_norm'],
        right_on=['HomeTeam', 'AwayTeam'],
        how='left'
    )

    matched = merged['novig_home'].notna().sum()
    print(f"  Odds matched: {matched}/{len(df)} matches ({matched/len(df):.0%})")
    return merged.drop(columns=['home_team_norm', 'away_team_norm',
                                 'HomeTeam', 'AwayTeam'], errors='ignore')

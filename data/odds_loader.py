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
import difflib

ODDS_DIR = Path("data/football_data_odds")

MANUAL_ALIASES = {
    "South Korea": "Korea Republic",
    "USA": "United States",
    "IR Iran": "Iran",
    "North Korea": "Korea DPR",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "Republic of Ireland": "Ireland",
    "Czech Rep": "Czech Republic",
    "China PR": "China"
}

def map_team_name(name, valid_names):
    if pd.isna(name): return None
    name = str(name).strip().title()
    if name in valid_names:
        return name
    if name in MANUAL_ALIASES:
        return MANUAL_ALIASES[name]
    
    # Fuzzy match with high threshold
    matches = difflib.get_close_matches(name, valid_names, n=1, cutoff=0.85)
    if matches:
        return matches[0]
    
    print(f"    [Audit] Unmatched team name: '{name}'")
    return None

def load_football_data_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # 1. Look for Closing Odds first (AvgC, B365C, PSC) for no-vig baseline
    h_col = next((c for c in ['AvgCH', 'B365CH', 'PSCH'] if c in df.columns), None)
    d_col = next((c for c in ['AvgCD', 'B365CD', 'PSCD'] if c in df.columns), None)
    a_col = next((c for c in ['AvgCA', 'B365CA', 'PSCA'] if c in df.columns), None)

    # 2. Fallback to Pre-Closing Odds (Avg, B365, PS)
    if not (h_col and d_col and a_col):
        h_col = next((c for c in ['AvgH', 'B365H', 'PSH', 'BbAvH'] if c in df.columns), None)
        d_col = next((c for c in ['AvgD', 'B365D', 'PSD', 'BbAvD'] if c in df.columns), None)
        a_col = next((c for c in ['AvgA', 'B365A', 'PSA', 'BbAvA'] if c in df.columns), None)

    if not (h_col and d_col and a_col):
        warnings.warn(f"Missing odds columns in {filepath}. Cannot compute no-vig probs.")
        return df

    # Remove overround for the feature baseline
    raw_h = 1 / df[h_col].clip(1.01)
    raw_d = 1 / df[d_col].clip(1.01)
    raw_a = 1 / df[a_col].clip(1.01)
    overround = raw_h + raw_d + raw_a

    df['novig_home']    = raw_h / overround
    df['novig_draw']    = raw_d / overround
    df['novig_away']    = raw_a / overround
    df['overround_pct'] = (overround - 1) * 100

    # 3. Extract Best Available Market Odds for Arbitrage
    # Common bookies in football-data.co.uk: B365, PS (Pinnacle), WH (William Hill), VC (BetVictor), BW (Betway), IW (Interwetten)
    h_cols_all = [c for c in df.columns if c in ['B365H', 'PSH', 'WHH', 'VCH', 'BWH', 'IWH', 'AvgH', 'B365CH', 'PSCH', 'WHCH', 'VCCH', 'BWCH', 'IWCH', 'AvgCH']]
    d_cols_all = [c for c in df.columns if c in ['B365D', 'PSD', 'WHD', 'VCD', 'BWD', 'IWD', 'AvgD', 'B365CD', 'PSCD', 'WHCD', 'VCCD', 'BWCD', 'IWCD', 'AvgCD']]
    a_cols_all = [c for c in df.columns if c in ['B365A', 'PSA', 'WHA', 'VCA', 'BWA', 'IWA', 'AvgA', 'B365CA', 'PSCA', 'WHCA', 'VCCA', 'BWCA', 'IWCA', 'AvgCA']]

    if h_cols_all and d_cols_all and a_cols_all:
        df['max_h'] = df[h_cols_all].max(axis=1)
        df['max_d'] = df[d_cols_all].max(axis=1)
        df['max_a'] = df[a_cols_all].max(axis=1)
    else:
        # Fallback if specific bookie columns are missing
        df['max_h'] = df[h_col]
        df['max_d'] = df[d_col]
        df['max_a'] = df[a_col]

    col_map = {
        'Date':     ['Date', 'date'],
        'HomeTeam': ['HomeTeam', 'Home', 'home_team'],
        'AwayTeam': ['AwayTeam', 'Away', 'away_team'],
    }
    for target, candidates in col_map.items():
        for cand in candidates:
            if cand in df.columns and target not in df.columns:
                df = df.rename(columns={cand: target})
                break

    return df

def load_all_available_odds() -> pd.DataFrame:
    if not ODDS_DIR.exists():
        ODDS_DIR.mkdir(parents=True)
        return pd.DataFrame()

    csvs = list(ODDS_DIR.glob("*.csv"))
    frames = []
    for csv_path in csvs:
        try:
            df = load_football_data_csv(str(csv_path))
            frames.append(df)
            print(f"  ✅ Loaded odds: {csv_path.name} ({len(df)} matches)")
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path.name}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def enrich_dataset_with_odds(df: pd.DataFrame) -> pd.DataFrame:
    odds_df = load_all_available_odds()
    if odds_df.empty:
        df['novig_home']    = np.nan
        df['novig_draw']    = np.nan
        df['novig_away']    = np.nan
        df['overround_pct'] = np.nan
        df['max_h']         = np.nan
        df['max_d']         = np.nan
        df['max_a']         = np.nan
        return df

    valid_names = set(df['home_team'].str.strip().str.title().unique()) | set(df['away_team'].str.strip().str.title().unique())
    
    odds_df['HomeTeamNorm'] = odds_df['HomeTeam'].apply(lambda x: map_team_name(x, valid_names))
    odds_df['AwayTeamNorm'] = odds_df['AwayTeam'].apply(lambda x: map_team_name(x, valid_names))
    odds_df = odds_df.dropna(subset=['HomeTeamNorm', 'AwayTeamNorm'])

    df_copy = df.copy()
    df_copy['home_team_norm'] = df_copy['home_team'].str.strip().str.title()
    df_copy['away_team_norm'] = df_copy['away_team'].str.strip().str.title()

    merged = df_copy.merge(
        odds_df[['HomeTeamNorm', 'AwayTeamNorm', 'novig_home', 'novig_draw', 'novig_away', 'overround_pct', 'max_h', 'max_d', 'max_a']],
        left_on=['home_team_norm', 'away_team_norm'],
        right_on=['HomeTeamNorm', 'AwayTeamNorm'],
        how='left'
    )

    matched = merged['novig_home'].notna().sum()
    print(f"  Odds matched: {matched}/{len(df)} matches ({matched/len(df):.0%})")
    
    return merged.drop(columns=['home_team_norm', 'away_team_norm', 'HomeTeamNorm', 'AwayTeamNorm'], errors='ignore')


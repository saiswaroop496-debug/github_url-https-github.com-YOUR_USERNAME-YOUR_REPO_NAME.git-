import pandas as pd
import numpy as np
import os
import json
from statsbombpy import sb
import warnings

# Suppress statsbombpy warnings about unauthenticated requests (open data is free)
warnings.filterwarnings("ignore", category=UserWarning)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "statsbomb")
os.makedirs(CACHE_DIR, exist_ok=True)

class StatsBombGNNScraper:
    """
    Pulls high-resolution event data (x, y coordinates, passing networks) from the
    StatsBomb Open Data API for use in the V8 Graph Neural Network (GNN).
    """
    def __init__(self):
        # FIFA World Cup is competition_id=43
        self.competition_id = 43
        # 106 = 2022 World Cup, 3 = 2018 World Cup
        self.available_seasons = [106, 3]

    def _get_matches(self) -> list:
        """Fetch all available World Cup matches from StatsBomb Open Data."""
        all_matches = []
        for season in self.available_seasons:
            try:
                matches_df = sb.matches(competition_id=self.competition_id, season_id=season)
                all_matches.extend(matches_df.to_dict('records'))
            except Exception as e:
                print(f"StatsBomb API Error fetching matches for season {season}: {e}")
        return all_matches

    def extract_passing_network(self, match_id: int) -> list:
        """
        Extracts passing events from a match to construct the GNN Adjacency Matrix.
        
        Returns a list of event dictionaries formatted for the GNN:
        [
            {
                'passer_id': int,
                'receiver_id': int,
                'pass_success': bool,
                'passer_location': (x, y),
                'pressing_intensity': float
            }, ...
        ]
        """
        cache_file = os.path.join(CACHE_DIR, f"match_{match_id}_passes.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        print(f"Fetching StatsBomb Event Data for match_id: {match_id}...")
        try:
            # Fetch events for the specific match
            events = sb.events(match_id=match_id)
            
            # Filter for passes
            if 'type' not in events.columns:
                return []
                
            passes = events[events['type'] == 'Pass'].copy()
            
            gnn_events = []
            for _, row in passes.iterrows():
                # Skip passes without a known recipient or location
                if pd.isna(row.get('player_id')) or pd.isna(row.get('pass_recipient_id')):
                    continue
                    
                if 'location' not in row or not isinstance(row['location'], list) or len(row['location']) != 2:
                    continue

                # Pass success: True if 'pass_outcome' is NaN (StatsBomb records unsuccessful passes explicitly)
                pass_success = pd.isna(row.get('pass_outcome'))
                
                # Under pressure metric (1.0 if under pressure, 0.0 otherwise)
                pressure = 1.0 if row.get('under_pressure') == True else 0.0

                gnn_events.append({
                    'passer_id': int(row['player_id']),
                    'receiver_id': int(row['pass_recipient_id']),
                    'pass_success': bool(pass_success),
                    'passer_location': (float(row['location'][0]), float(row['location'][1])),
                    'pressing_intensity': pressure,
                    'team_id': int(row['team_id']),
                    'team_name': str(row['team'])
                })

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(gnn_events, f)
                
            return gnn_events

        except Exception as e:
            print(f"StatsBomb Event Extraction Error for {match_id}: {e}")
            return []

    def build_worldcup_graph_dataset(self) -> dict:
        """
        Builds the entire dataset of passing networks for all available World Cup matches.
        Used to pre-train the GNN.
        """
        matches = self._get_matches()
        dataset = {}
        for m in matches:
            match_id = m['match_id']
            home_team = m['home_team']
            away_team = m['away_team']
            
            events = self.extract_passing_network(match_id)
            if events:
                dataset[match_id] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'events': events
                }
        return dataset

if __name__ == "__main__":
    print("Testing StatsBomb Open Data Scraper (V8 GNN Feeder)...")
    scraper = StatsBombGNNScraper()
    matches = scraper._get_matches()
    print(f"Found {len(matches)} open World Cup matches.")
    
    if len(matches) > 0:
        sample_match = matches[0]['match_id']
        sample_events = scraper.extract_passing_network(sample_match)
        print(f"Extracted {len(sample_events)} passing events for match {sample_match}.")
        if len(sample_events) > 0:
            print("Sample Event:", sample_events[0])

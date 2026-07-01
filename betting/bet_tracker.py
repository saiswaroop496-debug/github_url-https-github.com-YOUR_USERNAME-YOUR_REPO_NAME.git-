import pandas as pd
import numpy as np
from datetime import datetime

class CLVTracker:
    """
    Automated Closing Line Value (CLV) Ledger.
    Tracks all placed bets and compares them to the closing line to attribute performance.
    """
    def __init__(self, ledger_path="betting_ledger.csv"):
        self.ledger_path = ledger_path
        self.bets = []

    def log_bet(self, fixture_id, market, placed_odds, stake, expected_value, model_prob):
        """Log a newly placed bet."""
        bet = {
            "timestamp": datetime.utcnow().isoformat(),
            "fixture_id": fixture_id,
            "market": market,
            "placed_odds": placed_odds,
            "stake": stake,
            "expected_value": expected_value,
            "model_prob": model_prob,
            "closing_odds": None,
            "result": None
        }
        self.bets.append(bet)
        
    def update_closing_line(self, fixture_id, market, closing_odds):
        """Update the ledger with the sharp closing odds just before kickoff."""
        for bet in self.bets:
            if bet["fixture_id"] == fixture_id and bet["market"] == market:
                bet["closing_odds"] = closing_odds
                
    def settle_bet(self, fixture_id, market, won: bool):
        """Settle a bet post-match."""
        for bet in self.bets:
            if bet["fixture_id"] == fixture_id and bet["market"] == market:
                bet["result"] = 1 if won else 0

    def generate_weekly_report(self) -> dict:
        """Calculate CLV and Return on Investment (ROI)."""
        if not self.bets:
            return {"total_bets": 0, "clv_avg": 0.0, "roi": 0.0}
            
        df = pd.DataFrame(self.bets)
        
        # Calculate CLV: (placed_odds / closing_odds) - 1
        # E.g. backed at 2.20, closes at 2.00 -> CLV = (2.20 / 2.00) - 1 = +10%
        df['clv'] = np.where(df['closing_odds'].notna(), 
                             (df['placed_odds'] / df['closing_odds']) - 1.0, 
                             0.0)
                             
        clv_avg = df['clv'].mean()
        
        # Calculate ROI if settled
        settled = df[df['result'].notna()]
        if not settled.empty:
            profit = (settled['result'] * settled['placed_odds'] * settled['stake']) - settled['stake']
            roi = profit.sum() / settled['stake'].sum()
        else:
            roi = 0.0
            
        return {
            "total_bets": len(self.bets),
            "clv_avg": round(clv_avg, 4),
            "roi": round(roi, 4)
        }

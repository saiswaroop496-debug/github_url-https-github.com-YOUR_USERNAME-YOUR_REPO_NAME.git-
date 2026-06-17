class BetTracker:
    def __init__(self, initial_bankroll=1000.0):
        self.bankroll = initial_bankroll
        self.history = []
        
    def record_bet(self, match_id, outcome, stake, odds, result_won):
        profit = (stake * odds - stake) if result_won else -stake
        self.bankroll += profit
        
        self.history.append({
            'match_id': match_id,
            'outcome': outcome,
            'stake': stake,
            'odds': odds,
            'result_won': result_won,
            'profit': profit,
            'bankroll_after': self.bankroll
        })
        
    def get_metrics(self):
        if not self.history:
            return {'ROI': 0.0, 'WinRate': 0.0, 'TotalBets': 0}
            
        wins = sum(1 for b in self.history if b['result_won'])
        total_staked = sum(b['stake'] for b in self.history)
        total_profit = sum(b['profit'] for b in self.history)
        
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0
        win_rate = (wins / len(self.history)) * 100
        
        return {'ROI': roi, 'WinRate': win_rate, 'TotalBets': len(self.history)}

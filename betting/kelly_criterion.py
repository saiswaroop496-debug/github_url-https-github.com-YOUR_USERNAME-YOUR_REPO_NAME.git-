class KellyPortfolio:
    def __init__(self, max_stake_fraction=0.20, max_stake_units=50.0):
        self.max_stake_fraction = max_stake_fraction
        self.max_stake_units = max_stake_units
        self.open_bets = set() # Track teams with open bets
        
    def size_bet(self, model_prob, decimal_odds, bankroll, home_team, away_team):
        b = decimal_odds - 1.0
        q = 1.0 - model_prob
        
        # Guard against zero/negative odds
        if b <= 0: return 0.0
        
        f_star = (b * model_prob - q) / b
        f_quarter = f_star / 4.0
        
        # Hard limits
        stake = min(f_quarter * bankroll, self.max_stake_fraction * bankroll, self.max_stake_units)
        stake = max(stake, 0.0)
        
        # Correlation Guard
        if home_team in self.open_bets or away_team in self.open_bets:
            stake *= 0.5
            
        return stake
        
    def add_open_bet(self, home_team, away_team):
        self.open_bets.add(home_team)
        self.open_bets.add(away_team)
        
    def resolve_bets(self):
        self.open_bets.clear()

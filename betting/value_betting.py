class ValueBettingEngine:
    def __init__(self, min_edge=0.025):
        self.min_edge = min_edge
        
    def evaluate(self, model_probs, market_odds, hawkes_veto=False):
        """
        model_probs: dict {'Home': p, 'Draw': p, 'Away': p}
        market_odds: dict {'Home': odds, 'Draw': odds, 'Away': odds}
        """
        raw_h = 1.0 / market_odds['Home']
        raw_d = 1.0 / market_odds['Draw']
        raw_a = 1.0 / market_odds['Away']
        
        overround = raw_h + raw_d + raw_a
        
        novig_h = raw_h / overround
        novig_d = raw_d / overround
        novig_a = raw_a / overround
        
        edge_h = model_probs['Home'] - novig_h
        edge_d = model_probs['Draw'] - novig_d
        edge_a = model_probs['Away'] - novig_a
        
        edges = {'Home': edge_h, 'Draw': edge_d, 'Away': edge_a}
        novig_probs = {'Home': novig_h, 'Draw': novig_d, 'Away': novig_a}
        
        best_outcome = max(edges, key=edges.get)
        best_edge = edges[best_outcome]
        
        if hawkes_veto or best_edge < self.min_edge:
            return "NO BET", best_outcome, best_edge, novig_probs[best_outcome], model_probs[best_outcome]
            
        return "BET", best_outcome, best_edge, novig_probs[best_outcome], model_probs[best_outcome]

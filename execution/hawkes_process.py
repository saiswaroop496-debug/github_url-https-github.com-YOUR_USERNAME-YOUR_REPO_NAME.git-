import numpy as np

class HawkesOrderFlowModel:
    def __init__(self, mu=0.5, alpha=0.8, beta=1.5):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.veto_threshold = -0.20
        
    def _intensity(self, current_time, event_times):
        """
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta*(t-t_i))
        """
        decay_sum = 0.0
        for t_i in event_times:
            if t_i < current_time:
                decay_sum += self.alpha * np.exp(-self.beta * (current_time - t_i))
        return self.mu + decay_sum

    def compute_imbalance(self, current_time, buy_times, sell_times):
        lambda_buy = self._intensity(current_time, buy_times)
        lambda_sell = self._intensity(current_time, sell_times)
        
        imbalance = (lambda_buy - lambda_sell) / (lambda_buy + lambda_sell + 1e-9)
        
        if imbalance < self.veto_threshold:
            return imbalance, "VETO — TOXIC FLOW"
        return imbalance, "PASS"

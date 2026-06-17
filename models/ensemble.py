import numpy as np

class EnsembleBlender:
    def __init__(self, weight_dc=0.5, weight_meta=0.5):
        self.weight_dc = weight_dc
        self.weight_meta = weight_meta
        
    def blend(self, dc_probs, meta_probs):
        """
        Blends probabilities from Dixon-Coles and MetaLearner.
        Inputs are dicts or arrays. Returns normalized dict or array.
        """
        if isinstance(dc_probs, dict):
            blended = {
                'Home': self.weight_dc * dc_probs['Home'] + self.weight_meta * meta_probs['Home'],
                'Draw': self.weight_dc * dc_probs['Draw'] + self.weight_meta * meta_probs['Draw'],
                'Away': self.weight_dc * dc_probs['Away'] + self.weight_meta * meta_probs['Away']
            }
            total = sum(blended.values())
            return {k: v / total for k, v in blended.items()}
        else:
            blended = self.weight_dc * dc_probs + self.weight_meta * meta_probs
            return blended / blended.sum(axis=1, keepdims=True)

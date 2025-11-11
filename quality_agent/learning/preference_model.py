import numpy as np
from scipy.optimize import minimize


class PreferenceModel:
    def __init__(self, num_features, critical_feature_index=0, critical_threshold=0.99):
        self.weights = np.zeros(num_features)
        self.critical_feature_index = critical_feature_index
        self.critical_threshold = critical_threshold

    def _passes_critical_constraint(self, item_features):
        if self.critical_feature_index is None:
            return True
        if self.critical_feature_index >= len(item_features):
            return True
        return item_features[self.critical_feature_index] >= self.critical_threshold

    def extract_feature_vector(self, item1_features, item2_features):
        return np.array(item1_features) - np.array(item2_features)

    def bradley_terry_probability(self, item1, item2):
        item1 = np.asarray(item1, dtype=float)
        item2 = np.asarray(item2, dtype=float)
        item1_ok = self._passes_critical_constraint(item1)
        item2_ok = self._passes_critical_constraint(item2)

        if item1_ok and not item2_ok:
            return 1.0
        if item2_ok and not item1_ok:
            return 0.0

        exp_score_item1 = np.exp(np.dot(self.weights, item1))
        exp_score_item2 = np.exp(np.dot(self.weights, item2))
        return exp_score_item1 / (exp_score_item1 + exp_score_item2)

    def calculate_map_score(self, comparisons):
        score = 0
        for item1, item2, winner in comparisons:
            probability = self.bradley_terry_probability(item1, item2)
            score += winner * np.log(probability) + (1 - winner) * np.log(1 - probability)
        return score

    def loss_function(self, weights, comparisons):
        self.weights = weights
        return -self.calculate_map_score(comparisons)

    def learn_weights(self, comparisons):
        initial_weights = np.zeros(len(self.weights))
        result = minimize(self.loss_function, initial_weights, args=(comparisons,), method='BFGS')
        self.weights = result.x

        if self.critical_feature_index is not None:
            self.weights[self.critical_feature_index] = max(self.weights[self.critical_feature_index], 0)

    def predict_preference(self, item1, item2):
        return self.bradley_terry_probability(item1, item2)

import numpy as np
from scipy.optimize import minimize

class PreferenceModel:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)

    def extract_feature_vector(self, item1_features, item2_features):
        return np.array(item1_features) - np.array(item2_features)

    def bradley_terry_probability(self, item1, item2):
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

    def predict_preference(self, item1, item2):
        return self.bradley_terry_probability(item1, item2)

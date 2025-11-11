import json
import os

class PreferenceStorage:
    def __init__(self, storage_path='data/preferences.json'):
        self.storage_path = storage_path
        self.preferences = {}
        self.load_preferences()

    def load_preferences(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self.preferences = json.load(f)
        else:
            self.preferences = {}

    def save_preferences(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.preferences, f, indent=4)

    def save_comparison(self, user_id, comparison_data):
        if user_id not in self.preferences:
            self.preferences[user_id] = {}
        self.preferences[user_id]['comparisons'] = self.preferences[user_id].get('comparisons', [])
        self.preferences[user_id]['comparisons'].append(comparison_data)
        self.save_preferences()

    def get_user_comparisons(self, user_id):
        return self.preferences.get(user_id, {}).get('comparisons', [])

    def update_user_weights(self, user_id, weights):
        if user_id not in self.preferences:
            self.preferences[user_id] = {}
        self.preferences[user_id]['weights'] = weights
        self.save_preferences()

    def get_user_weights(self, user_id):
        return self.preferences.get(user_id, {}).get('weights', {})

    def get_user_stats(self, user_id):
        user_data = self.preferences.get(user_id, {})
        return {
            'comparisons': len(user_data.get('comparisons', [])),
            'weights': user_data.get('weights', {})
        }
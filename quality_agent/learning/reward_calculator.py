class RewardCalculator:
    def __init__(self):
        self.previous_feedback = []
        self.current_feedback = []

    def calculate_comparison_reward(self, new_feedback):
        """
        Calculate reward based on comparison of new feedback with previous feedback.
        """
        if self.previous_feedback:
            reward = sum(1 for fb in new_feedback if fb > max(self.previous_feedback))
            self.previous_feedback = new_feedback
            return reward
        self.previous_feedback = new_feedback
        return 0

    def _calculate_consistency_bonus(self):
        """
        Calculate bonus for consistency based on previous feedback.
        """
        if len(self.previous_feedback) < 2:
            return 0
        return 1 if self.previous_feedback[-1] == self.previous_feedback[-2] else 0

    def calculate_improvement_reward(self, new_feedback):
        """
        Calculate reward based on improvement in feedback over time.
        """
        if self.current_feedback:
            improvement = sum(1 for fb in new_feedback if fb > self.current_feedback[-1])
            self.current_feedback = new_feedback
            return improvement
        self.current_feedback = new_feedback
        return 0

    def get_average_reward(self):
        """
        Get the average reward calculated from previous feedback.
        """
        all_feedback = self.previous_feedback + self.current_feedback
        return sum(all_feedback) / len(all_feedback) if all_feedback else 0

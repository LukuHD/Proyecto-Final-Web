class QualityEvaluator:
    def __init__(self):
        self.metrics = []

    def add_metric(self, metric):
        self.metrics.append(metric)

    def evaluate(self):
        if not self.metrics:
            return 0
        total_score = sum(self.metrics)
        return total_score / len(self.metrics)

    def final_quality_score(self):
        score = self.evaluate()
        return f"Final Quality Score: {score:.2f}"

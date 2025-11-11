class FeedbackCollector:
    def __init__(self):
        self.connectivity = None
        self.density = None
        self.room_sizes = None
        self.room_count = None
        self.corridors = None
        self.layout = None
        self.complexity = None

    def create_feedback_template(self):
        # Create a structured feedback template
        template = {
            'connectivity': '',
            'density': '',
            'room_sizes': '',
            'room_count': '',
            'corridors': '',
            'layout': '',
            'complexity': ''
        }
        return template

    def process_feedback(self, feedback):
        # Process the feedback received
        for key, value in feedback.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def suggest_parameter_changes(self):
        # Analyze the feedback and suggest parameter changes
        suggestions = []
        if self.connectivity is not None:
            suggestions.append(f'Consider improving connectivity: {self.connectivity}')
        if self.density is not None:
            suggestions.append(f'Optimize density levels: {self.density}')
        # Additional logic can be added here
        return suggestions

    def generate_feedback_questions(self):
        # Generate a set of questions to ask users for feedback
        questions = [
            'How would you rate the connectivity of the map?',
            'What are your thoughts on the density of rooms?',
            'Are the room sizes adequate for your needs?',
            'How many rooms would you prefer on the map?',
            'What do you think about the corridors layout?',
            'How complex did you find the overall layout?'
        ]
        return questions

# Main Gemma class
from config import Config

class Gemma:
    def __init__(self, config):
        self.config = Config

    def generate(self, prompt):
        # get prompt and pass it through the model layers
        result={};

        return result
    def train(self, data):
        # Training logic here
        pass

    def evaluate(self, data):
        # Evaluation logic here
        pass
    
    def save_model(self, path):
        # Logic to save the model to the specified path
        pass

    def get_config(self):
        return self.config

    

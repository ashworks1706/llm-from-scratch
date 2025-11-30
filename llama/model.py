# Main Llama class
from ..utils.config import Config

class Llama: 
    def __init__(self, config, logger):
        self.config = Config(**config)
        self.logger = logger

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

    

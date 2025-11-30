# Main Llama class
from ..utils.config import Config

class Llama: 
    def __init__(self, config, logger):
        self.config = Config(**config)
        self.logger = logger

   

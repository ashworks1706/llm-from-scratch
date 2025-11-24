import torch
from typing import Dict, List, Optional, Union, Tuple, Iterable
from torch import nn
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCasualLM(config.text_config)
        self.langauge_model = language_model
        self.pad_token_id = config.pad_token_id if self.config.pad_token_id is not None else -1
        
    def tie_weights(self):
        return self.langauge_model.tie_weights()
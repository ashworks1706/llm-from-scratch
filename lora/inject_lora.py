import torch
import torch.nn as nn
from lora_layer import LoRALayer
# After training, the small, trained LoRA matrices can be mathematically merged with the original 
# base model weights, resulting in a fine-tuned model that has no additional inference latency compared 
# to the original model
# this script injects LoRA adapters into an existing model
# we replace attention projection layers (wq, wk, wv, wo) with LoRA-wrapped versions

def inject_lora_to_model(model, rank=8, alpha=16, target_modules=['wq', 'wk', 'wv', 'wo']):
    # TODO: loop through all model layers
    # TODO: for each decoder block, replace target modules with LoRA versions
    # TODO: count how many parameters are trainable vs frozen
    pass


def count_parameters(model):
    # TODO: count trainable parameters (LoRA adapters)
    # TODO: count frozen parameters (base model)
    # TODO: print the ratio
    pass


if __name__ == "__main__":
    # example usage
    pass

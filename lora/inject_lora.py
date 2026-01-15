import torch
import torch.nn as nn
from lora_layer import LoRALayer

# this script injects LoRA adapters into an existing model
# we replace attention projection layers (wq, wk, wv, wo) with LoRA-wrapped versions
# why? so we can fine-tune the model with only ~0.1% of parameters instead of 100%


def inject_lora_to_model(model, rank=8, alpha=16, target_modules=['wq', 'wk', 'wv', 'wo']):
    # the process:
    # 1. loop through all decoder blocks in the model
    # 2. for each target module (wq, wk, wv, wo), wrap it with LoRALayer
    # 3. original weights stay frozen, only LoRA adapters (A, B) are trainable
    
    lora_count = 0
    
    # loop through all decoder blocks
    # model.layers is nn.ModuleList containing all DecoderBlock instances
    for layer_idx, layer in enumerate(model.layers):
        # each DecoderBlock has an attention module
        # attention module has wq, wk, wv, wo as nn.Linear layers
        attention = layer.attention
        
        # wrap each target module with LoRA
        for module_name in target_modules:
            # check if this module exists in the attention layer
            if hasattr(attention, module_name):
                # get the original layer (nn.Linear)
                original_layer = getattr(attention, module_name)
                
                # wrap it with LoRA
                # this creates A and B matrices, freezes original weights
                lora_layer = LoRALayer(original_layer, rank=rank, alpha=alpha)
                
                # replace the original layer with LoRA-wrapped version
                # after this, when we call attention.wq(x), it goes through LoRA
                setattr(attention, module_name, lora_layer)
                # setatrr is python's way to dynamically set an attribute on an object
                
                lora_count += 1
                print(f"Layer {layer_idx}: Injected LoRA into {module_name}")
    
    print(f"\nTotal LoRA modules injected: {lora_count}")
    
    # print parameter counts to show efficiency
    trainable, frozen = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} ({trainable/(trainable+frozen)*100:.2f}%)")
    print(f"Frozen parameters: {frozen:,} ({frozen/(trainable+frozen)*100:.2f}%)")
    print(f"Total parameters: {trainable+frozen:,}")
    print(f"Memory savings: {frozen/(trainable+frozen)*100:.1f}% of model stays frozen!")
    
    return model


def count_parameters(model):
    # trainable: LoRA adapters (A, B matrices) - these get gradients and updates
    # frozen: original model weights - these stay unchanged during training
    
    trainable = 0
    frozen = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()  # number of elements in this parameter
        
        if param.requires_grad:
            trainable += param_count
        else:
            frozen += param_count
    
    return trainable, frozen


# the entirem indset is that get the model, inject the LoRA adapter, put them in place, then finetune (SFT) 
# the model, in that process, those tiny weights get updated, then boom, you have adaptes updated
# these are the LoRA adapters, now we can continue to use it in same model or just pick those adapters and use it anywhere else
# so we can have multiple adapters for one base model, like one answers math, one biology,etc
# Train once, use anywhere!
def save_lora_adapters(model, path):
    # save ONLY the LoRA adapters, not the full model
    
    # why? because base model is frozen and huge (multiple GB)
    # LoRA adapters are tiny (few MB) and that's all we trained
    # we can share just the adapters and anyone with the base model can use them
    lora_state_dict = {}
    
    # extract only LoRA parameters (those with 'lora_A' or 'lora_B' in name)
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param
    
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA adapters to {path}")
    print(f"Adapter size: {sum(p.numel() for p in lora_state_dict.values()):,} parameters")


def load_lora_adapters(model, path):
    # load LoRA adapters into a model that already has LoRA injected
    
    # lora adapers are the A and B matrices 
    # these adapters ARE the finetuning 

    # workflow:
    # 1. start with base model
    # 2. inject LoRA (creates A, B with random/zero init)
    # 3. load trained adapter weights
    # 4. now model has learned behavior without modifying base weights
    lora_state_dict = torch.load(path)
    
    # load only the LoRA parameters
    model_dict = model.state_dict()
    model_dict.update(lora_state_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded LoRA adapters from {path}")


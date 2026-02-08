# loading and using pretrained models
# the transformers library provides standardized access to thousands of models

# topics to cover:
# - AutoModel, AutoTokenizer, AutoConfig pattern
# - loading models from hub (model_name string)
# - understanding model config and architecture params
# - .from_pretrained() method and what happens behind scenes
# - model.config attributes (hidden_size, num_layers, etc)
# - moving models to device (cuda/cpu)
# - model.eval() vs model.train() modes
# - understanding model outputs (logits, hidden_states, attentions)

# OBJECTIVE: learn to load any model and understand its structure without training
# you should be able to load llama, gpt2, bert and inspect their configs

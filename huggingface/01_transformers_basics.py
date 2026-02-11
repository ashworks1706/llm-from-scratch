import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

model_name = "distilbert-base-uncased"


config = AutoConfig.from_pretrained(model_name)


print(config)


model = AutoModel.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


model.eval()


text = "hello world"
inputs = tokenizer(text, return_tensors='pt').to(device)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape) # embedding size that is 1 batch, 4  tokens, 768 dimensions each 


# tokenizer : text -> token ids 
# Model : token ids -> hdiden representations 
# classification head : hdiden state -> final prdictions (logits )

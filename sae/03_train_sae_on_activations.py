# freeze the gpt2 model and pass input text through it to get activations, 
# then train sae to reconstruct those activations on the fly 
# we train on activation vectors: 
# choose layers/component, for each token position, extract x -> R^d, collect many xs across many prompts 
# so training set is D ={x_n}_{n=1..N} where x_n is the activation vector for a particular token in a particular prompt 









# - Implement reconstruction loss choices (MSE, normalized MSE) for activation reconstruction.
# - Implement L1 sparsity penalty on latent activations and tune lambda tradeoffs.
# - Understand top-k / hard sparsity alternatives and when they are more stable.
# - Track dead-feature and always-on-feature failure modes.
# - Build metrics for sparsity level, feature utilization, and reconstruction quality.



# - Cache activations from a chosen transformer layer and stream them as a dataset.
# - Build a training loop with optimizer, scheduler, logging, and checkpointing.
# - Handle activation normalization / centering before feeding into SAE.
# - Balance reconstruction and sparsity losses during optimization.
# - Add basic stability guards: gradient clipping, NaN checks, and periodic eval.


import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForCausalLM 
import SAE from 01_sae_fundamentals

class SAE_Model:
    def __init__(self, input_dim, latent_dim, output_dim, train_loader, test_loader):
        self.loss_fn = nn.MSELoss()
        self.model = SAE(input_dim, latent_dim, output_dim)
        self.gamma = nn.Parameter(torch.tensor(0.5)) # this is the lambda tradeoff parameter for 
        # sparsity vs recon loss, we make it learnable so that model can learn how much 
        # to focus on sparsity vs recon 
        self.gpt2tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 0.001)

    def train(self, epochs=100):
         
        self.model.train()
        # the sparsity loss is just the L1 norm of the latent code z 
        avgloss=0
        for epoch in range(epochs):
            for batch in self.train_loader:
                # get input text and pass through gpt2 to get activations 
                input_text = batch["text"]
                inputs = self.gpt2tokenizer(input_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.gpt2model(**inputs, output_hidden_states=True)
                    activations = outputs.hidden_states[-1] # get last layer activations 

                u,z,x_hat = self.model(activations)

                # compute losses 
                # recon_loss = self.loss_fn(x_hat, activations)
                # p=1 is L1 norm, p=2 is L2 norm 
                # sparsity_loss = torch.norm(self.model.encoder.ffn1.weight, p=1) # L1 norm of encoder weights as sparsity penalty 
                loss = recon_loss + self.gamma * sparsity_loss

                # backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avgloss += loss.item()





# so now we have a dictionary of features, let's analyze it and see what we have 
# 
# theres feature firing rates, where for each latent i, p_i = P(z_i > eps) that we use to classify 
# dead: very low p_i, rare specialist: low p_i but non zero 
#
# theres activation magnitude stats: per feature where E[z_i | z_i > eps] which quantiles usually 50/90/99% 
# this separates weak frequent features fro mstrong parse specialists 
#
# then theres pre activation diagnostics where from u: mean, std, percentiles per feature, if u_i is strongly negative across 
# data, ReLU gating kills that feature. 
#
# for decoder side, we check (f_i) norm of each feature vector, cosine similarity between feature vectors, which tells us duplicate features 
# if present, or detect norm collapse or exploding imbalance 
#
# we can also analyze recon quality slices by token position (early.mid.late) maybe prompt length buckets 




# we need dead ration, median firing rate, mean recon loss, top k for most dead, most active, largest deocder norm 
# then we can form histograms for firing rates, decoder norms, u and z distributions 




import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from 01_sae_fundamentals import SAE

def analyze_sae(sae_model, gpt2_model, tokenizer, dataloader, device, eps, max_batches):
    sae_model.eval()
    gpt2_model.eval()
    loss_fn = nn.MSELoss()
    eps = 1e-10
    for data in dataloader:
        batch_idx, seqlen, dim  = data
        data = tokenizer(data)
        with torch.no_grad():
            logits = model(**inputs, output_hidden_states=True)
            logits = outputs.hidden_states[:,:,-1]
            logits = outputs.view(batch_idx*seqlen, dim )
            u, z, x_hat=  sae_model.encoder(logits)
            # remember u here is original activation of input, z is the output of encoder, x_hat is reconned activation of input 
            active_count += (z > eps ).sum(0) # higher the more active neurons 
            z_sum+=z.sum(0) 
            z_active_sum += (z*(z>eps)).sum(0) 
            u_sum +=u.sum(0)
            u_sq_sum += (u**2).sum(0)
            recon_se_sum += loss_fn(x_hat,x).sum()
            recon_elem_count += x.numel()

    firing_rate = active_count/ total_tokens 
    mean_z = z_sum / total_tokens 
    mean_z_when_active = z_active_sum / (active_count + eps)
    u_mean = u_sum /total_tokens
    u_std = math.sqrt(u_sq_sum / total_tokens - u_mean^2 + eps)
    recon_mse = recon_se_sum / recon_elem_count




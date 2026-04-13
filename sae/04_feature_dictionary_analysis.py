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

    print("Firing rate per feature: ", firing_rate)
    print("Mean activation per feature: ", mean_z)
    print("Mean activation when active per feature: ", mean_z_when_active)
    print("Pre-activation mean per feature: ", u_mean)
    print("Pre-activation std per feature: ", u_std)
    print("Reconstruction MSE: ", recon_mse)
    
    feature_norms, cosine_sim = analyze_decoder_dictionary(sae_model)


def analyze_decoder_dictionary(sae_model):
    decoder_weights = sae_model.decoder.weight.data 
    feature_norms = torch.norm(decoder_weights, dim=1) 
    cosine_sim = F.cosine_similarity(decoder_weights.unsqueeze(1), decoder_weights.unsqueeze(0), dim=-1)
    print("Decoder feature norms: ", feature_norms)
    print("Decoder feature cosine similarity: ", cosine_sim)
    return feature_norms, cosine_sim 


def report(firing_rate, mean_z, mean_z_when_active, u_mean, u_std, recon_mse, feature_norms, cosine_sim):
    # dead ration, saturated ratio, top 10 dead features, most active features, largest decoder norms
    dead_ratio = (firing_rate < 0.01).float().mean()
    saturated_ratio = (firing_rate > 0.5).float().mean()
    top_k = e-10
    most_dead = torch.topk(firing_rate, k=top_k, largest=False)
    most_active = torch.topk(firing_rate, k=top_k, largest=True)
    largest_decoder_norms = torch.topk(feature_norms, k=top_k, largest=True)
    print(f"Dead ratio: {dead_ratio}, Saturated ratio: {saturated_ratio}")
    print(f"Top {top_k} dead features: {most_dead.indices}, firing rates: {most_dead.values}")
    print(f"Top {top_k} active features: {most_active.indices}, firing rates: {most_active.values}")
    print(f"Top {top_k} largest decoder norms: {largest_decoder_norms.indices}, norms: {largest_decoder_norms.values}")




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
import matplotlib.pyplot as plt

def analyze_sae(sae_model, gpt2_model, tokenizer, dataloader, device, eps, max_batches):
    sae_model.eval()
    gpt2_model.eval()
    latent_dim = sae_model.encoder.ffn1.out_features

    active_count = torch.zeros(latent_dim, device=device)
    z_sum = torch.zeros(latent_dim, device=device)
    z_active_sum = torch.zeros(latent_dim, device=device)
    u_sum = torch.zeros(latent_dim, device=device)
    u_sq_sum = torch.zeros(latent_dim, device=device)
    recon_se_sum = torch.tensor(0.0, device=device)
    recon_elem_count = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_text = batch["text"]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = gpt2_model(**inputs, output_hidden_states=True)
            activations = outputs.hidden_states[-1]
            B, S, D = activations.shape
            activations = activations.reshape(B * S, D)

            u, z, x_hat = sae_model(activations)
            # remember u here is original activation of input, z is the output of encoder, x_hat is reconned activation of input 
            active_count += (z > eps ).sum(0) # higher the more active neurons 
            z_sum+=z.sum(0) 
            z_active_sum += (z*(z>eps)).sum(0) 
            u_sum +=u.sum(0)
            u_sq_sum += (u**2).sum(0)
            recon_se_sum += ((x_hat - activations) ** 2).sum()
            recon_elem_count += activations.numel()
            total_tokens += activations.shape[0]

    firing_rate = active_count/ total_tokens 
    mean_z = z_sum / total_tokens 
    mean_z_when_active = z_active_sum / (active_count + eps)
    u_mean = u_sum /total_tokens
    u_std = torch.sqrt(u_sq_sum / total_tokens - u_mean**2 + eps)
    recon_mse = recon_se_sum / recon_elem_count

    print("Firing rate per feature: ", firing_rate)
    print("Mean activation per feature: ", mean_z)
    print("Mean activation when active per feature: ", mean_z_when_active)
    print("Pre-activation mean per feature: ", u_mean)
    print("Pre-activation std per feature: ", u_std)
    print("Reconstruction MSE: ", recon_mse)
    
    feature_norms, cosine_sim = analyze_decoder_dictionary(sae_model)

    return {
        "firing_rate": firing_rate,
        "mean_z": mean_z,
        "mean_z_when_active": mean_z_when_active,
        "u_mean": u_mean,
        "u_std": u_std,
        "recon_mse": recon_mse,
        "feature_norms": feature_norms,
        "cosine_sim": cosine_sim,
    }


def analyze_decoder_dictionary(sae_model):
    decoder_weights = sae_model.decoder.ffn1.weight.data
    # decoder weights shape: (input_dim, latent_dim), columns are feature vectors
    feature_norms = torch.norm(decoder_weights, dim=0)

    normalized = F.normalize(decoder_weights, dim=0)
    cosine_sim = normalized.T @ normalized

    print("Decoder feature norms: ", feature_norms)
    print("Decoder feature cosine similarity: ", cosine_sim)
    return feature_norms, cosine_sim 


def report(firing_rate, mean_z, mean_z_when_active, u_mean, u_std, recon_mse, feature_norms, cosine_sim):
    # dead ration, saturated ratio, top 10 dead features, most active features, largest decoder norms
    dead_ratio = (firing_rate < 1e-3).float().mean()
    saturated_ratio = (firing_rate > 0.5).float().mean()
    top_k = min(10, firing_rate.numel())
    most_dead = torch.topk(firing_rate, k=top_k, largest=False)
    most_active = torch.topk(firing_rate, k=top_k, largest=True)
    largest_decoder_norms = torch.topk(feature_norms, k=top_k, largest=True)
    print(f"Dead ratio: {dead_ratio}, Saturated ratio: {saturated_ratio}")
    print(f"Reconstruction MSE: {recon_mse}")
    print(f"Top {top_k} dead features: {most_dead.indices}, firing rates: {most_dead.values}")
    print(f"Top {top_k} active features: {most_active.indices}, firing rates: {most_active.values}")
    print(f"Top {top_k} largest decoder norms: {largest_decoder_norms.indices}, norms: {largest_decoder_norms.values}")


    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.hist(firing_rate.cpu().numpy(), bins=50)
    plt.title("Firing Rate Distribution")
    plt.subplot(2, 3, 2)
    plt.hist(mean_z.cpu().numpy(), bins=50)
    plt.title("Mean Activation Distribution")
    plt.subplot(2, 3, 3)
    plt.hist(mean_z_when_active.cpu().numpy(), bins=50)
    plt.title("Mean Activation When Active Distribution")
    plt.subplot(2, 3, 4)
    plt.hist(u_mean.cpu().numpy(), bins=50)
    plt.title("Pre-activation Mean Distribution")
    plt.subplot(2, 3, 5)
    plt.hist(u_std.cpu().numpy(), bins=50)
    plt.title("Pre-activation Std Distribution")
    plt.subplot(2, 3, 6)
    plt.hist(feature_norms.cpu().numpy(), bins=50)
    plt.title("Decoder Feature Norm Distribution")
    plt.tight_layout()
    plt.show()



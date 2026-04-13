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





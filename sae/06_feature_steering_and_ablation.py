# now that we understand what SAEs are, what those features are looking and understanding how they are represented 
# If "we directly manipulate latent feature i, does model behavior change in the predicted direction?"



# casual framing :
# we know that z_i fires on pattern P 
# so what if we force z_i down (ablation) or up (steering) and check if behavior X tied to P changes 
# if yes then we can derive that z_i -> X concept is casually relevent


# intervention math : 
# we intervene that latent z after u,z,x_hat = SAE(x) by either 
# ablation : remove the feature by setting it to 0 for "what happens if this feature is unavailable?"
# steering : change selected latent or latents, by adding or multplicate them, "ammplify or supress a feature" " can we push model behavior by increasing/decreasing this concept?"



## What counts as evidence?
#- logit shift for target tokens
#  - KL divergence between baseline and intervened next-token distributions
#  - token rank changes
#  - generation-level qualitative shifts
#
#  Strong evidence looks like:
#
#  - predictable directional change
#  - consistent across prompts, not one-off.

# what does NOT count as evidence?
# "feature z fires on pattern p so it causes behavior X " <- this is not valid because feature may jsut be reading out somethign antoher feature computed, so this is classic observer vs driverproblem, also transformers are highly stochastic and senstive to context
# "when i multiply z by 1, behavior changes dramatically" <- we are likely pushign the model off manifold, so the issue is here model was never traiend on such acvtivatiosn  causing this rather than the claim 
# <- we are likely pushign the model off manifold, so the issue is here model was never traiend on such acvtivatios causing this rather than the claim 
#


# we cannot jsut do normal testing, we need lots of testst to prove our hypothesis about the model's behavior 
# Polysemnatic features : if a feature encodes multiple concepts, steering causes mixed effects 
# <- this is also inspired from superposition behavior of transforemrs where there MLP can rerpesent many concepts in same dimension if viewed differently or itnerpreted differently 
# Off-manifold risk : large "a" can push activations out of distribution and creat eweird outputs unrelated to true casual role 
# so starting small and sweeping a is much better 




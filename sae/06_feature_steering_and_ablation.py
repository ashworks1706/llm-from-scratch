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




# Latent MoE (Nemotron 3, Kimi K3)
#
# LEARNING OBJECTIVES:
# - Project tokens down into a latent dim before routing
# - Run many experts in the latent space and project back up
# - Compare parameters, flops and loss against the deepseekv3 moe

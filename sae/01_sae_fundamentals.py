# sparse autoencoder fundamentals
#
# LEARNING OBJECTIVES:
# - Define the SAE architecture: linear encoder, sparse latent, linear decoder.
# - Understand overcomplete dictionaries (latent dim > input dim) and why they help interpretability.
# - Separate reconstruction objective from sparsity objective conceptually.
# - Identify where SAEs are typically trained in transformers (residual stream, mlp output, attention output).
# - Clarify what each tensor shape should look like at every stage.


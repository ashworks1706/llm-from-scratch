sparse autoencoders are a way to open up the hidden states of transformers and understand what internal features the model is using.

in a normal autoencoder we compress and reconstruct, but in sae we force sparsity so only a small set of latent features activate for each token. this makes features more interpretable and more reusable for mechanistic understanding.

the core flow in this folder is:

1. build basic sae blocks and understand reconstruction + sparsity tradeoff
2. implement sparse penalties and compare l1 vs top-k style sparsity
3. train sae on transformer activations (usually residual stream)
4. analyze learned feature dictionary and dead/alive features
5. interpret features with example activations and token triggers
6. try steering or ablation experiments to test if features are causal

`01_sae_fundamentals.py` starts with encoder/decoder intuition and baseline objectives.

`02_sparse_objectives.py` focuses on enforcing sparsity and preventing feature collapse.

`03_train_sae_on_activations.py` lays out the training loop over cached model activations.

`04_feature_dictionary_analysis.py` covers diagnostics like feature frequency, dead latents, and reconstruction quality.

`05_feature_interpretability.py` covers how to map latent features back to human-readable behavior.

`06_feature_steering_and_ablation.py` focuses on intervention experiments to validate that features matter causally.


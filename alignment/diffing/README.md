diffing compares a base model with its finetuned version to find what finetuning changed.

the simple start is per token kl between the two models. crosscoders learn one shared dictionary over both models and show which features are new, but l1 crosscoders make fake chat only features, so we use batchtopk with latent scaling.

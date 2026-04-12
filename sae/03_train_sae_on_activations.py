# LEARNING OBJECTIVES:
# - Cache activations from a chosen transformer layer and stream them as a dataset.
# - Build a training loop with optimizer, scheduler, logging, and checkpointing.
# - Handle activation normalization / centering before feeding into SAE.
# - Balance reconstruction and sparsity losses during optimization.
# - Add basic stability guards: gradient clipping, NaN checks, and periodic eval.


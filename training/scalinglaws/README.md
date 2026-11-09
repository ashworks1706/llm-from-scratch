scaling laws predict how loss changes with model size, data and compute so you can pick the best model for a budget without training the big one first.

we train many tiny models, fit isoflop curves and extrapolate. mup and completep change how weights are initialized and how learning rates scale so the best hyperparameters found on a small model still work on a wider or deeper one.

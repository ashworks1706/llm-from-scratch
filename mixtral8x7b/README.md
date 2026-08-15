this is the architecture of mixtral reusing some components from llama 3. the main difference is using sparse mixture of experts instead of a dense feedforward layer.

mixtral has 8 expert networks and for each token a router picks the top 2 experts to process it. this means only 25 percent of the parameters are active for any given token which makes it much faster than a dense model of equivalent size.

the attention mechanism is exactly the same as llama 3 with gqa and rope. only the feedforward part is different with the moe routing logic.

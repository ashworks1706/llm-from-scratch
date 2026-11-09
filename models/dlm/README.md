diffusion language models generate text by starting from a fully masked sequence and unmasking tokens over several steps instead of predicting one token at a time left to right.

masked diffusion like llada trains a bidirectional transformer to predict masked tokens at a random masking ratio. block diffusion like diffusiongemma keeps left to right generation between blocks but runs diffusion inside each block, which lets it reuse a kv cache. the notebook analyses real dlm runs and the python files build both versions on a small model.

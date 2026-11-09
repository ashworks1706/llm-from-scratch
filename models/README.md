models holds the architecture implementations. every other track uses them, interp runs on them, training trains them and inference serves them. run files as modules from the repo root, for example python -m models.llama3.inference.

llama3 is the baseline transformer decoder with grouped query attention and rotary embeddings. this is the foundation that most modern llms build on.

mixtral8x7b adds sparse mixture of experts where only some experts activate per token. this dramatically increases model capacity without proportional compute increase.

deepseekv3 implements multi head latent attention which compresses the kv cache for efficient long context. uses mixture of experts with load balancing.

kimik2 focuses on extreme long context by optimizing rope parameters and attention patterns for sequences up to millions of tokens.

paligemma2 is a vision language model that processes both images and text using a shared transformer architecture.

qwen35, deepseekv4, gemma4 and mamba2 are the 2026 efficiency architectures, linear attention hybrids, sparse and compressed attention, kv sharing and state space models.

dlm covers diffusion language models, the notebook analyses real runs and the python files build masked and block diffusion.

stablediffusion implements diffusion models for image generation starting from noise and iteratively denoising, and dit is the modern flow matching transformer version.

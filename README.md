this repository is where i build machine learning models from scratch to deeply understand how they work. everything is implemented from the ground up with detailed notes explaining the math, intuition and design choices behind each component.

the goal is complete ml engineering knowledge covering architectures, training pipelines, optimization techniques and deployment.

## architectures

llama3 is the baseline transformer decoder with grouped query attention and rotary embeddings. this is the foundation that most modern llms build on.

mixtral8x7b adds sparse mixture of experts where only some experts activate per token. this dramatically increases model capacity without proportional compute increase.

deepseekv3 implements multi head latent attention which compresses the kv cache for efficient long context. uses mixture of experts with load balancing.

kimik2 focuses on extreme long context by optimizing rope parameters and attention patterns for sequences up to millions of tokens.

paligemma2 is a vision language model that processes both images and text using a shared transformer architecture.

stable diffusion implements diffusion models for image generation starting from noise and iteratively denoising.

## training pipelines

pretraining trains models on massive text corpora to predict next tokens. this is where models learn language patterns, facts and reasoning.

sft does supervised finetuning on instruction response pairs to teach models to follow instructions and have conversations. only computes loss on responses not instructions.

rl implements direct preference optimization to align models with human preferences using chosen vs rejected response pairs. simpler and more stable than ppo.

distillation compresses a large teacher model into a smaller student by training on soft probability distributions rather than hard labels.

## optimization techniques

lora is parameter efficient finetuning using low rank matrices. trains less than 1 percent of parameters by injecting small adapters into attention layers.

inference folder has quantization to reduce model size to int8, batched inference for throughput, and efficient kv cache management.

## fundamentals

pytorch implements core concepts from scratch including tensors, autograd, modules, optimizers and training loops. understanding these is essential for everything else.

cnn covers convolutional networks for vision including convolution, pooling, resnet blocks and image classification on cifar10.

lstm implements recurrent networks for sequences including vanilla rnn, lstm cells, sequence prediction and sentiment analysis.

vae covers variational autoencoders for generative modeling with latent spaces and generation of new samples.


TODO - 

[ ] Create a chatbot expert in a specific field ros library, verilog 

[ ] Align a chatbot to be a good teacher

[ ] Process documents with verilog circuits, flowcharts

[ ] deep seek r1

[ ] PPO

[ ] GRPO

[ ] Stable diffusion 

[ ] DLoRA

[ ] temporal attention 

[ ] Mochi 1 (video generation)

[ ] NeRF (3 model generation)


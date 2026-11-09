foundations is everything the rest of the repo builds on.

pytorch implements core concepts from scratch including tensors, autograd, modules, optimizers and training loops. understanding these is essential for everything else.

einops covers readable tensor manipulation. rearrange, reduce, repeat and einsum make shape changes explicit which is important when implementing attention and multi head models.

tokenizers covers how text becomes model inputs. includes bpe, sentencepiece, tiktoken usage and training a tokenizer from scratch.

huggingface covers the practical ecosystem around modern models including transformers, datasets, generation, trainer and accelerate.

cnn covers convolutional networks for vision including convolution, pooling, resnet blocks and image classification on cifar10.

lstm implements recurrent networks for sequences including vanilla rnn, lstm cells, sequence prediction and sentiment analysis.

vae covers variational autoencoders for generative modeling with latent spaces and generation of new samples.

tutorial and gpt from scratch notebooks are end to end walkthroughs for building transformer language models from raw text through tokenization, training and generation.

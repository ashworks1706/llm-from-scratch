this repository is where i build machine learning models from scratch to deeply understand how they work. everything is implemented from the ground up with detailed notes explaining the math, intuition and design choices behind each component.

the focus is on three areas, inference, alignment and generalization. foundations, models and training are the shared base that all three build on.

## layout

foundations has the basics, pytorch, einops, tokenizers, huggingface, cnn, lstm, vae and the end to end tutorial notebooks.

models has the architecture implementations, from llama3, mixtral, deepseekv3 and paligemma2 to the 2026 hybrids like qwen35, deepseekv4, gemma4 and mamba2, plus diffusion models like dit.

training covers pretraining, sft, lora, rl, distillation and the training systems around them like evals, scaling laws, data curation and distributed training.

inference covers practical model execution after training. the python reference files introduce quantization, batching and paged kv caches while the numbered rust track uses candle for real model execution, systems lessons for scheduling and serving, and cudarc gpu work for kernels.

alignment covers mechanistic interpretability with patching, probes, saes, steering and model diffing, and alignment with reward hacking, emergent misalignment, cot monitoring, ai control and model organisms.

generalization covers going beyond next token imitation with energy based transformers, jepa world models, video world models and test time compute.

extras holds useful topics outside the main focus like mlops, embeddings, speech codecs and transfusion.

revision has short problem sets for brushing up python, rust, typescript, dsa, ml, cuda and systems. data, utils, assets and guide are shared helpers.

all todos live here, in the order to go through them.

TODO -

1. Unblockers (other lessons need these)

[ ] fix broken files
- [ ] training/sft/train.py syntax error at line 190, training/sft/dataset.py pass stub
- [ ] models/paligemma2/processor.py indentation error at line 20
- [ ] foundations/tutorial.py syntax error at line 68
- [ ] models/deepseekv3/inference.py imports a config.py that does not exist

[ ] training/evals: 01 bits per byte, 02 few shot harness, 03 error bars, 05 llm judge pitfalls

[ ] training/rl: reinforce, ppo, grpo with dapo / gspo / cispo options, rlvr rewards (dpo done)

2. Alignment

[ ] interp: 01 activation patching, 02 attribution patching, 03 linear probes, 04 lenses, 05 induction heads

[ ] sae: 02 superposition and sae variants, 05 evaluation and baselines, 06 feature steering, 07 transcoders, 08 attribution graphs

[ ] steering: 01 caa, 02 refusal direction, 03 persona vectors and sycophancy, 04 steering reliability

[ ] safety: 01 reward hacking, 02 emergent misalignment, 03 harmful finetuning, 04 unlearning vs filtering

[ ] oversight: 01 cot monitoring, 04 sleeper agents, 05 sandbagging, 03 probe classifiers, 02 ai control, 06 auditing model organisms

[ ] later: diffing, params, interp 06 thought anchors, safety 05 character training, safety 06 weak to strong

3. Inference (in parallel with alignment)

[ ] gpu 13 host file (the run list uses 13_reductions_and_shared_memory_host but it does not exist)

[ ] gpu 16, 17 (missing files)

[ ] systems 21-25: batching and streaming, prefill decode scheduler, kv cache policy, server and api, metrics (20 done)

[ ] gpu 28 flash attention, 29 paged attention (kernels empty)

[ ] gpu 30 quantized attention kernel, 31 gptq / awq / kv quantization, 32 sparse attention (30 and 32 missing files)

[ ] systems 33 speculative decoding, 34 prefix caching, 35 disaggregated serving, 36 moe serving

4. Models (build each one when an inference lesson needs it)

[ ] gemma4, before systems 23 kv cache policy: sliding window and global layers, qk norm, cross layer kv sharing, k=v, per layer embeddings, encoder free vision

[ ] deepseekv4, before gpu 32 sparse attention: lightning indexer, csa and hca compressed attention, mhc, attention residuals, engram, muon

[ ] qwen35: gated deltanet hybrid with chunkwise training, gated attention, partial rope, mtp, kimi delta attention extension

[ ] mamba2: state space duality, selective scan, latent moe, nemotron 3 style hybrid

[ ] dlm: masked diffusion (llada) and block diffusion (diffusiongemma)

[ ] paligemma2: native resolution vit upgrade

[ ] stablediffusion: clip, encoder, decoder, unet as .py files (only the notebook has code)

5. Generalization

[ ] testtimecompute: self consistency, best of n with verifier, beam search, compute optimal allocation

[ ] ebt: energy based transformers and energy based reward models (only the notebook exists)

[ ] vjepa: image jepa, video tubelets, ema target, sigreg, action conditioned world model with cem planning

[ ] dit: adaln zero, flow matching, mmdit, single stream dit, repa, meanflow, diffusionnft

[ ] videodiffusion: 3d causal vae, video dit, diffusion forcing, self forcing

6. Training (whenever there is time)

[ ] training/rl rest: rollout correction, length control, generative reward models, agentic rl

[ ] training/deepseekr1: all files are scaffold only

[ ] training/distillation: on-policy distillation, multi teacher distillation

[ ] training/evals: 04 contamination checks

[ ] speedrun: modded nanogpt tricks one at a time, muon / soap / schedule free

[ ] scalinglaws: isoflop sweep, fitting laws, mup, completep

[ ] datacuration: html extraction, quality classifier, minhash dedup, pii filters, decontamination

[ ] distributed: ddp, zero / fsdp, tensor parallel, activation checkpointing, fp8 training

[ ] longcontext: rope scaling (pi, ntk, yarn, abf), long data mixing, needle and ruler evals, context parallelism

[ ] data: datasets, packing

[ ] foundations/pytorch/01_tensors.py todo 5

7. Extras (only if curious)

[ ] embeddings: infonce, llm to embedder, matryoshka, reranker, ann search

[ ] speechcodec: rvq codec, semantic codebook, audio encoder asr, thinker talker

[ ] transfusion: one model for text next token and image flow matching

Revision (alongside the matching track, rust and cuda with inference, ml with training)

[ ] Python Revision (01-07 done, 08-24 left)

[ ] Rust Revision (01-22 left)

[ ] TypeScript Revision (01-17 left)

[ ] DSA Revision (01-10 left)

[ ] ML Revision (01-11 left)

[ ] Cuda Revision (01-10 left, kernel bodies empty)

[ ] Systems Revision (01-06 left)

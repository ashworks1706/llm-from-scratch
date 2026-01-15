this repository is where i build language models from scratch to understand how they work. the goal is educational so i can learn the internals of modern llms by implementing them myself with detailed notes and comments.

each subdirectory contains a different model architecture. llama3 is the baseline transformer decoder using grouped query attention. mixtral8x7b extends that with sparse mixture of experts. deepseekv3 adds multi-head latent attention for compression. kimik2 focuses on extreme long context with optimized rope parameters.

i try to keep the code readable with inline explanations of what each component does and why certain design choices were made. this helps me remember the reasoning when i come back to review the code later.

Distillation (2-3 hours)
  - distillation/teacher_student.py - Knowledge distillation (~1.5 hours)
  - distillation/dataset.py - Generate teacher outputs (~45 min)
  - distillation/README.md (~15 min)

Integration & Testing (1-2 hours)
  - End-to-end pipeline test
  - Examples




this repository is where i build language models from scratch to understand how they work. the goal is educational so i can learn the internals of modern llms by implementing them myself with detailed notes and comments.

each subdirectory contains a different model architecture. llama3 is the baseline transformer decoder using grouped query attention. mixtral8x7b extends that with sparse mixture of experts. deepseekv3 adds multi-head latent attention for compression. kimik2 focuses on extreme long context with optimized rope parameters.

i try to keep the code readable with inline explanations of what each component does and why certain design choices were made. this helps me remember the reasoning when i come back to review the code later.


1. Pretraining (2.5 hours remaining)

  - ✅ dataset.py 
  - ⏳ train.py (filling TODOs + testing)
  - 📝 README.md

2. SFT Pipeline (2-3 hours)

Files to create:

  - sft/dataset.py - Handle instruction-response format (~45 min)
  - sft/train.py - Similar to pretraining but different data (~30 min)
  - sft/format_data.py - Convert raw data to instruction format (~30 min)
  - sft/README.md (~15 min)


3. Inference Pipeline (2-3 hours)

Files to create:

  - inference_opt/batched_inference.py - Batch multiple requests (~45 min)
  - inference_opt/quantization.py - 8-bit/4-bit quantization (~1 hour)
  - inference_opt/server.py - Simple API server (~45 min)
  - Already have: paged_kv_cache.py
  - inference_opt/README.md (~15 min)

4. RL Pipeline (3-4 hours) - MOST COMPLEX

Files to create:

  - rl/reward_model.py - Train reward model (~1 hour)
  - rl/ppo_trainer.py - PPO implementation (~1.5 hours)
  - rl/dpo_trainer.py - DPO (simpler alternative) (~1 hour)
  - rl/dataset.py - Preference pairs dataset (~30 min)
  - rl/README.md (~15 min)

Key concepts: Reward modeling, policy gradients, KL divergence penalty

5. Distillation (2-3 hours)

Files to create:

  - distillation/teacher_student.py - Knowledge distillation (~1.5 hours)
  - distillation/dataset.py - Generate teacher outputs (~45 min)
  - distillation/README.md (~15 min)

6. Integration & Testing (1-2 hours)

  - End-to-end pipeline test
  - Documentation
  - Examples


⏰ TIMELINE SUMMARY

┌──────────────────────┬─────────────┬───────────┐
│ Task                 │ Time        │ Priority  │
├──────────────────────┼─────────────┼───────────┤
│ Pretraining (finish) │ 2.5h        │ 🔴 HIGH   │
├──────────────────────┼─────────────┼───────────┤
│ SFT Pipeline         │ 2-3h        │ 🔴 HIGH   │
├──────────────────────┼─────────────┼───────────┤
│ Inference Pipeline   │ 2-3h        │ 🟡 MEDIUM │
├──────────────────────┼─────────────┼───────────┤
│ RL Pipeline          │ 3-4h        │ 🟡 MEDIUM │
├──────────────────────┼─────────────┼───────────┤
│ Distillation         │ 2-3h        │ 🟢 LOW    │
├──────────────────────┼─────────────┼───────────┤
│ Integration          │ 1-2h        │ 🟢 LOW    │
├──────────────────────┼─────────────┼───────────┤
│ TOTAL                │ 14-18 hours │           │
└──────────────────────┴─────────────┴───────────┘



this folder implements reinforcement learning from human feedback. after supervised finetuning the model can follow instructions but may not align with human preferences on subjective qualities like helpfulness, harmlessness and honesty.

dpo uses direct preference optimization for single-stage offline training. it takes preference pairs where humans labeled one response as better than another for the same prompt. instead of absolute labels we have relative preferences which are more natural. the key insight is that we can directly optimize the policy to increase likelihood of chosen responses and decrease rejected ones, while using kl divergence to prevent drifting from the reference model (sft).

ppo uses proximal policy optimization with a learned reward model. it generates responses on-policy, scores them with a reward model, computes advantages using a value function, and updates with clipped policy gradients. this allows online interaction but requires training a separate reward model. the clipping mechanism prevents too-large policy changes which keeps training stable.

grpo uses group relative policy optimization by generating k responses per prompt and scoring each one. the advantage of a response is its reward minus the group mean, divided by the group std, so there is no value model. most 2026 labs use grpo with fixes from dapo, gspo and cispo like token level loss, clip higher, dropping groups where every answer got the same reward, and no kl penalty.

rlvr means the rewards come from checks that can be run, like comparing a math answer or running unit tests, instead of a learned reward model. this is what turned chat models into reasoning models. reinforce is a small warm up before ppo and grpo.

rollout correction fixes the mismatch between the fast engine that generates rollouts and the model being trained, using truncated or masked importance sampling. length control trains low, medium and high effort modes. generative reward models use an llm judge with a rubric for tasks that cannot be checked by code, and agentic rl trains multi turn tool use in a sandbox where only the final outcome is rewarded.

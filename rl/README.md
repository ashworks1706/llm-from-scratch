this folder implements reinforcement learning from human feedback specifically using direct preference optimization. after supervised finetuning the model can follow instructions but may not align with human preferences on subjective qualities like helpfulness, harmlessness and honesty.

the dpo dataset contains preference pairs where humans have labeled one response as better than another for the same prompt. instead of absolute labels we have relative preferences which are more natural for humans to provide. each example has a prompt, a chosen response that was preferred, and a rejected response.

the key insight of dpo is that we can directly optimize the policy model to increase the likelihood of chosen responses and decrease likelihood of rejected ones, while using kl divergence to prevent the model from drifting too far from the reference model. the reference model is typically the sft model frozen in place.

the trainer implements this by computing log probabilities for both chosen and rejected responses under both the policy and reference models. the loss encourages the policy to assign higher probability to chosen responses relative to rejected ones compared to what the reference model would do. the beta parameter controls how much we allow the policy to diverge.

this approach is simpler than ppo which requires training a separate reward model and doing multiple rounds of on policy sampling. dpo directly optimizes for human preferences in a single stage and tends to be more stable. after rl the model better matches human preferences while still maintaining its capabilities from pretraining and sft.

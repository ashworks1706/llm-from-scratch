lora stands for low rank adaptation and is a parameter efficient finetuning technique. instead of updating all billions of parameters in a model, we freeze the original weights and add small trainable matrices that adapt the behavior.

the core idea is that weight updates during finetuning are low rank, meaning they can be decomposed into smaller matrices. for a weight matrix W, instead of updating it directly, we keep it frozen and add the product of two smaller matrices A and B. so the new weight becomes W + BA where B has shape output x rank and A has shape rank x input.

the rank is typically very small like 4 or 8 compared to the full dimensions which might be thousands. this means we only train a tiny fraction of parameters, often less than 1 percent of the original model. this dramatically reduces memory usage and training time.

i inject lora into the attention projection layers since those capture most of the models behavior. the scaling factor alpha controls how much the lora adaptation affects the output. during inference we can merge the lora weights back into the original weights so there is no speed penalty.

lora is especially useful when you want to create multiple specialized versions of a model or when gpu memory is limited. you can train different lora adapters for different tasks and swap them as needed without retraining the entire model each time.

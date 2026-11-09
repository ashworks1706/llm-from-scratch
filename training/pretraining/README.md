this folder contains the pretraining pipeline for language models. pretraining is the first stage where the model learns to predict the next token in a sequence by training on massive amounts of raw text data.

the dataset file handles loading and tokenizing text data. it splits long sequences into chunks of max sequence length and creates input target pairs where the target is just the input shifted by one token. this teaches the model to predict what comes next at each position.

the training script implements the core pretraining loop. for each batch it does a forward pass to get predictions, computes cross entropy loss between predictions and true next tokens, then backpropagates to update weights. the model learns language patterns, grammar, facts and reasoning abilities through this next token prediction objective.

i use gradient accumulation to simulate larger batch sizes when gpu memory is limited. checkpointing saves model state periodically so training can resume if interrupted. the loss typically starts high around 10 and decreases to around 2-3 as the model learns.

pretraining is computationally expensive and can take weeks on large datasets but it creates a foundational model that understands language. this pretrained model is then used as the starting point for supervised finetuning and alignment.

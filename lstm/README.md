recurrent neural networks and lstms for sequence modeling implemented from scratch. rnns process sequences one element at a time maintaining hidden state as memory.

vanilla rnn implements the simplest recurrent architecture where hidden state is fed back as input. shows how recurrence enables processing variable length sequences. demonstrates the vanishing gradient problem that limits learning long range dependencies.

lstm cell implements long short term memory with three gates. forget gate decides what to remove from memory. input gate decides what new information to add. output gate decides what to expose from cell state. these gates allow learning long term dependencies that vanilla rnns cannot handle.

sequence prediction trains character level language model. the lstm learns to predict the next character given previous characters. includes text generation by sampling from the model with temperature control.

sentiment analysis uses lstm for text classification. processes word sequences to classify sentiment as positive or negative. shows how to handle variable length inputs and use embeddings.

comparison to transformers explains why attention mechanisms replaced recurrent architectures in modern nlp. transformers process sequences in parallel rather than sequentially enabling much faster training. the llama model built earlier uses self attention instead of recurrence but both solve sequence modeling problems.

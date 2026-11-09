mamba 2 is a state space model, which means it processes a sequence like an rnn with a fixed size hidden state instead of attending over all previous tokens. what makes it different from old rnns is that the state update is selective, so the input decides how much to keep and how much to forget at each step.

the main idea of mamba 2 is state space duality. if the decay is a scalar per head then the recurrence can be written as a masked attention like matrix multiply. this means training can use fast matmuls in chunks, and inference can use the cheap recurrent form. this is very close to how gated deltanet works in qwen3next, so it is good to compare the two side by side.

nemotron 3 and similar models do not use pure mamba. they mix mostly mamba 2 layers with a few attention layers and moe feedforward, for the same reason qwen3 next keeps some full attention. the hybrid model file builds that version.

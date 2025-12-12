this is my implementation of kimi k2 from moonshot ai. the main thing about kimi is the extremely long context window of 200k tokens which is much longer than most other models.

the architecture is pretty similar to llama 3 using GQA, RoPE, SwiGLU, and RMSNorm. the key difference is in how rope is configured with a much higher theta value of 500000 instead of the standard 10000. this slower rotation rate lets the model maintain positional distinctions across way longer sequences.

unlike deepseek which uses compressed attention or mixtral which uses mixture of experts, kimi keeps things simple with a standard dense architecture. the bet here is that for understanding very long documents you need all the parameters working together rather than routing to sparse experts or compressing representations.

the implementation reuses most components from llama 3 since the core architecture is the same. the main changes are in rope.py for the theta parameter and making sure the kv cache can handle the much longer context efficiently.

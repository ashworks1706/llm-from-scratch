this is my implementation of deepseek v3 which is the most complex architecture so far. it combines multi-head latent attention with mixture of experts to be extremely parameter efficient.

instead of storing full key and value vectors, mla compresses them into tiny latent representations which saves a ton of memory in the kv cache. it splits the key into content and position parts to apply rope correctly on compressed vectors. the model also uses 64 experts with top 6 routing which is much more fine grained than mixtral.

the feedforward uses shared experts plus routed experts for better parameter sharing. all this compression and routing makes it complicated but very efficient for the number of parameters.

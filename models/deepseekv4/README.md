deepseek v4 keeps the moe from v3 but changes attention and the residual path so it can handle a one million token context.

the first change is sparse attention. in v3.2 deepseek added a lightning indexer, which is a very cheap scorer with a few small heads in low precision. it scores every past token for the current query and only the top k tokens get passed to real attention. this means the expensive attention only looks at a small selected set and not the whole context.

v4 builds on this with two kinds of compressed attention. compressed sparse attention groups a few neighbouring kv entries into one compressed entry and then runs the lightning indexer over those compressed entries. heavily compressed attention squeezes the kv much harder and runs dense attention over the small result. there is also a sliding window so the most recent tokens are always seen exactly.

the second change is manifold constrained hyper connections. instead of one residual stream there are several parallel streams. a mixing matrix moves information between them and sinkhorn iterations keep that matrix doubly stochastic, which means every row and column adds up to one. because of this the signal cannot blow up or vanish across very deep stacks. kimi k3 changes the residual differently with attention residuals, where each layer attends over the outputs of earlier layers, so both are here to compare.

v4.1 flash drops heavily compressed attention, reuses kv across layers, and adds engram, which is a big hashed n gram lookup table that gives the model memory without extra compute. qwen 3.8 next added n gram embeddings too. the model is trained with the muon optimizer which orthogonalizes the update matrix with a newton schulz iteration instead of using adam on matrix weights.

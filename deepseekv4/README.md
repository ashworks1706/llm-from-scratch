deepseek v4 keeps the moe and multi token prediction from v3 but changes attention and the residual path so it can handle a one million token context.

the first change is sparse attention. in v3.2 deepseek added a lightning indexer, which is a very cheap scorer with a few small heads in low precision. it scores every past token for the current query and only the top k tokens get passed to real attention. this means the expensive attention only looks at a small selected set and not the whole context.

v4 builds on this with two kinds of compressed attention. compressed sparse attention groups a few neighbouring kv entries into one compressed entry and then runs the lightning indexer over those compressed entries. heavily compressed attention squeezes the kv much harder and runs dense attention over the small result. there is also a sliding window so the most recent tokens are always seen exactly. together the model keeps compressed summaries of the far past, picks out the parts it needs and sees the recent past exactly.

the second change is manifold constrained hyper connections. instead of one residual stream there are several parallel streams. a mixing matrix moves information between them and sinkhorn iterations keep that matrix doubly stochastic, which means every row and column adds up to one. because of this the signal cannot blow up or vanish across very deep stacks.

the model is trained with the muon optimizer which orthogonalizes the update matrix with a newton schulz iteration instead of using adam on matrix weights.

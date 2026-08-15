this is the architecture of llama 3 from scratch. its a standard transformer decoder with grouped query attention which reduces the kv cache size by having multiple query heads share the same key and value heads.

uses rope for positional embeddings instead of absolute positions which helps the model generalize to longer sequences. the feedforward layer uses swiglu activation which is a gating mechanism that works better than standard relu for language models. rmsnorm is used instead of layernorm since it works just as well but is simpler and faster.

this is the foundation that the other models build on top of with their own modifications and improvements.

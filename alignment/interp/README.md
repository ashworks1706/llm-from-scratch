interp covers the basic tools of mechanistic interpretability, finding which parts of a model cause a behaviour and reading what the model represents in the middle of the network.

activation patching swaps activations between a clean and a corrupted prompt to find what matters, and attribution patching does the same with gradients so it is fast enough for whole circuits. linear probes and lenses read information out of hidden states. most of this runs on gpt 2 small with the ioi task.

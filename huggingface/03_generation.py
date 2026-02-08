# text generation using different decoding strategies
# going from model logits to actual text output

# topics to cover:
# - model.generate() method and parameters
# - greedy decoding (argmax at each step)
# - beam search (keeping top k candidates)
# - sampling methods (temperature, top_k, top_p)
# - generation config (max_length, do_sample, temperature)
# - stopping criteria (eos token, max length)
# - how attention mask affects generation
# - streaming generation token by token

# OBJECTIVE: generate text with different strategies and see quality differences
# understand tradeoff between deterministic (greedy) and creative (sampling)

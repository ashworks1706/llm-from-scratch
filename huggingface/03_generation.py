# there are three ways to decode the model's logits 

# greedy decoding, which is just getting argmax of every next word from llm 
# this is usually fast deterministic but repetetive/ boring, no creativity in next word, it goes wrong, it goes wrong forever

# or beam search. to keep top K cnadidates at each step not just 1
# like it does parallele results and picks top two from probability ofo which one is better 
# but how does beam search even determine the probability ? 
# in the last layer of llms we have logits which are unnormalized probabilities of the next word, we 
# can apply softmax to get actual probabilities, and then we can pick the top 
# K candidates based on those probabilities.
# this is better quality than greedy just a lil bit slower 

# Sampling is another method where instead of picking the most probable next word, we randomly sample 
# from the distribution of possible next words. This can lead to more diverse and creative 
# outputs, but it can also result in less coherent or relevant responses.
# there are different sampling strategies like top-k sampling, where we only consider the top K most probable 
# next words and sample from them, and nucleus sampling (top-p), where we consider the smallest set of words 
# whose cumulative probability exceeds a certain threshold p and sample from that set.
# this is generally the best quality but also the slowest method, it can produce very creative and 
# diverse outputs, but it can also produce less coherent responses if not tuned properly.









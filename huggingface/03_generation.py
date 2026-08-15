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


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids 

# Greedy decoding
greedy_output = model.generate(
    input_ids, 
    max_length=50, 
    do_sample=False, 
    temperature=1.0,
    num_beams=1 # this is the default value for greedy decoding, 
    # it means we are only keeping the top 1 candidate at each step
)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


# Beam search decoding
beam_output = model.generate(
    input_ids,
    max_length=50,
    do_sample=False,
    num_beams=5, # this means we are keeping the top 5 candidates at each step
    early_stopping=True # this means we will stop generating when we reach the end of the sequence
)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Sampling decoding
sampling_output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True, # this means we are sampling from the distribution of next words
    temperature=0.7, # this controls the randomness of the sampling, lower values make it more deterministic, higher values make it more random
    top_k=50, # this means we are only considering the top 50 most probable next words for sampling
    top_p=0 # this means we are not using nucleus sampling, we are only using top-k sampling

)

# diff b/w top k sample vs top p sample 
# top-k sampling limits the number of candidates to a fixed K, while nucleus sampling limits the candidates based on a 
# cumulative probability threshold p. 
# top-k sampling: we only consider the top K most probable next words and sample from them. This means that if K 
# is set to 50, we will only look at the 50 most likely next words and randomly select one of them based on their 
# probabilities.
# nucleus sampling (top-p): we consider the smallest set of words whose cumulative probability exceeds a certain threshold 
# pand sample from that 
# set. For example, if p is set to 0.9, we will look at the next words in order of their probabilities and keep adding 
# them to our candidate set until the cumulative probability of those words exceeds 0.9. Then we will randomly sample 
# from that set of words.
print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))


# if temp = 0 
# softmax becomes infinitely sharp, the highest logit becomes probabiltity 1 and rest 0, so it becomes ggreedy 
# top_p=0.9 filters out low prob tokens "tail" of distribution, it ignores garbage 10% tokens


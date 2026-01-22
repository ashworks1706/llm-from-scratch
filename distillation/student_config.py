# define smaller model for distillation
# student learns from teacher's soft targets (probability distributions)
# instead of just hard labels (0/1), capturing teacher's reasoning

# teacher model typically has:
# num_layers = 32, embedding_size = 4096, num_attention_heads = 32, num_kv_heads = 8
# total params: ~7B

# student model (smaller):
# num_layers = 12, embedding_size = 2048, num_attention_heads = 16, num_kv_heads = 4
# total params: ~1B (7x smaller!)

class StudentConfig:
    num_layers = 12  # vs teacher's 32
    # each layer has millions of parameters
    # cutting layers = biggest size reduction
    
    embedding_size = 2048  # vs teacher's 4096
    # affects ALL weight matrices (wq, wk, wv, wo, MLPs)
    # most matrices are embedding_size x embedding_size
    
    num_attention_heads = 16  # vs teacher's 32
    # less impact on size but affects model capacity
    # cutting heads reduces parallelism in attention
    
    num_kv_heads = 4  # vs teacher's 8
    # must divide num_attention_heads evenly (16 / 4 = 4 queries per KV)
    # grouped query attention: saves memory
    
    vocab_size = 128000  # MUST match teacher (same tokenizer!)
    # if different, can't use same tokenizer = incompatible
    
    max_sequence_length = 2048  # can be same or smaller than teacher
    # student can handle same context length
    
    tokenizer_type = "tiktoken"  # MUST match teacher
    # using Llama3's byte-pair encoding tokenizer
    
    dropout_rate = 0.1  # same as teacher
    rms_norm_eps = 1e-5  # same as teacher (for numerical stability)
    rope_theta = 500000.0  # same as teacher (RoPE frequency)
    
    learning_rate = 1e-4  # can be higher than teacher (fewer params to update)
    batch_size = 8  # can be larger (student is smaller)
    num_epochs = 5  # typically train for fewer epochs than pretraining
    
    temperature = 5.0  # softens teacher's probability distributions
    # higher T = softer (reveals more info about wrong answers)
    # typical range: 3-10
    
    alpha = 0.9  # balance between soft and hard loss
    # 0.9 means: 90% learn from teacher, 10% learn from true labels
    # if teacher makes mistakes, hard loss corrects it
    
    model_name = "llama3_student"
    version = "distilled_1b"




class Config:
    def __init__(self, model_name, version, max_sequence_length, embedding_size,
                 num_attention_heads, num_layers, dropout_rate, learning_rate,
                 batch_size, num_epochs, vocab_size, tokenizer_type, num_kv_heads, rms_norm_eps, rope_theta):
        self.model_name = model_name
        self.version = version
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.vocab_size = vocab_size
        self.tokenizer_type = tokenizer_type
        self.num_kv_heads = num_kv_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

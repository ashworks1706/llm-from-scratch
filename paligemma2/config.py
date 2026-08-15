# config for paligemma model

class VisionConfig:
    def __init__(
        self,
        hidden_size=1152,
        image_size=224,
        patch_size=16,
        num_attention_heads=16,
        num_hidden_layers=27,
        intermediate_size=4304
    ):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size


class TextConfig:
    def __init__(
        self,
        hidden_size=2048,
        num_attention_heads=8,
        num_key_value_heads=1,
        num_hidden_layers=18,
        intermediate_size=16384,
        vocab_size=257216,
        rms_norm_eps=1e-6
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=256000,
        pad_token_id=0
    ):
        self.vision_config = vision_config or VisionConfig()
        self.text_config = text_config or TextConfig()
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id

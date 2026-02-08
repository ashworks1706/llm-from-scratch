# supervised fine tuning with SFTTrainer
# simplified training on instruction datasets

# topics to cover:
# - SFTTrainer vs standard Trainer differences
# - dataset formatting (prompt, completion columns)
# - automatic loss masking on prompts
# - packing multiple examples per sequence
# - max_seq_length and truncation strategies
# - comparing to manual sft in sft/train.py
# - dataset_text_field for simple text datasets
# - formatting_func for custom formats
# - when to use SFTTrainer vs custom loop

# OBJECTIVE: train instruction following model using SFTTrainer
# compare code simplicity to manual implementation

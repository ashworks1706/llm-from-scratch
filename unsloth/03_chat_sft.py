"""
Chat SFT with Unsloth - Conversational Fine-tuning

Level up from basic SFT! Now we'll train on multi-turn conversations.
Perfect for building chatbots and assistants.

Dataset: ShareGPT format (multi-turn conversations)
Training time: 1-2 hours on T4 GPU

What's new:
- Chat templates for multi-turn conversations
- Proper system/user/assistant formatting
- Handling conversation history
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# STEP 1: Model + Chat Template Setup
# ============================================================================

model_name = "unsloth/Llama-3.2-1B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Apply chat template - this formats conversations correctly
# Llama uses specific tokens like <|begin_of_text|>, <|start_header_id|>, etc.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Options: llama-3, mistral, chatml, etc.
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================================
# STEP 2: Dataset - ShareGPT Format
# ============================================================================

# ShareGPT format: list of conversations with multiple turns
# {"conversations": [
#     {"from": "human", "value": "Hello!"},
#     {"from": "gpt", "value": "Hi there!"},
#     {"from": "human", "value": "How are you?"},
#     {"from": "gpt", "value": "I'm doing well!"}
# ]}

# You can use real datasets like:
# - "OpenAssistant/oasst1" - high quality human conversations
# - "HuggingFaceH4/ultrachat_200k" - diverse chat data
# - "stingning/ultrachat" - large scale conversations

# For this example, let's create a small custom dataset
from datasets import Dataset

chat_data = [
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a branch of AI where computers learn patterns from data without being explicitly programmed."},
            {"role": "user", "content": "Can you give an example?"},
            {"role": "assistant", "content": "Sure! Email spam filters learn to identify spam by analyzing thousands of spam and non-spam emails."}
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a Python function to calculate factorial"},
            {"role": "assistant", "content": "Here's a recursive implementation:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```"},
        ]
    },
    # Add more examples...
]

# For real training, use a larger dataset:
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train[:1000]")

def format_chat(example):
    """Convert ShareGPT format to model's chat format"""
    # The tokenizer.apply_chat_template handles all the special tokens
    conversation = example["conversations"]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False  # Don't add prompt for training
    )
    return {"text": text}

dataset = dataset.map(format_chat)

# ============================================================================
# STEP 3: Training with Chat Data
# ============================================================================

training_args = TrainingArguments(
    output_dir="./outputs_chat_sft",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_steps=10,
    save_steps=100,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

print("🚀 Starting chat SFT training...")
trainer.train()

# ============================================================================
# STEP 4: Save and Test
# ============================================================================

model.save_pretrained("lora_chat_model")
tokenizer.save_pretrained("lora_chat_model")

# Test with multi-turn conversation
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Add generation prompt for inference
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*50)
print("Chat Test:")
print(response)
print("="*50)

# ============================================================================
# PRO TIP: Continue the conversation!
# ============================================================================

# To continue chatting, keep appending to messages list:
# messages.append({"role": "assistant", "content": response})
# messages.append({"role": "user", "content": "Tell me more about it"})
# ... then generate again

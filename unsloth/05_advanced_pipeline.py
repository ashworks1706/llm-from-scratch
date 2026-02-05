"""
Full Production Pipeline - Data → SFT → DPO → Deploy

This combines everything into a realistic workflow you'd use in production.

Pipeline stages:
1. Data preprocessing and quality filtering
2. SFT training with validation
3. DPO alignment with evaluation  
4. Model merging and quantization
5. Inference optimization

This is what you'd actually run for a real project!
"""

import os
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, DPOTrainer, DPOConfig
from transformers import TrainingArguments
import json

# ============================================================================
# Configuration
# ============================================================================

class PipelineConfig:
    # Model
    base_model = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    load_in_4bit = True
    
    # LoRA
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0
    
    # Paths
    output_dir = "./pipeline_output"
    sft_output = f"{output_dir}/sft"
    dpo_output = f"{output_dir}/dpo"
    final_output = f"{output_dir}/final_model"
    
    # Training
    sft_epochs = 1
    dpo_epochs = 1
    batch_size = 2
    gradient_accumulation = 4
    learning_rate_sft = 2e-4
    learning_rate_dpo = 5e-5
    
    # DPO
    dpo_beta = 0.1
    
    # Datasets
    sft_dataset = "philschmid/guanaco-sharegpt-style"
    sft_samples = 5000
    dpo_dataset = "Intel/orca_dpo_pairs"  
    dpo_samples = 1000

config = PipelineConfig()
Path(config.output_dir).mkdir(exist_ok=True, parents=True)

# ============================================================================
# STAGE 1: Data Preprocessing
# ============================================================================

print("="*70)
print("STAGE 1: Data Preprocessing")
print("="*70)

def clean_and_filter_sft_data(dataset, max_length=2048):
    """Filter out bad examples"""
    def is_valid(example):
        # Remove empty conversations
        if not example.get("conversations"):
            return False
        
        # Remove too short or too long
        text = str(example["conversations"])
        if len(text) < 50 or len(text) > max_length * 4:
            return False
            
        return True
    
    return dataset.filter(is_valid)

def clean_dpo_data(dataset):
    """Filter preference pairs"""
    def is_valid(example):
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        
        # Must have all fields
        if not (prompt and chosen and rejected):
            return False
        
        # Responses should be different
        if chosen == rejected:
            return False
            
        # Not too short
        if len(chosen) < 20 or len(rejected) < 20:
            return False
            
        return True
    
    return dataset.filter(is_valid)

# Load and prepare SFT data
print("Loading SFT dataset...")
sft_dataset = load_dataset(config.sft_dataset, split=f"train[:{config.sft_samples}]")
sft_dataset = clean_and_filter_sft_data(sft_dataset)
print(f"SFT dataset: {len(sft_dataset)} examples after filtering")

# Load and prepare DPO data
print("Loading DPO dataset...")
dpo_dataset = load_dataset(config.dpo_dataset, split=f"train[:{config.dpo_samples}]")
dpo_dataset = clean_dpo_data(dpo_dataset)
print(f"DPO dataset: {len(dpo_dataset)} examples after filtering")

# ============================================================================
# STAGE 2: SFT Training
# ============================================================================

print("\n" + "="*70)
print("STAGE 2: Supervised Fine-Tuning")
print("="*70)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.base_model,
    max_seq_length=config.max_seq_length,
    dtype=None,
    load_in_4bit=config.load_in_4bit,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Format SFT data
def format_sft(example):
    conversation = example["conversations"]
    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    return {"text": text}

sft_dataset = sft_dataset.map(format_sft)

# SFT training
sft_args = TrainingArguments(
    output_dir=config.sft_output,
    num_train_epochs=config.sft_epochs,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation,
    learning_rate=config.learning_rate_sft,
    warmup_steps=10,
    logging_steps=10,
    save_steps=200,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    dataset_text_field="text",
    max_seq_length=config.max_seq_length,
    args=sft_args,
)

print("🚀 Starting SFT training...")
sft_trainer.train()

# Save SFT model
model.save_pretrained(config.sft_output)
tokenizer.save_pretrained(config.sft_output)
print(f"✅ SFT complete! Model saved to {config.sft_output}")

# ============================================================================
# STAGE 3: DPO Alignment
# ============================================================================

print("\n" + "="*70)
print("STAGE 3: DPO Alignment")
print("="*70)

# Reload SFT model for DPO (fresh LoRA adapters)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.sft_output,
    max_seq_length=config.max_seq_length,
    dtype=None,
    load_in_4bit=config.load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# DPO training
dpo_config = DPOConfig(
    output_dir=config.dpo_output,
    num_train_epochs=config.dpo_epochs,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation,
    learning_rate=config.learning_rate_dpo,
    beta=config.dpo_beta,
    warmup_steps=5,
    logging_steps=10,
    save_steps=200,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dpo_dataset,
)

print("🚀 Starting DPO training...")
dpo_trainer.train()

# Save DPO model
model.save_pretrained(config.dpo_output)
tokenizer.save_pretrained(config.dpo_output)
print(f"✅ DPO complete! Model saved to {config.dpo_output}")

# ============================================================================
# STAGE 4: Model Merging & Quantization
# ============================================================================

print("\n" + "="*70)
print("STAGE 4: Model Merging & Quantization")
print("="*70)

# Reload for merging
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.dpo_output,
    max_seq_length=config.max_seq_length,
    dtype=None,
    load_in_4bit=config.load_in_4bit,
)

# Save in different formats for different use cases

# 1. Full precision merged (best quality, largest size)
print("Saving full precision model...")
model.save_pretrained_merged(
    f"{config.final_output}/16bit",
    tokenizer,
    save_method="merged_16bit",
)

# 2. GGUF quantized formats (for llama.cpp, Ollama)
print("Saving GGUF quantized models...")
model.save_pretrained_gguf(
    f"{config.final_output}/gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Good balance of size/quality
)

# 3. 4-bit quantized (for memory-constrained serving)
print("Saving 4-bit quantized model...")
model.save_pretrained_merged(
    f"{config.final_output}/4bit",
    tokenizer,
    save_method="merged_4bit",
)

print(f"✅ All formats saved to {config.final_output}/")

# ============================================================================
# STAGE 5: Evaluation & Testing
# ============================================================================

print("\n" + "="*70)
print("STAGE 5: Evaluation")
print("="*70)

FastLanguageModel.for_inference(model)

# Test prompts covering different capabilities
test_prompts = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a string.",
    "What are the ethical considerations of AI?",
    "Tell me a joke about programming.",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Prompt: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

# Save metadata
metadata = {
    "base_model": config.base_model,
    "sft_samples": len(sft_dataset),
    "dpo_samples": len(dpo_dataset),
    "config": vars(config),
}

with open(f"{config.output_dir}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print(f"Models saved in: {config.final_output}/")
print("Formats available:")
print("  - 16bit/     : Full precision (best quality)")
print("  - 4bit/      : Quantized (memory efficient)")
print("  - gguf/      : For llama.cpp/Ollama")

"""
Basic SFT with Unsloth - Instruction Fine-tuning

This is your STARTING POINT. We'll fine-tune Llama 3.2 1B on Alpaca dataset.
Training time: ~30-60 mins on T4 GPU (free Colab tier works!)

Key concepts you'll learn:
- Loading models efficiently with 4-bit quantization
- Injecting LoRA adapters with Unsloth
- Creating instruction datasets with proper formatting
- Training with SFTTrainer from TRL library
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# - 1B: Works on 8GB GPU (T4, Colab free)
# - 3B: Needs 16GB GPU (T4 with Colab Pro, RTX 4060)
# - 8B: Needs 24GB GPU (A10, RTX 4090)

model_name = "unsloth/Llama-3.2-1B-Instruct" 

max_seq_length = 2048  # Longer = more memory but can handle longer conversations
dtype = None  # Auto-detect. Options: torch.float16, torch.bfloat16
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# LoRA injects tiny trainable matrices into the model
# Instead of training 1B parameters, we train ~8-16M (1-2% of original)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank: higher = more capacity but slower. 8-32 is typical
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ],
    lora_alpha=16,  # Scaling factor. Usually set equal to r
    lora_dropout=0,  # Dropout in LoRA layers. 0 works well with small models
    bias="none",     # Don't train bias terms
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
    random_state=42,
    use_rslora=False,  # Rank-stabilized LoRA (experimental)
    loftq_config=None,  # LoftQ quantization (advanced)
)

# Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
# We'll convert this to chat format that Llama expects

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:5000]")  # Use 5k for quick experiment

# Unsloth provides chat templates for proper formatting
# This ensures model learns the correct instruction format

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    """Convert Alpaca format to model's expected format"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Combine instruction and input if input exists
        if input_text:
            instruction = instruction + "\n" + input_text
            
        text = alpaca_prompt.format(instruction, "", output)
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

training_args = TrainingArguments(
    output_dir="./outputs_sft_basic",
    
    # Training duration
    num_train_epochs=1,  # 1 epoch is often enough for small datasets!
    max_steps=-1,  # -1 means train for full epochs
    
    # Batch sizes (key for memory usage!)
    per_device_train_batch_size=2,  # Actual batch per GPU
    gradient_accumulation_steps=4,  # Effective batch = 2*4 = 8
    
    # Learning rate (crucial hyperparameter!)
    learning_rate=2e-4,  # 2e-4 works well for LoRA. Original model uses 5e-5
    warmup_steps=5,  # Gradually increase LR at start
    
    # Optimization
    optim="adamw_8bit",  # 8-bit Adam saves memory
    weight_decay=0.01,  # L2 regularization
    lr_scheduler_type="linear",  # Linear decay to 0
    
    # Logging
    logging_steps=10,
    save_steps=100,
    
    # Memory optimization
    fp16=not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 unavailable
    bf16=torch.cuda.is_bf16_supported(),  # BF16 is better if available
    gradient_checkpointing=True,  # Reduces memory by recomputing activations
    
    # Other
    seed=42,
    report_to="none",  # Change to "wandb" for experiment tracking
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Column name with training text
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,  # Pack multiple examples into one sequence (advanced)
)

print("🚀 Starting training...")
trainer.train()

# Save LoRA adapters only (tiny - just a few MB!)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# OR save merged model (LoRA weights merged into base model)
# model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

print("✅ Training complete! Model saved to ./lora_model")

# ============================================================================
# STEP 7: Quick Test
# ============================================================================

# Enable fast inference mode
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    alpaca_prompt.format(
        "Tell me about machine learning",  # instruction
        "",  # input
        "",  # output (leave empty for generation)
    ),
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print("\n" + "="*50)
print("Test Generation:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*50)

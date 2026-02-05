"""
DPO (Direct Preference Optimization) with Unsloth

This is the RL phase! After SFT, we align the model to human preferences.
DPO is simpler and more stable than PPO - no separate reward model needed.

Dataset: Preference pairs (chosen vs rejected responses)
Training time: 1-2 hours on T4 GPU

Key concept: Model learns to prefer "chosen" responses over "rejected" ones
while staying close to the SFT model (reference model).
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import PatchDPOTrainer
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import TrainingArguments

# First, patch DPO trainer for 2x faster training
PatchDPOTrainer()

# ============================================================================
# STEP 1: Load SFT Model (becomes our reference model)
# ============================================================================

# Load your SFT checkpoint or use a pre-trained instruct model
model_name = "unsloth/Llama-3.2-1B-Instruct"  # Or path to your SFT model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters (we'll train these for the policy model)
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
# STEP 2: Preference Dataset
# ============================================================================

# DPO dataset format:
# {
#     "prompt": "Question or instruction",
#     "chosen": "Better response (preferred by humans)",
#     "rejected": "Worse response (not preferred)"
# }

# Great datasets to start with:
# - "Anthropic/hh-rlhf" - Helpful & Harmless conversations
# - "Intel/orca_dpo_pairs" - High quality preference pairs
# - "argilla/ultrafeedback-binarized-preferences-cleaned" - Large scale

dataset = load_dataset(
    "Intel/orca_dpo_pairs",
    split="train[:1000]"  # Use 1k samples for quick experiment
)

# Check dataset structure
print("Dataset columns:", dataset.column_names)
print("\nExample:")
print("Prompt:", dataset[0]["prompt"][:200])
print("Chosen:", dataset[0]["chosen"][:200])
print("Rejected:", dataset[0]["rejected"][:200])

# ============================================================================
# STEP 3: Format Dataset for DPO
# ============================================================================

def format_dpo(example):
    """
    DPOTrainer expects specific format:
    - prompt: string or list of messages
    - chosen: list of messages  
    - rejected: list of messages
    """
    # If your dataset has different column names, adjust here
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_dpo)

# ============================================================================
# STEP 4: DPO Training Configuration
# ============================================================================

dpo_config = DPOConfig(
    output_dir="./outputs_dpo",
    
    # Training duration
    num_train_epochs=1,
    
    # Batch sizes
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 8
    
    # Learning rate (typically lower than SFT)
    learning_rate=5e-5,  # 5e-5 to 1e-4 works well for DPO
    warmup_steps=5,
    
    # DPO specific parameters
    beta=0.1,  # KL divergence coefficient (0.1-0.5 typical)
               # Higher = stay closer to reference model
               # Lower = allow more deviation
    
    # Optimization
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # Cosine often works better for DPO
    
    # Logging
    logging_steps=10,
    save_steps=100,
    
    # Memory
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    gradient_checkpointing=True,
    
    seed=42,
    report_to="none",
    
    # DPO specific
    max_prompt_length=512,  # Trim long prompts
    max_length=1024,  # Max total length (prompt + response)
)

# ============================================================================
# STEP 5: DPO Trainer
# ============================================================================

# The magic of DPO: we use the SAME model for policy and reference!
# Internally, DPOTrainer:
# 1. Creates a frozen copy (reference model) from initial weights
# 2. Trains LoRA adapters (policy model)
# 3. Compares their outputs to compute DPO loss

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # None = automatically create reference from model
    args=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # You can also add eval_dataset for validation
)

print("🚀 Starting DPO training...")
print(f"Training on {len(dataset)} preference pairs")
print(f"Beta (KL coefficient): {dpo_config.beta}")

trainer.train()

# ============================================================================
# STEP 6: Save Aligned Model
# ============================================================================

model.save_pretrained("lora_dpo_model")
tokenizer.save_pretrained("lora_dpo_model")

print("✅ DPO training complete!")

# ============================================================================
# STEP 7: Test Alignment
# ============================================================================

FastLanguageModel.for_inference(model)

# Test on a prompt from the dataset
test_prompt = dataset[0]["prompt"]

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("\n" + "="*70)
print("Testing aligned model:")
print(f"Prompt: {test_prompt[:200]}...")
print("-"*70)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
print("="*70)

# ============================================================================
# Understanding DPO Loss
# ============================================================================

"""
DPO Loss = -log(sigmoid(β * (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)))))

Where:
- π_θ = policy model (with LoRA adapters being trained)
- π_ref = reference model (frozen SFT model)
- y_w = chosen (winning) response
- y_l = rejected (losing) response
- β = controls how much we allow deviation from reference

The loss encourages:
1. Policy to assign higher probability to chosen vs rejected
2. But not drift too far from reference (controlled by β)

Intuition:
- If policy prefers chosen MORE than reference does → good! Low loss
- If policy prefers rejected more than reference → bad! High loss
- If policy drifts too far from reference → penalized by KL term
"""

# ============================================================================
# Pro Tips
# ============================================================================

"""
1. **Beta tuning**: Start with 0.1
   - Too high (>0.5): Model barely changes
   - Too low (<0.05): Model might forget SFT behavior
   
2. **Dataset quality matters**: 
   - Need clear preference signal (chosen >> rejected)
   - Ambiguous pairs hurt performance
   
3. **SFT first**: Always train SFT before DPO
   - DPO assumes model already knows how to respond
   - DPO just teaches preferences, not capabilities
   
4. **Evaluation**: Use human eval or GPT-4 as judge
   - Compare responses before/after DPO
   - Check for helpfulness, safety, honesty
   
5. **Iteration**: Can do multiple rounds
   - SFT → DPO1 → DPO2 with different datasets
   - Each round refines different aspects
"""

# distillation with LoRA adapters on student
# we can perform distillation with LoRA where instead of training all student weights
# we only train a few lora adapters 
# benefits: even faster training (few parameters), less memory (only adapter gradients)
# can swap adapters easily (multiple students from one base)

# the key idea: student base model stays frozen, only LoRA adapters learn to mimic teacher
# this is extreme efficiency: training ~10M params instead of 1B to compress 7B teacher

# distillation with LoRA adapters on student
# we can performr distillation with LoRA obviosuly in this, where 
# instead of trainign all student weeights, only train a few lora adapters 
# so this turns in as even faster training (few parameters)
# less memory (only adapter gradients)
# can swap adapters easily (multiple students from one base )


# we use inject lora from our previous files
#
# the differnece between performing LoRA on SFT vs RL vs distillation is htat :

# SFT : LoRA adapts to how to recognize instruction format, how to give helpful direct answers vs rambling and conversational pattersn
# the main objective is to adapt the style of responses 
# RL : LoRA adapts subjective qualities (toe, helpfullness, safety in human cognitive sense), don't want to unlearn instruction following from SFT 
# the main objective is to align with human conitive bias 
# LoRA adapts how to approximate teacher's decision making, which features teacher considers important, the confidence patterns 
# the main objective is to compress large teacher model into small student model 


import sys
sys.path.append('..')
from lora.inject_lora import inject_lora_to_model, save_lora_adapters
from distill_trainer import DistillationTrainer


def create_lora_student(base_student_model, rank=4, alpha=16, target_modules=['wq', 'wk', 'wv', 'wo']):
    
    # for distillation we use smaller rank (4-8) since:
    # - student is already small (1B vs 7B teacher)
    # - only needs to adapt to mimic teacher, not learn from scratch
    # - fewer params = faster training
    
    # inject_lora_to_model automatically:
    # 1. wraps target layers with LoRALayer
    # 2. freezes original weights (param.requires_grad = False)
    # 3. creates trainable A and B matrices
    lora_student = inject_lora_to_model(
        model=base_student_model,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules
    )
    
    print("LoRA injection complete. Student base frozen, only adapters trainable.")
    return lora_student


def train_distillation_with_lora(teacher, student, config, dataset, lora_rank=4, lora_alpha=16):
    # distillation training where student uses LoRA for extreme efficiency
    
    # workflow:
    # 1. inject LoRA into student (base frozen, adapters trainable)
    # 2. use regular distillation trainer (teacher frozen, student adapters train)
    # 3. student learns to match teacher's soft targets via tiny adapters
    # 4. save only the adapters (few MB instead of full model GB)
    
    # typical params:
    # - lora_rank: 4-8 for distillation (smaller than SFT since student is small)
    # - lora_alpha: 16 (standard scaling factor)
    # create LoRA student
    lora_student = create_lora_student(
        base_student_model=student,
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=['wq', 'wk', 'wv', 'wo']
    )
    
    # use regular distillation trainer
    # it works with LoRA automatically since optimizer only sees trainable params
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=lora_student,
        config=config,
        train_dataset=dataset
    )
    
    # train (only LoRA adapters get updated)
    print("\nStarting distillation with LoRA...")
    trainer.train()
    
    # save only the adapters (not full model)
    save_lora_adapters(lora_student, "distill_checkpoints/student_lora_adapters.pth")
    
    print("\nDistillation complete!")
    print("Saved LoRA adapters separately for easy distribution.")
    
    return lora_student

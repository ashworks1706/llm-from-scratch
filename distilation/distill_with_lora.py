# distillation with LoRA adapters on student
# we can performr distillation with LoRA obviosuly in this, where 
# instead of trainign all studnet weeights, only train a few lora adapters 
# so this turns in as even faster training (few parameters)
# less memory (only adapter gradients)
# can swap adapters easily (multiple students from one base )


# we use inject lora from our previous files 


from lora.inject_lora import inject_lora_to_model
from distill_trainer import DistillationTrainer

def create_lora_student(base_student_model, lora_config):
    # inject LoRA to student model 
    # freeze base weights 
    # only keep LoRA adapeters trainable 
    pass 


def train_distillation_with_lora(teacher, student, conifg, dataset):
    # create LoRA student
    # use regular distillation trainer p

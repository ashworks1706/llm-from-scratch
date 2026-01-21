# main training logic
# distillation is a process of training a student model from teacher model's responses
# quite specifically, teacher's model respones are treated as soft targets 
# the loss is not just teacher's responses, tho we do have KL divergence for that but we also have hard loss too 
# so as to ensure if the teacher is wrong on a model response, the student llm doesnt drift too far 
# loss = hardloss + softloss 
# this is a bit related to RLHF where-

# they're similar in structure:

# DPO/PPO:
# Policy model (being trained)
# Reference model (frozen SFT model)
# KL divergence: keeps policy close to reference
# Goal: Learn from preferences while staying grounded
# Distillation:
# Student model (being trained)
# Teacher model (frozen large model)
# KL divergence: keeps student close to teacher
# Goal: Learn from teacher while staying accurate

# Both use KL divergence to prevent drifting!
# Key difference:
# - RL: Learn from human preferences (chosen vs rejected)
# - Distillation: Learn from teacher's knowledge (soft probabilities)



class DistillationTrainer:
    def __init__(self, teacher_model, student_model, config):
        self.teacher = teacher_model # frozen (not supposed to be trained)
        self.student = student_model # training 
        self.temperature = config.temperature
        self.alpha = config.alpha # balance soft vs hard loss

    # what happens in one trainign step?
    # normal training (without teacher):
    # forwrad pass -> get logits 
    # compute loss (crossentropy with true labels)
    # backward pass -> update student weights 

    # Distillation training (with teacher):
    # forward pass throguh BOTH models 
    # compute two losses :
    # a) soft loss : math teacher's distribution 
    # b) hard loss : match true labels 
    # combine losses 
    # backward pass -> update student only (teacher frozen)

    def train_step(self, input_ids, labels):
        # get teacehr predictions (no grad)
        # get student predictions (with grad)
        # compute soft loss (KL divergence)
        # compute hard loss (cross entropy)
        # combine and backprop



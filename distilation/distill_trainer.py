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

# LoRA with distillation:
# - We can freeze student's base weights
# - Only train LoRA adapters on student
# - Makes distillation even more efficient!

#Input → Teacher (frozen) → Teacher Logits → Soft Targets (T=5)
#       ↓                                              ↓
#       └──→ Student (training) → Student Logits → Predictions (T=5)
#                                       ↓
#                                 KL Divergence = Soft Loss
#                                       ↓
#                                 + Hard Loss (with true labels)
#                                      ↓
#                                 Total Loss → Backprop → Update Student Only

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
        
        # we use temperatures to make distributioins softer high temperature -> more creative response -> more SD 
        # we divide the logits by temeprature so the softmax logits dont overfit or are too confident
         
        # compute soft loss (KL divergence)
        
        # KL divergence measures difference between two probaiblity distributions 
        # where KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        # - P = teacher's distribution (what we want student to match)
        # - Q = student's distribution (what student currently produces)
        # we just use torch built in kl_dv function for this, 
        # we need log_softmax for student not for teacher because kldivergence formula needs log probabilities for Q 

        # since loss depednds on gradients, softmax derivative scales with 1/T so loss scales with 1/T², we multiply by T² to normalize
        # without this, soft loss would be too small compared to hard loss 
        # softmax(x) = exp(x) / sum(exp(x))
        # log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))

        # compute hard loss (cross entropy)
        # we use student's logits without temperature on cross entropy loss formula since we want accurate predicts on true labels 
        
        # combine and backprop
        # we combine totalloss = alpha * soft_loss + (1-alpha) * hardloss 
        # we use both so that soft loss learns teacher's reasoning and nuances and hardloss ensures accuracy on true labels i.e correct teacher's mistakes

        
        return 



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
import os
import torch 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, config, train_dataset):
        self.teacher = teacher_model # frozen (not supposed to be trained)
        self.student = student_model # training 
        self.temperature = config.temperature
        self.alpha = config.alpha # balance soft vs hard loss
        # add optimizer for student 
        self.optimizer = optim.AdamW(student_model.parameters(), lr=config.learning_rate)

        self.data_loader = DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True)

        self.epochs = config.num_epochs

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad=False

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

        # move tensors to GPU
        device = next(self.student.parameters()).device 
        input_ids = input_ids.to(device)
        labels = labels.to(device)
       
        # get teacehr predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        
        # get student predictions (with grad)
        student_logits = self.student(input_ids)
        
        # we use temperatures to make distributioins softer high temperature -> more creative response -> more SD 
        # we divide the logits by temeprature so the softmax logits dont overfit or are too confident
        T = self.temperature 
        teacher_soft = teacher_logits / T 
        student_soft = student_logits / T 

         
        # compute soft loss (KL divergence)
        teacher_probs = F.softmax(teacher_soft, dim=-1)
        student_log_probs = F.log_softmax(student_soft, dim=-1)

        # KL divergence measures difference between two probaiblity distributions 
        # where KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        # - P = teacher's distribution (what we want student to match)
        # - Q = student's distribution (what student currently produces)
        # Each term P(i) * log(P(i)/Q(i)) means if teacher assings high probability P(i) to token i, but student assings 
        # low probability Q(i), this term becomes LARGE (big penalty)
        # KL penalizes when student disagrees on teacher's high confience predictions 
        # we just use torch built in kl_dv function for this, 
        # for ex, if teacher has a probability distribution that says token 0 is very likely and student says 
        # token 0 is somewhat likely, KL measures how much Q differ from P 
        # KL =0 means P and Q are identical, >0 means P and Q differ (student needs to learn)
        # Larger the KL bigger the difference 
        soft_loss = F.kl_div(student_log_probs,teacher_probs,reduction='batchmean')

        # we need log_softmax for student not for teacher because kldivergence formula needs log probabilities for Q 
        # we need softmax because the logits have raw scores, which can be negative, cannot always sum up to 1
        # so inorder to fit these tokens in a range and make sure that its summing up to 1 
        # we use softmax, we exponentiate all integers to make all positive, divide by sum, then it fits the range
        # now we use log softmax, because softmax has numerical issues, taking exponents can be quadratically crazy 
        # so we instead take log than just doing exponential quadratics sincei n log prababilities negative numbers, closer 0 are more likely 
        # since loss depednds on gradients, softmax derivative scales with 1/T so loss scales with 1/T², we multiply by T² to normalize
        # without this, soft loss would be too small compared to hard loss 
        # softmax(x) = exp(x) / sum(exp(x))
        # log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
        soft_loss = soft_loss * (T**2)

        # compute hard loss (cross entropy)
        # we use student's logits without temperature on cross entropy loss formula since we want accurate predicts on true labels 
        # why not MSE? KL Divergence respects that probabilities must sum to 1
        # It measures "information divergence" not just numeric difference
        # the alpha in totla loss is 0.9 usually, 90% trust teacher, 10% trust labels 
        # why? because teacher is usually right, soft targets are richer than hard labels,
        # hard loss is just a saftey net for when teacher fails 
        vocab_size = student_logits.size(-1)

        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size ), # reshape to (batch*seq, vocab)
            labels.view(-1) # reshape to (batch*seq, )
        )
        
        # combine and backprop
        # we combine totalloss = alpha * soft_loss + (1-alpha) * hardloss 
        # we use both so that soft loss learns teacher's reasoning and nuances and hardloss ensures accuracy on true labels i.e correct teacher's mistakes
        total_loss = self.alpha * soft_loss + (1-self.alpha) * hard_loss 

        total_loss.backward()

        
        return total_loss.item(), soft_loss.item(), hard_loss.item()


    def train_epoch(self, epoch):
        # set student to training mode 
        self.student.train() 
        average_loss=0
        for batch_idx, (input_ids, labels) in enumerate(self.data_loader):
            # zero gradients 
            self.optimizer.zero_grad()
            # use train step 
            total_loss, soft_loss, hard_loss = self.train_step(input_ids, labels)
            # optimizer step 
            self.optimizer.step()

            average_loss+=total_loss 

            print(f"Epoch {epoch}")
            print(f"Batch {batch_idx}")
            print(f"Soft loss {soft_loss}")
            print(f"Hard loss {hard_loss}")

        average_loss = average_loss / len(self.data_loader) # divide by num batches
        return average_loss

    def train(self):
        os.makedirs("distill_checkpoints", exist_ok=True)
        for epoch in range(self.epochs):
            average_loss = self.train_epoch(epoch)
            print(f"Average loss : {average_loss}")
                
            checkpoint = {
                'epoch': epoch,
                'student_model_state_dict': self.student.state_dict(),
                'teacher_model_state_dict': self.teacher.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': average_loss,
            }

            torch.save(checkpoint, f"distill_checkpoints/student_epoch_{epoch+1}.pth")

        return 























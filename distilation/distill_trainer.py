# main training logic


class DistillationTrainer:
    def __init__(self, teacher_model, student_model, config):
        self.teacher = teacher_model # frozen (not supposed to be trained)
        self.student = student_model # training 
        self.temperature = config.temperature
        self.alpha = config.alpha # balance soft vs hard loss

    def train_step(self, input_ids, labels):
        # get teacehr predictions (no grad)
        # get student predictions (with grad)
        # compute soft loss (KL divergence)
        # compute hard loss (cross entropy)
        # combine and backprop



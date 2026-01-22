# temperature KL divergence


def temperature_softmax(logits, temperature):
    # apply temperature scaling
    return softmax(logits / temperature)

def kl_divergence_loss(student_probs, teacher_probs):
    # compute kl divergence between distributions
    return kl_div(student_probs, teacher_probs)

def count_parameters(model):
    # count total parameters
    pass 

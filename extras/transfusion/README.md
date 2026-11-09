transfusion is one transformer that does both next token prediction for text and diffusion for images. models like bagel and hunyuan image 3 use this idea to understand and generate in one model.

text tokens use causal attention and a cross entropy loss, image patches use bidirectional attention inside the image and a flow matching loss. it is trained on captioned mnist or cifar.

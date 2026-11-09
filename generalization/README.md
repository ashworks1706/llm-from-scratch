generalization covers how models can go beyond imitating the next token, by thinking longer, refining their answers or learning a model of the world.

ebt explores energy based transformers where predictions are refined by minimizing an energy score instead of only doing one feed forward token prediction. the notebook uses gsm8k style reasoning data and inspects energy traces, btr style variants and failure cases.

vjepa learns world models by predicting representations instead of pixels, and ends with an action conditioned world model that plans. videodiffusion is the generative side of world models, generating frames one after another.

testtimecompute covers spending more compute at inference to get better answers, with voting, verifiers and search.

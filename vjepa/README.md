jepa is a joint embedding predictive architecture. instead of reconstructing missing pixels or predicting the next token, it predicts the representation of the missing part. this means the model does not have to spend capacity on details that cannot be predicted, like the exact texture of leaves, and can focus on what is predictable about the scene.

there are three networks. the context encoder sees only the visible patches. the target encoder sees the full input and is an exponential moving average of the context encoder, so it gets no gradients. the predictor takes the context features plus mask tokens for the hidden positions and tries to predict what the target encoder produced there. the loss is l1 between the predicted and target features.

v jepa does this on video. frames are split into tubelets that cover a small patch over a few frames, and large blocks are masked across space and time. to predict what is hidden the model has to understand how things move. the encoder is then frozen and checked with a small attentive probe on action recognition, and v jepa 2 also uses it as a world model for planning.

the folder starts with image jepa on small images before moving to video.

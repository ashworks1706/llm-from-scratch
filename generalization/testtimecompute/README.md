test time compute means spending more compute at inference to get better answers instead of training a bigger model.

self consistency samples many answers and takes a vote. best of n uses a verifier to pick the best one. beam search and dvts search over reasoning steps, and compute optimal allocation decides how much to spend on each question.

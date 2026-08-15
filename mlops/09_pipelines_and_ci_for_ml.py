# a pipeline wires the earlier steps into one repeatable flow instead of manual runs. train,
# evaluate, register, deploy become a dag you can trigger and rerun, and each step's output
# feeds the next. ci for ml adds a gate: a deploy only happens if the eval scores from lesson
# 04/05 clear a bar, so a worse model physically can't ship. this is what makes model updates
# routine instead of scary, because the process, not a person's memory, enforces quality.

# LEARNING OBJECTIVES:
# - Express train -> eval -> register -> deploy as an ordered pipeline, not ad hoc scripts
# - Pass artifacts between stages (a checkpoint out of train becomes input to eval)
# - Gate deployment on an eval threshold so a regression blocks the release automatically
# - Make a pipeline rerunnable and idempotent so the same inputs give the same result
# - Understand ci/cd for models: what a git push should trigger and what it should block on
# - Tie stages to mlflow runs and dvc versions so every pipeline run is fully traceable

import subprocess

import mlflow


def main():
    # step 1: define the stages as functions: train, evaluate, register, deploy
    # step 2: chain them so evaluate consumes train's checkpoint and register consumes the score
    # step 3: add a gate: if eval metric < threshold, stop before register/deploy
    # step 4: log each stage as an mlflow run so the whole pipeline run is inspectable
    # step 5: sketch what a ci config would run on push (lint -> train small -> eval -> gate)
    # step 6: rerun with the same inputs and confirm you get the same decision (idempotent)
    pass


if __name__ == "__main__":
    main()

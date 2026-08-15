# experiment tracking is where mlops starts. a training run is not just its final
# loss, it's the params you chose, the metrics over time, and the artifacts it produced.
# if you don't log all of that you can't tell which change helped or reproduce a good
# result a week later. mlflow and wandb both do this, we look at both here.

# LEARNING OBJECTIVES:
# - Log params, metrics and artifacts of a run instead of trusting memory or a notebook cell
# - Understand a "run" as the unit: one training attempt with its full config and outputs
# - Compare multiple runs side by side to see which hyperparameters actually mattered
# - Make a result reproducible by capturing config, code version and data version together
# - See the difference between mlflow (local-first, open source) and wandb (cloud-first)
# - Log to a local mlflow tracking store so nothing leaves the machine

import mlflow
import wandb


def main():
    # step 1: start an mlflow run and log the params you're testing (lr, batch size, seed)
    # step 2: inside a fake training loop, log a metric per step so you get a curve, not one point
    # step 3: log an artifact (a plot, a config file, a small checkpoint) attached to the run
    # step 4: open the mlflow ui (mlflow ui) and compare two runs with different params
    # step 5: do the same with wandb.init / wandb.log and note what the cloud dashboard adds
    # step 6: write down what you'd need to log to fully reproduce the run months from now
    pass


if __name__ == "__main__":
    main()

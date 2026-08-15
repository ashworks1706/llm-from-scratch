# a checkpoint on disk is a loose file with no identity. a model registry gives it a
# name, a version number and a stage (staging, production), so you can say "promote
# model X version 3 to production" instead of copying a .pt file around. packaging goes
# one step further and bundles the weights with the code that runs them, so the artifact
# you evaluated is byte for byte the thing you deploy, with no "works on my machine" gap.

# LEARNING OBJECTIVES:
# - Register a model in mlflow's registry and give it versions instead of filenames
# - Move a version through stages (none -> staging -> production) and see why stages matter
# - Load a model back by name+stage, not by a hardcoded path, so callers don't pin files
# - Package a model as a self contained service artifact with bentoml (weights + code + deps)
# - Understand the difference: registry tracks/versions, packaging makes it runnable elsewhere
# - See how this hands off to lesson 07 (serving) and the inference folder's engines

import mlflow
import bentoml


def main():
    # step 1: log a trained model with mlflow.<flavor>.log_model so it becomes an artifact
    # step 2: register it, producing model name + version 1 in the registry
    # step 3: transition version 1 to staging, then production, and query by stage
    # step 4: load it with models:/<name>/Production and confirm callers never touch a path
    # step 5: save the same model into the bentoml store and inspect the packaged bento
    # step 6: note what a bento bundles that a bare checkpoint doesn't (signature, deps, runner)
    pass


if __name__ == "__main__":
    main()

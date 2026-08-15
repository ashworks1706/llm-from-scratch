# monitoring and drift is the long tail after launch, where most real failures actually live.
# a model is trained on one distribution but production traffic slowly moves away from it, so
# quality degrades without any code change. on top of that, costs creep and bad outputs slip
# through. this lesson covers detecting drift, putting guardrails on inputs and outputs, closing
# the feedback loop so production data improves the next model, and watching cost so a serving
# bill doesn't surprise you. this is the step that connects the end of the lifecycle back to
# the start.

# LEARNING OBJECTIVES:
# - Define drift: input (data) drift vs output (concept) drift, and why each hurts quality
# - Detect drift by comparing a live distribution against a training-time baseline
# - Add guardrails that validate or block unsafe/malformed inputs and outputs
# - Build a feedback loop: capture production cases and route them into evals and training data
# - Monitor cost and latency as first-class metrics, not afterthoughts
# - Close the loop: how monitoring findings feed lesson 09's pipeline for the next model

import numpy as np


def main():
    # step 1: take a training-time feature/prompt distribution as the baseline to compare against
    # step 2: simulate a live stream that slowly shifts and measure the distance from baseline
    # step 3: set a drift threshold that would trigger an alert or a retrain
    # step 4: add a simple input/output guardrail and show it blocking a bad case
    # step 5: capture flagged production cases and turn them into new eval examples (lesson 05)
    # step 6: track cost/latency over the run and connect a drift trigger back to the pipeline
    pass


if __name__ == "__main__":
    main()

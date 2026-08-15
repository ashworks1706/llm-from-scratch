# benchmarking is how you earn the claim that a model is good, on questions you didn't
# get to pick. standardized suites like mmlu, hellaswag and arc ask the same questions of
# every model, so a score is comparable across the field instead of being marketing. the
# lm-evaluation-harness is the de facto standard runner for these; lighteval is hugging
# face's take. the real skill is reading the number honestly: what it measures, what it
# doesn't, and how few-shot count or prompt format can quietly move it.

# LEARNING OBJECTIVES:
# - Run a small model against a standardized benchmark task with the eval harness
# - Understand few-shot vs zero-shot and how the number of examples shifts the score
# - Read a result as accuracy on a fixed distribution, not a general "smartness" score
# - Know which benchmarks probe what (knowledge vs reasoning vs commonsense)
# - See why prompt formatting and tokenization choices make scores hard to compare naively
# - Log benchmark results back into experiment tracking (lesson 01) so they live with the run

import lm_eval


def main():
    # step 1: pick a small local model and one task (e.g. hellaswag) to keep the run fast
    # step 2: run the harness zero-shot, then 5-shot, and compare the two accuracies
    # step 3: inspect a few individual examples to see exactly what the model was asked
    # step 4: try a second task and note how "good" is task-dependent, not one number
    # step 5: repeat the same eval with lighteval and compare setup and output format
    # step 6: log the scores as metrics on an mlflow run so a checkpoint carries its evals
    pass


if __name__ == "__main__":
    main()

# fixed benchmarks can't score the tasks people actually ship: open-ended answers,
# summaries, rag responses. there's no single right string to match against. so we use
# task-specific metrics and llm-as-judge, where a strong model grades outputs against a
# rubric. this is powerful and also where evals lie: judges have biases, metrics can be
# gamed, and a single run is noisy. the defensive move is prompt regression tests, a fixed
# set of inputs with expected behavior that must keep passing as you change the system.

# LEARNING OBJECTIVES:
# - Evaluate an open-ended / rag output where exact-match scoring is impossible
# - Use ragas metrics (faithfulness, answer relevancy, context precision) on a rag pipeline
# - Set up an llm-as-judge and reason about its biases (position, verbosity, self-preference)
# - Build a prompt regression test: fixed inputs + expected behavior that must not break
# - Understand why one eval run is noisy and when you need multiple samples or seeds
# - Decide when a cheap heuristic metric is enough vs when you must pay for a judge model

import ragas
import deepeval


def main():
    # step 1: take a small rag pipeline (retrieved context + question + answer) as the target
    # step 2: score it with ragas faithfulness and answer relevancy, read what each penalizes
    # step 3: write an llm-as-judge prompt with an explicit rubric and grade a few answers
    # step 4: deliberately trigger a judge bias (make one answer longer) and watch the score move
    # step 5: assemble a small regression suite with deepeval-style asserts over fixed prompts
    # step 6: run the suite, break the prompt on purpose, and confirm the suite catches it
    pass


if __name__ == "__main__":
    main()

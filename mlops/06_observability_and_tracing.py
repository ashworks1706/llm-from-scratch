# observability is debugging once the model is serving real traffic. one user request can
# fan out into many steps: retrieval, several model calls, tool calls, post-processing. when
# the final answer is wrong, a single log line won't tell you which step failed. a trace
# captures the whole tree with inputs, outputs, token counts, latency and cost per step, so
# you can see why a behavior happened, not just that it did. monitoring says up/fast/cheap;
# observability says why.

# LEARNING OBJECTIVES:
# - Distinguish monitoring (is it up, fast, within budget) from observability (why did this happen)
# - Instrument a multi-step llm call as a trace with nested spans
# - Capture per-span tokens, latency and cost and find the expensive step
# - Attach the input and output to each span so a bad answer is debuggable after the fact
# - Use langfuse and arize phoenix and see what a tracing ui gives over raw logs
# - Connect traces back to evals (lesson 05) so production failures become new test cases

import langfuse
import phoenix


def main():
    # step 1: take a small multi-step pipeline (retrieve -> generate -> post-process) as target
    # step 2: wrap each step in a span so the whole request becomes one nested trace
    # step 3: record tokens, latency and estimated cost on each span
    # step 4: open the langfuse ui, find the slowest and most expensive step in the trace
    # step 5: do the same locally with arize phoenix and compare the two dashboards
    # step 6: take a traced failure and turn its inputs into a regression test from lesson 05
    pass


if __name__ == "__main__":
    main()

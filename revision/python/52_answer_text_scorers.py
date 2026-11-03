# some scorers read the answer: grounding (right status, cites the expected source, mentions the terms),
# refusal (declines exactly when the case says so), voice (never names the machinery).
# these are deterministic string checks with known blind spots, so they are a regression gate, not truth.
# a citation matches by source key prefix, falling back to the title when the key is empty.
#
# from sparky: apps/evals/suites/grounding.py, refusal.py (the _DECLINE regex), voice.py (INTERNALS).
# turn shape here: {"status": str, "text": str, "citations": [{"key": str, "title": str}]}.
#
# explain: how can a refusal regex pass a bad answer, and fail a good one? what would an llm judge add and cost?
# explain: why are problems collected into a list and joined, rather than returning on the first problem?

import re

DECLINE = re.compile(r"(can(?:no|')t|cannot|unable|don't have|do not have|outside)", re.IGNORECASE)
INTERNALS = ("search_live", "search_knowledge", "run_sandbox", "tool call", "source key")


def score_grounding(last: dict, source_key: str | None, mentions: list[str]) -> tuple[bool, str]:
    # problems = [] ; status != "answered" adds one; a source_key with no citation whose (key or title)
    # startswith it adds one that names the first two cited keys; each mention missing from text.lower() adds one.
    # pass when problems is empty, detail is "; ".join(problems) or a short success phrase.
    pass


def score_refusal(text: str, expect_refuse: bool) -> tuple[bool, str]:
    # declined = DECLINE matches anywhere. expect_refuse -> pass iff declined; else pass iff not declined.
    pass


def score_voice(text: str, not_mentions: list[str]) -> tuple[bool, str]:
    # forbidden = INTERNALS + not_mentions, case-insensitive substring match on the text.
    # detail lists every forbidden word found.
    pass


if __name__ == '__main__':
    # asserts to write: citation matched by title when key is empty; answered status but missing mention;
    # "I can't find that" counts as a decline; the word "outside" in a normal answer is a false positive (document it).
    pass

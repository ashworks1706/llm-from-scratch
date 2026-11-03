// a streaming answer is not shown token by token. the draft holds back text until a sentence or line ends
// (". " "! " "? " ": " or a newline), or until block_chars have piled up without one, then cuts at the last space.
// only text past what was already shown counts, and nothing is released if the draft equals the last release.
// a withheld draft releases nothing for the rest of the call: used when the guardrail refuses a block.
// when the call ends with tool calls or an error, the shown draft is withdrawn with a clear event.
//
// from sparky: apps/engine/src/runtime/harness/agent/call/draft.rs (Draft::next_answer, block_end,
// ends_block, withhold), call/relay.rs (the withdraw rule) and config agent.stream_block_chars. skip reasoning and the think-tag handling here.
//
// explain: why release at sentence ends instead of on a timer?
// explain: why must the draft be withdrawable (AnswerDraftCleared) when tool calls arrive after text?

fn block_end(text: &str) -> Option<usize> {
    // byte offset just past the LAST sentence end or newline in text, or None.
    // a sentence end is one of . ! ? : followed by whitespace. use char_indices, not byte arithmetic on chars.
    todo!()
}

struct Draft { text: String, shown: String, block_chars: usize, withheld: bool }

impl Draft {
    fn push(&mut self, piece: &str) -> Option<String> {
        // append piece, then return the answer up to the end of its last block if it is new.
        // 1 withheld -> None.
        // 2 fresh = text after what was shown. cut = block_end(fresh), else if fresh has >= block_chars
        //   chars cut at its last whitespace (or the whole of fresh when it has none), else None.
        // 3 candidate = text[..shown_len + cut] trimmed at the end; None if empty or equal to shown.
        // 4 record it as shown and return it. each call is O(len(fresh)).
        todo!()
    }

    fn withhold(&mut self) {
        todo!()
    }
}

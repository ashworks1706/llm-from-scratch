// server-sent events are text frames separated by a blank line. a frame has "event: name" and one or
// more "data: ..." lines; multiple data lines join with a newline. network chunks do not align with frames,
// so the reader keeps a buffer: take every complete frame, leave the partial tail for the next chunk.
// the engine sends progress frames, then one answer or error frame, then done. the client acts on the first three only.
//
// from sparky: apps/discord/src/engine/sse.rs (drain_frames, take_complete, decode) and the writer in
// apps/engine/src/routes/chat.rs (stream: progress frames, then answer or error, then done).
//
// explain: what breaks if you split each chunk on blank lines without a buffer?
// explain: why does the engine send an error as a normal frame instead of closing the connection?

fn drain_frames(buf: &mut String) -> Vec<(String, String)> {
    // pull every complete frame (ends with "\n\n") out of buf and return (event_name, data).
    // leave a trailing partial frame in buf untouched.
    // a line "event:x" or "event: x" both work (trim). data lines starting "data:" join with "\n".
    // a frame with no data lines yields nothing. missing event name gives an empty name.
    // complexity target: linear in the buffer; do not rescan drained text.
    todo!()
}

enum Update { Progress(String), Answer(String), Failed(String) }

fn decode(name: &str, data: &str) -> Option<Update> {
    // progress, answer, error map to the matching variant; any other name (including done) is None.
    todo!()
}

fn main() {
    // tests: feed the same stream split at every possible byte offset and assert the frames are identical.
    // a utf-8 char split across chunks is the hard case: sparky keeps bytes in a Vec<u8> and decodes only complete frames.
}

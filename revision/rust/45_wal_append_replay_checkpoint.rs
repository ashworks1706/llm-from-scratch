// a write-ahead log records each change before it is applied. every entry gets the next
// sequence number. on open, replay entries with seq greater than the last checkpointed seq.
// a checkpoint makes the state durable FIRST (snapshot plus last_seq), and only then rotates
// the log. rotating before the snapshot is durable loses data on a crash.
// the log starts with a version header; replay refuses a version it does not know.
//
// from piramid: apps/engine/database/src/storage/wal/log.rs (log, replay, rotate),
// apps/engine/database/src/collection/checkpoint.rs (checkpoint order) and
// apps/engine/database/src/collection/open.rs (replay_wal) are the reference.
// model the file as a Vec of entries; no real I/O.
// explain: why replay skips seq <= min_seq instead of just trusting the log was rotated?
// explain: what does sync_on_write buy, and what is lost on a crash if it is off?

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
enum Entry { Insert { id: u64, seq: u64 }, Delete { id: u64, seq: u64 }, Checkpoint { seq: u64 } }

struct Wal { version: u32, entries: Vec<Entry>, next_seq: u64 }
struct Durable { snapshot: HashSet<u64>, last_seq: u64 }

const WAL_VERSION: u32 = 1;

impl Wal {
    fn log(&mut self, mut entry: Entry) -> u64 {
        // stamp entry with next_seq, append it, bump next_seq, return the seq it got.
        todo!()
    }

    fn replay(&self, min_seq: u64) -> Result<Vec<Entry>, String> {
        // error if version != WAL_VERSION. otherwise entries with seq > min_seq, in order.
        todo!()
    }
}

fn apply(state: &mut HashSet<u64>, entry: &Entry) {
    // Insert adds the id, Delete removes it, Checkpoint changes nothing.
    todo!()
}

fn checkpoint(wal: &mut Wal, durable: &mut Durable, state: &HashSet<u64>) {
    // 1. write the snapshot and last_seq (= next_seq - 1) into durable.
    // 2. only then clear the entries. next_seq keeps counting up; it never resets.
    todo!()
}

fn open(wal: &Wal, durable: &Durable) -> Result<HashSet<u64>, String> {
    // start from the snapshot, replay entries after durable.last_seq. crash-safe at any point.
    todo!()
}

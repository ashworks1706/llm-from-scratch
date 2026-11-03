// a write ahead log records each change before it is applied, and every entry gets the next sequence number.
// on open, replay only entries with a sequence greater than the last checkpointed one.
// a checkpoint makes the snapshot durable first and only then rotates the log, since rotating earlier loses data on a crash.
// the log starts with a version header, and replay refuses a version it does not know.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
enum Entry { Insert { id: u64, seq: u64 }, Delete { id: u64, seq: u64 }, Checkpoint { seq: u64 } }

struct Wal { version: u32, entries: Vec<Entry>, next_seq: u64 }
struct Durable { snapshot: HashSet<u64>, last_seq: u64 }

const WAL_VERSION: u32 = 1;

impl Wal {
    fn replay(&self, min_seq: u64) -> Result<Vec<Entry>, String> {
        // error if version differs from WAL_VERSION. otherwise return entries with seq above min_seq, in order.
        todo!()
    }
}

fn checkpoint(wal: &mut Wal, durable: &mut Durable, state: &HashSet<u64>) {
    // write the snapshot and last_seq, which is next_seq minus one, into durable. only then clear the entries.
    // next_seq keeps counting up and never resets.
    todo!()
}

fn open(wal: &Wal, durable: &Durable) -> Result<HashSet<u64>, String> {
    // start from the snapshot and apply entries after durable.last_seq. a crash at any point must leave this correct.
    todo!()
}

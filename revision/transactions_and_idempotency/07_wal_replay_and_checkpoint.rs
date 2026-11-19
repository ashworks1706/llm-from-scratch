// Background: a wal records each change (with the next seq) before applying it. open replays only
// seq > last checkpointed. checkpoint: persist snapshot first, then rotate (else crash = data loss).

use std::collections::HashSet;

// Insert/Delete add/remove an id; Checkpoint marks a snapshot. all carry seq.
#[derive(Debug, Clone, PartialEq)]
enum Entry { Insert { id: u64, seq: u64 }, Delete { id: u64, seq: u64 }, Checkpoint { seq: u64 } }

// version: header format version. next_seq: seq the next record gets; only grows.
struct Wal { version: u32, entries: Vec<Entry>, next_seq: u64 }

// snapshot: ids as of the last checkpoint. last_seq: highest seq already in the snapshot.
struct Durable { snapshot: HashSet<u64>, last_seq: u64 }

const WAL_VERSION: u32 = 1;

impl Wal {
    // 1. replay
    // entries with seq > min_seq, in order (Checkpoint filtered the same way).
    // version != WAL_VERSION -> Err naming the unknown version, no entries.
    // ex: [Insert 1 seq 1, Insert 2 seq 2, Delete 1 seq 3], replay(1)
    //     -> Ok([Insert 2 seq 2, Delete 1 seq 3])
    //     version 2 -> Err for any min_seq
    fn replay(&self, min_seq: u64) -> Result<Vec<Entry>, String> {
        todo!()
    }
}

// 2. checkpoint
// first: durable.snapshot = copy of state, durable.last_seq = wal.next_seq.saturating_sub(1).
// only then clear wal.entries. never reset next_seq.
// ex: next_seq 4, 3 entries, state {2} -> snapshot {2}, last_seq 3, entries [], next_seq 4
fn checkpoint(wal: &mut Wal, durable: &mut Durable, state: &HashSet<u64>) {
    todo!()
}

// 3. open
// start from a copy of durable.snapshot, apply replay(durable.last_seq) in order:
// Insert adds, Delete removes, Checkpoint no-op. replay Err -> return it.
// must be correct after a crash between snapshot and log clear: never apply seq <= last_seq.
// ex: snapshot {1, 2}, last_seq 3, [Insert 9 seq 3, Delete 1 seq 4, Insert 5 seq 5] -> Ok({2, 5})
fn open(wal: &Wal, durable: &Durable) -> Result<HashSet<u64>, String> {
    todo!()
}

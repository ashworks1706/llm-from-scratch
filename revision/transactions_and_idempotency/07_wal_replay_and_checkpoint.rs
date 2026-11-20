// Background: a wal records each change (with the next seq) before applying it. open replays only
// seq > last checkpointed. checkpoint: persist snapshot first, then rotate (else crash = data loss).

use std::collections::HashSet;

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
    fn replay(&self, min_seq: u64) -> Result<Vec<Entry>, String> {
        todo!()
    }
}

// 2. checkpoint
// first: durable.snapshot = copy of state, durable.last_seq = wal.next_seq.saturating_sub(1).
// only then clear wal.entries. never reset next_seq.
fn checkpoint(wal: &mut Wal, durable: &mut Durable, state: &HashSet<u64>) {
    todo!()
}

// 3. open
// start from a copy of durable.snapshot, apply replay(durable.last_seq) in order:
// Insert adds, Delete removes, Checkpoint no-op. replay Err -> return it.
// must be correct after a crash between snapshot and log clear: never apply seq <= last_seq.
fn open(wal: &Wal, durable: &Durable) -> Result<HashSet<u64>, String> {
    todo!()
}

// ---- try it ----
fn main() {
    let entries = vec![Entry::Insert { id: 1, seq: 1 }, Entry::Insert { id: 2, seq: 2 }, Entry::Delete { id: 1, seq: 3 }];
    let wal = Wal { version: WAL_VERSION, entries: entries.clone(), next_seq: 4 };
    // println!("{:?}", wal.replay(1)); // Ok([Insert { id: 2, seq: 2 }, Delete { id: 1, seq: 3 }])
    // println!("{:?}", Wal { version: 2, entries: entries.clone(), next_seq: 4 }.replay(0)); // Err(..)
    let mut wal2 = Wal { version: WAL_VERSION, entries: entries.clone(), next_seq: 4 };
    let mut durable = Durable { snapshot: HashSet::new(), last_seq: 0 };
    // checkpoint(&mut wal2, &mut durable, &HashSet::from([2]));
    // println!("{:?} {} {} {}", durable.snapshot, durable.last_seq, wal2.entries.len(), wal2.next_seq); // {2} 3 0 4
    let wal3 = Wal { version: WAL_VERSION, entries: vec![Entry::Insert { id: 9, seq: 3 }, Entry::Delete { id: 1, seq: 4 }, Entry::Insert { id: 5, seq: 5 }], next_seq: 6 };
    let d3 = Durable { snapshot: HashSet::from([1, 2]), last_seq: 3 };
    // println!("{:?}", open(&wal3, &d3)); // Ok({2, 5}) (any order)
}

// Background: a write ahead log records each change before it is applied, and every entry gets the
// next sequence number. On open, replay only entries with a sequence greater than the last
// checkpointed one. A checkpoint makes the snapshot durable first and only then rotates the log,
// since rotating earlier loses data on a crash. The log starts with a version header, and replay
// refuses a version it does not know.

use std::collections::HashSet;

// One log record. Insert adds the document id to the set. Delete removes it. Checkpoint marks that
// a snapshot was taken. Every record carries seq, its sequence number in the log.
#[derive(Debug, Clone, PartialEq)]
enum Entry { Insert { id: u64, seq: u64 }, Delete { id: u64, seq: u64 }, Checkpoint { seq: u64 } }

// The write ahead log. version is the format version from the header. entries are the records in
// order. next_seq is the sequence number the next record will get; it only grows.
struct Wal { version: u32, entries: Vec<Entry>, next_seq: u64 }

// The durable part of the state. snapshot is the set of ids as of the last checkpoint, and last_seq
// is the highest sequence number already contained in that snapshot.
struct Durable { snapshot: HashSet<u64>, last_seq: u64 }

const WAL_VERSION: u32 = 1;

// Problem: Wal.
//
// Wal.replay hands back the log records that still need to be applied.
impl Wal {
    // Problem: replay.
    //
    // Given min_seq, return a Vec of the entries whose sequence number is greater than min_seq,
    // keeping their order. Every kind of entry, Checkpoint included, has a sequence number and is
    // filtered the same way. If version is not WAL_VERSION, return Err with a message that names the
    // unknown version, and return no entries.
    //
    // Example: with entries [Insert id 1 seq 1, Insert id 2 seq 2, Delete id 1 seq 3], replay(1)
    // gives Ok([Insert id 2 seq 2, Delete id 1 seq 3]).
    // Example: a log with version 2 gives Err for any min_seq.
    fn replay(&self, min_seq: u64) -> Result<Vec<Entry>, String> {
        todo!()
    }
}

// Problem: checkpoint.
//
// Make the current state durable and then shrink the log. First copy state into
// durable.snapshot and set durable.last_seq to wal.next_seq minus 1 (use a saturating subtraction
// so a next_seq of 0 does not underflow). Only after that clear wal.entries. next_seq must keep
// counting up and is never reset. The order matters: if a crash happens between the two steps, the
// log still holds entries, but open skips those that the snapshot already contains.
//
// Example: with next_seq 4, three entries and state {2}, afterwards durable.snapshot is {2},
// durable.last_seq is 3, wal.entries is empty and wal.next_seq is still 4.
fn checkpoint(wal: &mut Wal, durable: &mut Durable, state: &HashSet<u64>) {
    todo!()
}

// Problem: open.
//
// Rebuild the set of ids after a restart. Start from a copy of durable.snapshot, then apply every
// entry that replay(durable.last_seq) returns, in order: Insert adds its id, Delete removes its id,
// and Checkpoint changes nothing. If replay returns an error, return that error. The result must be
// correct no matter where a crash happened, including after the snapshot was written but before the
// log was cleared, so entries at or below last_seq must never be applied twice or at all.
//
// Example: with snapshot {1, 2}, last_seq 3 and entries [Insert id 9 seq 3, Delete id 1 seq 4,
// Insert id 5 seq 5], open gives Ok({2, 5}), because the entry with seq 3 is already in the snapshot.
fn open(wal: &Wal, durable: &Durable) -> Result<HashSet<u64>, String> {
    todo!()
}

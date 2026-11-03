// a crash can stop a multi-file update anywhere. the fix is one atomic decision point:
// write the new files under temporary names, then create a commit marker. recovery looks at
// the marker only. marker present: finish moving the new files into place. absent: delete leftovers.
// recovery must be idempotent, so a crash during recovery is also fine.
//
// from piramid: apps/engine/database/src/collection/compact.rs (compact, recover, finish_committed,
// discard_uncommitted) and apps/engine/database/src/storage/sidecars/durable.rs (write_atomic, replace_file).
// the toy file system is a map from name to bytes; "data" and "offsets" are the live files.
// explain: why is writing the marker the commit point and not the final rename of "data"?
// explain: why does open() call recover before it loads offsets?

use std::collections::BTreeMap;

struct Fs { files: BTreeMap<String, Vec<u8>> }

impl Fs {
    fn rename(&mut self, from: &str, to: &str) -> bool {
        // move bytes from -> to, overwriting to. false if from is missing.
        todo!()
    }
    fn remove(&mut self, name: &str) {
        // removing a missing file is fine.
        todo!()
    }
}

fn recover(fs: &mut Fs) {
    // files: data, offsets, data.compact, offsets.compact, compact.commit.
    // commit present: rename each compact file that still exists over its live name, then remove commit.
    // commit absent: remove data.compact and offsets.compact. safe to run twice.
    todo!()
}

fn compact(fs: &mut Fs, live_data: Vec<u8>, live_offsets: Vec<u8>, crash_after: Option<u8>) -> Result<(), String> {
    // steps: 1 write data.compact, 2 write offsets.compact, 3 write compact.commit, 4 call recover.
    // if crash_after == Some(n), stop with Err("crash") right after step n.
    todo!()
}

// write a test loop over crash_after in [None, 1, 2, 3, 4]: call recover afterwards and assert
// data and offsets are either both old or both new, and no *.compact or commit file remains.

// a filter is a list of conditions that must all hold. an empty filter matches everything.
// the edge cases are the lesson: a missing field fails Eq and the range checks, passes Ne,
// and fails In. range checks only compare numbers, and an integer is coerced to f64 first.
// a string compared with Gt is false, never an error.
//
// from piramid: apps/engine/core/src/metadata/filter.rs (FilterCondition::matches, compare_values)
// and apps/engine/core/src/metadata/value.rs (MetadataValue). the engine evaluates filters
// against resident metadata inside the scan, so a filtered query never reads a document from disk.
// explain: why is "Ne, field absent" true but "Gt, field absent" false?
// explain: what does coercing Integer to f64 lose for very large i64 values?

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum Value { Str(String), Int(i64), Float(f64), Bool(bool), Null }

#[derive(Debug, Clone)]
enum Cond {
    Eq(String, Value),
    Ne(String, Value),
    Gt(String, Value),
    Lte(String, Value),
    In(String, Vec<Value>),
}

type Metadata = HashMap<String, Value>;

fn as_number(v: &Value) -> Option<f64> {
    // Int and Float give Some(f64); everything else None.
    todo!()
}

fn matches_cond(cond: &Cond, meta: &Metadata) -> bool {
    // follow the rules in the header. no panics, no unwrap.
    todo!()
}

fn matches_all(conds: &[Cond], meta: &Metadata) -> bool {
    // every condition must hold; zero conditions holds.
    todo!()
}

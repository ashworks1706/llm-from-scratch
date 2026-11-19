// Background: the loop dispatches tool calls by name through Arc<dyn Tool> trait objects.
// definitions come out in name order so the prompt is deterministic.

use std::collections::BTreeMap;
use std::sync::Arc;

// what the model sees about one tool.
#[derive(Debug, Clone, PartialEq)]
struct ToolDefinition { name: String, description: String }

// available: offered to the model right now (default true). call: Ok(result) or Err(message).
trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn available(&self) -> bool { true }
    fn call(&self, args: &str) -> Result<String, String>;
}

// name -> tool. re-inserting a name replaces it (plain BTreeMap insert). tests may insert directly.
#[derive(Default, Clone)]
struct ToolSet { tools: BTreeMap<String, Arc<dyn Tool>> }

impl ToolSet {
    // 1. definitions
    // definitions of tools with available() true, in name (key) order. [] -> [].
    // ex: "b_search", "a_calc" both available -> [a_calc, b_search]
    //     a_calc unavailable -> [b_search]
    fn definitions(&self) -> Vec<ToolDefinition> {
        todo!()
    }

    // 2. dispatch
    // look up by name, return exactly what the tool returns. unavailable tools are still callable.
    // unknown name -> Err with a readable message (e.g. "unknown tool: nope"); never panic.
    // ex: "echo" tool: dispatch("echo", "hi") -> Ok("hi")
    //     dispatch("nope", "") -> Err("unknown tool: nope")
    fn dispatch(&self, name: &str, args: &str) -> Result<String, String> {
        todo!()
    }
}

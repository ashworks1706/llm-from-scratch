// Background: the loop dispatches tool calls by name through Arc<dyn Tool> trait objects.
// definitions come out in name order so the prompt is deterministic.

use std::collections::BTreeMap;
use std::sync::Arc;

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
        fn definitions(&self) -> Vec<ToolDefinition> {
        todo!()
    }

    // 2. dispatch
    // look up by name, return exactly what the tool returns. unavailable tools are still callable.
    // unknown name -> Err with a readable message (e.g. "unknown tool: nope"); never panic.
    fn dispatch(&self, name: &str, args: &str) -> Result<String, String> {
        todo!()
    }
}

// ---- try it ----
struct Echo { name: &'static str, on: bool }
impl Tool for Echo {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition { name: self.name.to_string(), description: "echoes".to_string() }
    }
    fn available(&self) -> bool { self.on }
    fn call(&self, args: &str) -> Result<String, String> { Ok(args.to_string()) }
}

fn main() {
    let mut tools = ToolSet::default();
    tools.tools.insert("b_search".to_string(), Arc::new(Echo { name: "b_search", on: true }));
    tools.tools.insert("a_calc".to_string(), Arc::new(Echo { name: "a_calc", on: false }));
    // println!("{:?}", tools.definitions());          // only b_search (a_calc unavailable)
    // println!("{:?}", tools.dispatch("a_calc", "hi")); // Ok("hi"), unavailable still callable
    // println!("{:?}", tools.dispatch("nope", ""));    // Err("unknown tool: nope")
}

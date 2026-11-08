// Background: the model sees tool definitions, and the loop dispatches by name through trait
// objects. A registry maps a name to an Arc<dyn Tool>, and inserting an existing name replaces the
// old tool. Definitions come out in name order so the prompt is deterministic. A switched off tool
// is not offered but can still be looked up by name.

use std::collections::BTreeMap;
use std::sync::Arc;

// What the model is told about one tool: its name and a short description of what it does.
#[derive(Debug, Clone, PartialEq)]
struct ToolDefinition { name: String, description: String }

// A callable tool. definition returns what the model sees. available says whether the tool is
// offered to the model right now, and defaults to true. call runs the tool with a text argument
// string and returns Ok with the result text or Err with a message.
trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn available(&self) -> bool { true }
    fn call(&self, args: &str) -> Result<String, String>;
}

// A registry of tools. tools maps each tool name to the tool. Inserting under a name that already
// exists replaces the old tool, which is the normal BTreeMap insert, so no extra code is needed for
// that. Tests can insert into tools directly.
#[derive(Default, Clone)]
struct ToolSet { tools: BTreeMap<String, Arc<dyn Tool>> }

// Problem: ToolSet.
//
// ToolSet answers two questions: which tools should the model be told about, and what happens when
// the model calls one by name.
impl ToolSet {
    // Problem: definitions.
    //
    // Return the definition of every tool whose available() is true, ordered by tool name
    // (the key order of the map), so the same registry always produces the same prompt. Tools that
    // are switched off are left out. An empty registry gives an empty Vec.
    //
    // Example: with tools registered under "b_search" and "a_calc", both available, the result is
    // [definition of a_calc, definition of b_search]. If a_calc reports available false, the
    // result is just [definition of b_search].
    fn definitions(&self) -> Vec<ToolDefinition> {
        todo!()
    }

    // Problem: dispatch.
    //
    // Given a tool name and an argument string, look the tool up and call it, returning exactly
    // what the tool returns. A tool that is switched off can still be called by name. An unknown
    // name must return Err with a readable message the model can learn from, such as
    // "unknown tool: nope", and must never panic.
    //
    // Example: a tool "echo" that returns its args gives dispatch("echo", "hi") equal to Ok("hi").
    // Example: dispatch("nope", "") gives Err("unknown tool: nope") or similar text.
    fn dispatch(&self, name: &str, args: &str) -> Result<String, String> {
        todo!()
    }
}

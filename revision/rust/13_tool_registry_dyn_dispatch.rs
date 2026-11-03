// the model sees tool definitions, and the loop dispatches by name through trait objects.
// a registry maps a name to an Arc<dyn Tool>, and inserting an existing name replaces the old tool.
// definitions come out in name order so the prompt is deterministic.
// a switched off tool is not offered but can still be looked up by name.

use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
struct ToolDefinition { name: String, description: String }

trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn available(&self) -> bool { true }
    fn call(&self, args: &str) -> Result<String, String>;
}

#[derive(Default, Clone)]
struct ToolSet { tools: BTreeMap<String, Arc<dyn Tool>> }

impl ToolSet {
    fn definitions(&self) -> Vec<ToolDefinition> {
        // name order, available tools only.
        todo!()
    }

    fn dispatch(&self, name: &str, args: &str) -> Result<String, String> {
        // an unknown name is an Err whose text the model can read, and it must never panic.
        todo!()
    }
}

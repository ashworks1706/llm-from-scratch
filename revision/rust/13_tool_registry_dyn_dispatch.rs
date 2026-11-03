// the model sees tool definitions; the loop dispatches by name through trait objects.
// a registry is a map name -> Arc<dyn Tool>. insertion by an existing name replaces the old tool.
// definitions come out in name order (BTreeMap) so the prompt is deterministic, and a tool that is
// switched off (available() == false) is not offered but can still be looked up by name.
//
// from sparky: apps/engine/src/core/traits/tools/mod.rs (Tool) and
// apps/engine/src/runtime/harness/tools.rs (ToolSet::with/get/definitions). sparky's Tool is
// async via async_trait; keep this toy synchronous so no dependency is needed.
//
// explain: why Arc<dyn Tool> instead of an enum of known tools? what does the enum give back?
// explain: why does an unavailable tool stay in the map but leave the definitions list?

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
    fn with(self, tool: Arc<dyn Tool>) -> Self {
        // consume self, key by definition().name, return the updated set. same name replaces.
        todo!()
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        todo!()
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        // name order, available tools only.
        todo!()
    }

    fn dispatch(&self, name: &str, args: &str) -> Result<String, String> {
        // unknown name is an Err whose text the model can read ("no tool named ..."); never panic.
        todo!()
    }
}

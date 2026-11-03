// the retrieval seam lets a forward-pass driver call out to retrieval without knowing how it works.
// two phases: launch starts the work and returns at once; join waits and writes into the hidden state.
// splitting them is what lets a search run on its own stream while the model keeps computing.
// today the driver joins right after launching, so nothing overlaps yet. contract: wants() gates
// everything, join runs exactly once per launch, and errors name the hook and the phase.
//
// from piramid: apps/engine/model/src/fusion/hook.rs (RetrievalHook, PendingRetrieval, NoopRetrievalHook)
// and apps/engine/model/src/inference/forward/mod.rs (Driver::step, Driver::retrieve).
// explain: what is the cost the architecture doc says co-location removes, and where would a host copy appear?
// explain: why does launch take a read-only request while join gets the mutable hidden state?

#[derive(Clone, Copy, Debug)]
enum Point { SequenceStart, ChunkBoundary(usize), LayerEntry(usize) }

trait Pending: Send {
    fn join(self: Box<Self>, hidden: &mut [f32]) -> Result<(), String>;
}

trait Hook: Send + Sync {
    fn name(&self) -> &'static str;
    fn wants(&self, point: Point) -> bool;
    fn launch(&self, tokens: &[u32]) -> Result<Box<dyn Pending>, String>;
}

struct Noop;

impl Hook for Noop {
    fn name(&self) -> &'static str { todo!() }
    fn wants(&self, _point: Point) -> bool { todo!() }
    fn launch(&self, _tokens: &[u32]) -> Result<Box<dyn Pending>, String> { todo!() }
}

fn retrieve(hook: &dyn Hook, point: Point, tokens: &[u32], hidden: &mut [f32]) -> Result<(), String> {
    // not wanted: return Ok without touching anything. otherwise launch, then join once.
    // map errors to "<name> launch: <e>" and "<name> join: <e>".
    todo!()
}

fn step(hook: &dyn Hook, layers: usize, first_step: bool, finished_chunk: Option<usize>, tokens: &[u32],
        hidden: &mut [f32], run_layer: &mut dyn FnMut(usize)) -> Result<(), String> {
    // order: SequenceStart if first_step, ChunkBoundary if a chunk finished, then for each layer:
    // LayerEntry(layer) hook point, then run_layer(layer).
    todo!()
}

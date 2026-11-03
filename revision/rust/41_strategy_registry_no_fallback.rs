// a registry maps an execution mode to one implementation. "auto" is not a strategy:
// it resolves to a concrete mode for this target first, then lookup happens.
// a mode that cannot run is an error that names why. a silent fallback means the caller
// cannot tell which implementation produced its numbers.
//
// from piramid: apps/engine/hardware/src/compute/mode.rs (ExecutionMode::resolve) and
// apps/engine/hardware/src/compute/strategies/mod.rs (for_mode, compiled) are the reference.
// model the capabilities as plain bools; do not build real SIMD or GPU code.
// explain: what bug class does "error, never fallback" remove from a benchmark that compares strategies?
// explain: why is gpu "compiled but not installed" a different error from "not compiled"?

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode { Auto, Scalar, Simd, Parallel, Gpu }

#[derive(Debug, Clone, Copy)]
struct Caps {
    simd_target: bool,   // x86_64 or aarch64
    gpu_compiled: bool,  // built with the cuda feature
    gpu_installed: bool, // a device was opened at startup
}

fn resolve(mode: Mode, caps: Caps) -> Mode {
    // Auto -> Simd when the target has it, else Scalar. every other mode passes through.
    todo!()
}

fn for_mode(mode: Mode, caps: Caps) -> Result<&'static str, String> {
    // return the strategy name ("scalar", "simd", "parallel", "cuda") for the resolved mode.
    // simd and parallel need caps.simd_target. gpu needs gpu_compiled, then gpu_installed.
    // each failure is a distinct message that says what to change. never return another mode.
    todo!()
}

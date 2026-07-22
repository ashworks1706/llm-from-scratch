gpu inference is mostly an exercise in moving less data, reusing memory and keeping the device busy.

these lessons use rust and cudarc for device discovery, allocations, streams and kernel launches. cuda source files are used for operations where thread layout, shared memory and fusion need to be studied directly.

simple kernels are implemented for learning and correctness. optimized matrix multiplication should eventually use cublas while custom kernels focus on transformer specific operations and fusion opportunities.

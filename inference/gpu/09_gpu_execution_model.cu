// How is the GPU execution model different from CPU execution?
// The GPU execution model is based on the SIMT (Single Instruction, Multiple Threads) paradigm
// GPU has more cores than CPU, but each core is simpler and optimized for parallel execution of many threads.
// a CPU has more complex cores that can execute multiple instructions per clock cycle, but they are 
// optimized for sequential execution of a few threads.
// GPU is generally more efficient for tasks that can be parallelized across many threads, such as 
// graphics rendering, scientific simulations, and machine learning workloads.
// while CPU is more efficient for tasks that require complex control flow, low-latency response times, and high single-threaded performance, 
// such as web servers, databases, and general-purpose computing tasks.

// we can think of CPU and GPU with cognitive psychology terms:
// CPU can be thought of as System 2 thinking, which is slow, deliberate, and analytical, while GPU 
// can be thought of as System 1 thinking, which is fast, automatic, and intuitive.

// What are threads?
// Threads are the smallest unit of execution in CUDA. Each thread executes a kernel function 
// and has its own unique thread index (threadIdx) within a block. Threads are grouped into warps, 
// which are sets of 32 threads that execute instructions in lockstep on the GPU.

// what are blocks?
// Blocks are groups of threads that can share data through shared memory 
// and synchronize their execution.

// what are grids?
// Grids are collections of blocks that can be executed on the GPU. Each block has a unique 
// block index (blockIdx) within the grid, and the grid can be one-dimensional, two-dimensional, 
// or three-dimensional, depending on the problem being solved.

// What are warps?
// Warps are groups of 32 threads that execute instructions in lockstep on the GPU.

// what are clusters?
// Clusters are groups of warps that can share data through shared memory and synchronize their execution.

// how is memory shared between threads?
// They share with a shared memory space that is made up of on-chip memory that is shared among threads 
// in a block. This allows threads to communicate and share data efficiently, without having to 
// go through global memory, which is slower and has higher latency. Shared memory is limited in size, 
// so it is important to use it wisely and avoid bank conflicts, which can occur when multiple threads 
// try to access the same memory bank at the same time.

// we can think of these as a hierarchy:
// - A grid is made up of blocks
// - A block is made up of threads
// - A warp is a group of 32 threads within a block 

// so what exactly is SIMT and how does it work? 
// Whenever a warp executes an instruction, all threads in the warp execute the same instruction at the 
// same time. for example, if we had a linear algebra equation with four variables, and we wanted to 
// compute the result of that equation for 32 different sets of input values, we could launch a kernel 
// with 32 threads, each thread would compute the result for one set of input values, and all threads 
// would execute the same instruction at the same time, but with different input values. This is 
// the essence of SIMT, where multiple threads execute the same instruction on different data in parallel.

// why is matrix multiplication a good example of SIMT?
// Because in matrices, each element can be computed independently of the others, how? well, in math, 
// matrix multiplication is defined as the dot product of rows and columns. Each element of the 
// resulting matrix can be computed independently by taking the dot product of a row from the 
// first matrix and a column from the second matrix. This means that each thread can compute one 
// element of the resulting matrix without needing to wait for other threads to finish their computations. 



// .cu is based on C++ and can be compiled with nvcc, NVIDIA's CUDA compiler, it allows us to write code 
// that runs on the GPU, and it provides extensions to C++ for parallel programming.

// if we had to work with rust, rust becomes the host language, and we can use rust's FFI (Foreign Function Interface) to call into the .cu code,
// and we can use rust's build system to compile the .cu code into a shared library, we will look into this later



// let's print out all of the things we discussed in actual cuda.

// there are rules whenever we write a kernel:
// 1. Define __global__ function, which is the kernel function that will be executed on the GPU.
// 2. Know what we want to compute, and how many threads we need to launch.
// 3. Compute the global thread index from threadIdx, blockIdx and blockDim.
// 4. Allocate memory on the GPU for input and output data.
// 5. Copy input data from the host (CPU) to the device (GPU).
// 6. Launch the kernel with the appropriate number of blocks and threads.
// 7. Copy output data from the device (GPU) back to the host (CPU).

#include <stdio.h>

__global__ void kernel(int *output) {
    // compute the global thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // write the global thread index into the output buffer
    output[idx] = idx;
}

// now lets execute this kernel on the GPU and see what happens
int main() {
    // allocate memory on the GPU 
    int *d_output;
    
    // cudamalloc requires a pointer to a pointer, so we need to cast the address of d_output to (void**)
    // with a size of 1024 integers, we can allocate memory for 1024 threads
    cudaMalloc((void**)&d_output, sizeof(int) * 1024);

    // calculate the number of blocks and threads per block
    // we calculate this with the formula: 
    // number of blocks = (number of threads + threads per block - 1) / threads per block
    // why this works? because we want to round up the number of blocks since we can't 
    // have a fraction of a block, and we want to make sure that we have enough blocks 
    // to cover all the threads.
    // the -1 is to make sure that we round up, because if we didn't have it, we would round down
    int threads_per_block = 256;
    int number_of_blocks = (1024 + threads_per_block - 1) / threads_per_block;

    // now we launch the kernel with the calculated number of blocks and threads per block
    // the <<< >>> syntax is used to launch a kernel on the GPU, and it takes two arguments: 
    // the number of blocks and the number of threads per block
    kernel<<<number_of_blocks, threads_per_block>>>(d_output);
    
    // now i'll go over to 09_gpu_execution_model_host.rs and setup host there to use cudarc to run this file
}


// now we execute this file with cmd: 
// nvcc 09_gpu_execution_model.cu -o 09_gpu_execution_model 


// LEARNING OBJECTIVES:
// - Compute a global thread index from threadIdx, blockIdx and blockDim
// - Understand warp-synchronous execution and why branch divergence costs throughput
// - Choose block and grid dimensions for a 1D (and later 2D) problem
// - See how one kernel body runs across thousands of threads at once
// - Write each thread's global index into an output buffer to make the mapping visible




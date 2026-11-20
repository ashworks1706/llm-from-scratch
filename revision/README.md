short problem sets for brushing up, grouped by concept. each folder mixes python, rust, typescript and cuda, numbered in the order to do them.

every lesson has a short spec above each stub and a "try it" block at the bottom with example calls, commented out, with the expected output next to each. write the solution, uncomment a call and run the file from this folder:

```
./run language_basics/03_dictionaries_and_sets.py
./run types_and_errors/05_option_result_and_errors.rs
./run gpu_kernels/03_vector_add_and_grid_stride.cu
```

python runs with python3, rust with cargo run (every .rs file is a binary in Cargo.toml), typescript with node (node 22 runs .ts directly) and cuda with nvcc, which needs the gpu driver loaded.

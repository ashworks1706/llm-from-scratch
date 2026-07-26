triton is a python based language and compiler for writing gpu kernels. it is common in modern inference projects because it makes block based kernel programming and fusion experiments more accessible than raw cuda.

this is a literacy track, not a replacement for rust, candle or cudarc. use it to understand how modern inference kernels are expressed, to prototype one or two operations and to read projects that use triton heavily.

the python reference path should remain the source of expected values. any triton experiment should be benchmarked against pytorch and treated as a focused performance investigation.

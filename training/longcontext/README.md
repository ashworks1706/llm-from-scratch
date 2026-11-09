long context covers how a model trained on short sequences gets extended to long ones.

rope scaling methods like position interpolation, ntk, yarn and abf change the rotation frequencies. then a short training phase on long data teaches the model to use them, and needle and ruler style tests check if it really works. context parallelism splits one long sequence across gpus.

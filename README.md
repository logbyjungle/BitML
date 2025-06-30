# BitML
a simple neural network header for c++ that uses only weights and biases with 1 bit values and uint8_t values during training

example usage:
```c++
    bitnn::NN net({ 98304, 8192, 4096 ,98304 });

    int epochs = 100;

    bitnn::Optimizer opt(net, epochs, Xs, Ys, 1);
    opt.randomsearch(1);

    net.save("model0");
```

## features that will (maybe) be added in the future:
- better ways to save and load models
- more activation functions to choose from
- optimizers
- make it optionally run on the gpu
- use arrays and bitsets instead of vectors
- simd
- async

# BitML
a simple neural network header for c++ that uses only weights and biases with 1 bit values and uint8_t values during training

example usage:
```c++
    bitnn::NN<98304, 8192, 4096 ,98304> net;

    int epochs = 100;
    int learning_rate = 2

    bitnn::Optimizer opt(net, epochs, Xs, Ys);
    opt.randomsearch(learning_rate);

    net.save("model0");
```

## features that will (maybe) be added in the future:
- ability to save training models
- optimizers
- make a version for cuda
- simd
- async
- object pool
- other NN types such as CNN(with GRU?), RNN or LSTM, GAN

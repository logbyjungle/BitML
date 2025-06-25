# BitML
a simple neural network header for c++ that uses only weights and biases with 1 bit values and uint8_t values during training

example usage:
```c++
    std::vector<bool> X = { false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false };
    std::vector<bool> Y = { true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true };

    NN net({ 1000,1000,1000,1000,1000,20 }, 20);
    float lowestloss = 99999999.0f;
    int learningrate = 1;
    int epochs = 10000;

    for (int i = 0; i < epochs; i++) {
        auto out = net.execute(X);
        float loss = Layer::getloss(out, Y);

        if (loss < lowestloss) {
            lowestloss = loss;
            net.newbestparams();
        }
        else {
            net.oldparams();
        }
        net.tweak(learningrate);
    }
```

## features that will (maybe) be added in the future:
- a way to save and load models
- more activation functions to choose from and maybe even custom ones

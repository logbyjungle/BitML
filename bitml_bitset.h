#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace bitnn {

template <size_t N> // N is the number of inputs
class Neuron {
private:
    std::bitset<N> weights;
    std::array<uint8_t, N> bitweights;
    std::array<uint8_t, N> bestbitweights;

public:
    Neuron() {
        bitweights.fill(127);
        bestbitweights = bitweights;
    }

    inline int result(const std::bitset<N> &inputs) const {
        return static_cast<int>((inputs & weights).count());
    }

    void tweak() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, 1);

        for (size_t i = 0; i < N; i++) {
            if (dist(gen) ) {
                if (bitweights[i] != 255) {
                    bitweights[i]++;
                    if (bitweights[i] >= 128) {
                        weights.set(i);
                    }
                }
            }
            else if (bitweights[i] != 0) {
                bitweights[i]--;
                if (bitweights[i] <= 127) {
                    weights.reset(i);
                }
            }
        }
    }
};

template <size_t N, size_t M> // 2nd is the number of neurons, 1st is the number of inputs
class Layer {
private:
    std::array<bitnn::Neuron<N>, M> neurons;

    static std::bitset<M> default_activation(const std::array<int, M> &inputs) {
        std::bitset<M> outputs;
        for (size_t i = 0; i < M; i++) {
            if (inputs[i]) {
                outputs.set(i);
            }
        }
        return outputs;
    }

public:
    Layer() {};

    std::array<int, M> forward(const std::bitset<N> &inputs) const {
        std::array<int, M> outputs;
        for (size_t i = 0; i < M; i++) {
            outputs[i] = neurons[i].result(inputs);
        }
        return outputs;
    }

    void tweak() {
        for (size_t i = 0; i < M; ++i) {
            neurons[i].tweak();
        }
    }


    std::bitset<M> activation(const std::array<int, M> &inputs, std::bitset<M> (*activation_func)(const std::array<int, M> &) = default_activation) const {
        return activation_func(inputs);
    }

    static std::array<float, M> softmax(const std::array<int, M> &inputs) {
        std::array<float, M> smaxarr;
        std::array<float, M> exparr;
        int maxval = *std::max_element(inputs.begin(), inputs.end());

        for (size_t i = 0; i < M; i++) {
            exparr[i] = std::exp(static_cast<float>(inputs[i] - maxval));
        }
        float sumexp = std::accumulate(exparr.begin(), exparr.end(), 0.0f);

        for (size_t i = 0; i < M; i++) {
            smaxarr[i] = exparr[i] / sumexp;
        }
        return smaxarr;
    }

    static float getloss(const std::array<float, M> &smaxedin, const std::bitset<M> &rightout) {
        constexpr float EPSILON = 1e-7f;
        float loss = 0.0f;
        for (size_t i = 0; i < M; i++) {
            float p = std::min(std::max(smaxedin[i], EPSILON), 1.0f - EPSILON);
            if (rightout[i]) {
                loss += -logf(p);
            } else {
                loss += -logf(1.0f - p);
            }
        }
        return loss;
    }
};

template <size_t... Values> class NN {
private:
    static constexpr size_t SIZE = sizeof...(Values);
    static constexpr std::array<size_t, SIZE> VALUES = {Values...};

    std::array<void *, SIZE - 1> voidarr;

    template<size_t I>
    static constexpr inline size_t get() {
        return VALUES[I];
    }

    template <size_t I = 0>
    void construct_layers() {
        if constexpr (I < SIZE - 1) {
            voidarr[I] = new Layer<get<I>(), get<I+1>()>;
            construct_layers<I + 1>();
        }
    }

    template <size_t I = 0>
    void delete_layers() {
        if constexpr (I < SIZE-1) {
            delete static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I]);
            delete_layers<I+1>();
        }
    }

public:
    NN() {
        construct_layers();
    }

    ~NN() {
        delete_layers();
    }

    // you cannot copy a NN because you would copy the pointers and you will burn, TODO: make it work
    NN(const NN&) = delete;
    NN& operator=(const NN&) = delete;

    template <size_t I = 1>
    std::array<float, VALUES[SIZE - 1]> execute(std::bitset<VALUES[I-1]> inputs) {


        auto outputs = (*static_cast<Layer<VALUES[I - 1], VALUES[I]> *>(voidarr[I - 1])).forward(inputs);

        if constexpr (I == SIZE-1) {
            return Layer<VALUES[SIZE-2],VALUES[SIZE-1]>::softmax(outputs);
        } else {

            auto activated = (*static_cast<Layer<VALUES[I - 1], VALUES[I]> *>(voidarr[I - 1])).activation(outputs);
            return execute<I+1>(activated);
        }

    }

    template <size_t I = 0>
    void tweak() {
        if constexpr (I < SIZE-1) {
            (*static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I])).tweak();
            tweak<I+1>();
        }
    }
};

} // namespace bitnn

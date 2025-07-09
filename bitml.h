#include <filesystem>
#include <bitset>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>


namespace bitnn {

template <size_t N> // N is the number of inputs
class Neuron {
private:
    std::bitset<N> weights;
    std::array<uint8_t, N> bitweights;
    std::array<uint8_t, N> bestbitweights;

public:
// maybe a bunch of functions are public for no reason, maybe not just in the Neuron class
    Neuron() {
        bitweights.fill(127);
        bestbitweights = bitweights;
    }

    const std::bitset<N>& get_weights() const {
        return weights;
    }

    void set_weights(const std::bitset<N>& new_weights) {
        weights = new_weights;
        for (size_t i = 0; i < N; i++) {
            if (weights[i]) {
                bitweights[i] = 128;
            } else {
                bitweights[i] = 127;
            }
        }
        bestbitweights = bitweights;
    }

    int result(const std::bitset<N> &inputs) const {
        return static_cast<int>((inputs & weights).count());
    }

    void tweak(const int& learningrate = 1) {
        assert(learningrate >= 0 && learningrate <= 255 && "learning rate isnt in the range 0-255");
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, 1);

        for (size_t i = 0; i < N; i++) {
            if (dist(gen) ) {
                if (bitweights[i] + learningrate <= 255) {
                    bitweights[i] += static_cast<uint8_t>(learningrate);
                    if (bitweights[i] >= 128) {
                        weights.set(i);
                    }
                }
                else {
                    bitweights[i] = 255;
                    weights.set(i);
                }
            }
            else if (bitweights[i] - learningrate >= 0) {
                bitweights[i] -= static_cast<uint8_t>(learningrate);
                if (bitweights[i] <= 127) {
                    weights.reset(i);
                }
            }
            else {
                bitweights[i] = 0;
                weights.reset(i);
            }
        }
    }

    void newbestparams() {
        bestbitweights = bitweights;
    }

    void oldparams() {
        bitweights = bestbitweights;
        for (size_t i = 0; i < N; i++) {
            if (bitweights[i] >= 128) {
                weights.set(i);
            } else {
                weights.reset(i);
            }
        }
    }
};

template <size_t N, size_t M> // 1st is the number of inputs 2nd is the number of neurons
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

    Neuron<N>& get_neuron(size_t index) {
        return neurons[index];
    }

    void tweak(const int& learningrate = 1) {
        for (size_t i = 0; i < M; ++i) {
            neurons[i].tweak(learningrate);
        }
    }

    void newbestparams() {
        for (size_t i = 0; i < M; ++i) {
            neurons[i].newbestparams();
        }
    }

    void oldparams() {
        for (size_t i = 0; i < M; ++i) {
            neurons[i].oldparams();
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
    static constexpr size_t get() {
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

    template<size_t I = 0>
    void copy_nn(const NN& othernn) {
        if constexpr (I < SIZE - 1) {
            voidarr[I] = new Layer<get<I>(), get<I+1>()>(*static_cast<Layer<get<I>(),get<I+1>()>*>(othernn.voidarr[I]));
            copy_nn<I+1>(othernn);
        }
    }

public:
    NN() {
        construct_layers();
    }

    ~NN() {
        delete_layers();
    }

    NN(const NN& othernn) {
        copy_nn(othernn);
    }

    template<size_t I = 0>
    void load_0(std::ifstream& file) {
        if constexpr (I < SIZE - 1) { // for layer in NN
            auto* layer = static_cast<Layer<get<I>(), get<I+1>()>*>(voidarr[I]);
            for (size_t i = 0; i < get<I+1>(); ++i) { // for neuron in layer
                auto& neuron = (*layer).get_neuron(i);
                size_t num_bytes = (get<I>() + 7) / 8;
                std::vector<uint8_t> bits(num_bytes);

                file.read(reinterpret_cast<char*>(bits.data()), static_cast<std::streamsize>(num_bytes));
                std::bitset<get<I>()> loaded_weights;
                for (size_t j = 0; j < get<I>(); ++j) { // for weight in neuron
                    if (bits[j / 8] & (1 << (j % 8))) {
                        loaded_weights.set(j);
                    }
                }
                neuron.set_weights(loaded_weights);

            }
            load_0<I + 1>(file);
        }
    }

    template <size_t I = 0>
    void save_0(std::ofstream& file) {
        if constexpr (I < SIZE - 1) { // for layer in NN

            auto* layer = static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I]);
            for (size_t i = 0; i < get<I+1>(); i++) { // for neuron in layer
                const auto& neuron = (*layer).get_neuron(i);
                const auto& weights = neuron.get_weights();
                size_t num_bytes = (get<I>() + 7) / 8;
                std::vector<uint8_t> bits(num_bytes, 0);
                for (size_t j = 0; j < get<I>(); j++) { // for inputweights in neuron
                    if (weights[j]) {
                        bits[j / 8] |= (1 << (j % 8));
                    }
                }
                file.write(reinterpret_cast<char*>(bits.data()), static_cast<std::streamsize>(num_bytes));
            }

            save_0<I+1>(file);
        }
    }

    void save(std::string filename = "", const int& mode = 0) {
        // add setting to override file if it already exists, also we have to input a name
        if (filename.empty() || filename.size() < 6 || std::filesystem::exists(filename)) {
            int number = 0;
            do {
                number++;
                filename = "model_" + std::to_string(number) + ".blmod";
            }
            while (std::filesystem::exists(filename));
        }
        else if (filename.substr(filename.size() - 6) != ".blmod") {
            filename += ".blmod";
        }
        std::ofstream file(filename, std::ios::binary);

        // saving just the bitset values and not the traning ones
        if (mode == 0) {
            save_0(file);
            std::cout << "saved model (no training values) to file " << filename << "\n";
        }

        file.close();
    }

    void load(const std::string& filename, const int& mode = 0) {
        if (! std::filesystem::exists(filename)) {
            std::cout << "unable to find file " << filename << "\n";
            return;
        }
        std::ifstream file(filename, std::ios::binary);
        if (mode == 0) {
            load_0(file);
            std::cout << "loaded model (no training values) to file " << filename << "\n";
        }
        file.close();
    }

    NN& operator=(const NN& othernn) {
        if (this != &othernn) {
            delete_layers();
            copy_nn(othernn);
        }
        return *this;
    }

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
    void tweak(const int& learningrate = 1) {
        if constexpr (I < SIZE-1) {
            (*static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I])).tweak(learningrate);
            tweak<I+1>(learningrate);
        }
    }

    template <size_t I = 0>
    void newbestparams() {
        if constexpr (I < SIZE-1) {
            (*static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I])).newbestparams();
            newbestparams<I+1>();
        }
    }

    template <size_t I = 0>
        void oldparams() {
            if constexpr (I < SIZE-1) {
                (*static_cast<Layer<get<I>(),get<I+1>()>*>(voidarr[I])).oldparams();
                oldparams<I+1>();
            }
        }

};

template <size_t ...Values>
class Optimizer {

  private:
    NN<Values...>& net;
    const int EPOCHS;
    static constexpr size_t SIZE = sizeof...(Values);
    static constexpr std::array<size_t, SIZE> ARR = {Values...};
    static constexpr size_t FIRST = ARR[0];
    static constexpr size_t LAST = ARR[SIZE-1];
    static constexpr size_t PRELAST = ARR[SIZE-2];

    const std::vector<std::bitset<FIRST>> XS;
    const std::vector<std::bitset<LAST>> YS;

    const int DEBUGMODE;

  public:

    Optimizer(NN<Values...>& network, const int& epochs , const std::vector<std::bitset<FIRST>>& Xs, const std::vector<std::bitset<LAST>>& Ys, const int& debugmode = 0)
    : net(network), EPOCHS(epochs), XS(Xs), YS(Ys),DEBUGMODE(debugmode) {
        assert((XS.size() == YS.size()) && "inputs and outputs size mismatch");
    }

    void randomsearch(const int& learningrate = 1) {
        float lowestloss = 99999999.9f;
        if  (DEBUGMODE >= 1) {
            std::cout << "starting training...\n";
        }

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            float totalloss = 0.0f;

            for (size_t i = 0; i < XS.size(); i++) {
                auto out = net.execute(XS[i]);
                float loss = Layer<PRELAST,LAST>::getloss(out, YS[i]);
                totalloss += loss;
            }

            if (totalloss < lowestloss) {
                lowestloss = totalloss;
                net.newbestparams();
                if (DEBUGMODE >= 2) {
                    std::cout << "new lowest loss:\n";
                    std::cout << "epoch: " << epoch << "  loss: " << totalloss/static_cast<float>(XS.size()) << "  lowestloss: " << lowestloss/static_cast<float>(XS.size()) << "\n";
                }
            } else {
                if (DEBUGMODE >= 3) {
                    std::cout << "epoch: " << epoch << "  loss: " << totalloss/static_cast<float>(XS.size()) << "  lowestloss: " << lowestloss/static_cast<float>(XS.size()) << "\n";
                }
                net.oldparams();
            }

            net.tweak(learningrate);

        }

        if (DEBUGMODE >= 1) {
            std::cout << "ending tranining...\n";
        }

    }
};


} // namespace bitnn

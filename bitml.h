#include <bitset>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <numeric>
#include <string>

namespace bitnn {
    bool uint8_t_to_bool(const uint8_t& val) {
        return val >= 128;
    }

    // default activation function provided by the bitnn module, can be replaced
    std::vector<bool> default_activation(const std::vector<int>& outputs) {
        std::vector<bool> activated;
        for (size_t i = 0; i < outputs.size(); i++) {
            if (outputs[i]) {
                activated.push_back(true);
            }
            else {
                activated.push_back(false);
            }
        }
        return activated;
    }

    // used to select which activation function to use
    std::vector<bool> activation(const std::vector<int>& values, std::vector<bool>(*func)(const std::vector<int>&) = default_activation) {
        return func(values);
    }

    std::vector<float> softmax(std::vector<int> outputs) {
        std::vector<float> smaxvec;
        std::vector<float> expvec;

        int maxval = *std::max_element(outputs.begin(), outputs.end());
        for (const auto& val : outputs) {
            expvec.push_back(std::exp(static_cast<float>(val - maxval)));
        }

        float sumexp = std::accumulate(expvec.begin(), expvec.end(), 0.0f);

        for (const auto& val : expvec) {
            smaxvec.push_back(val / sumexp);
        }

        return smaxvec;
    }

    float getloss(const std::vector<float>& smaxedin, const std::vector<bool>& rightout) {
        constexpr float EPSILON = 1e-7f;
        float loss = 0.0f;

        for (size_t i = 0; i < smaxedin.size(); i++) {
            float p = std::min(std::max(smaxedin[i], EPSILON), 1.0f - EPSILON);
            if (rightout[i]) {
                loss += -log(p);
            }
            else {
                loss += -log(1.0f - p);
            }
        }
        return loss;
    }


    class Layer {
    private:
        std::vector<std::vector<bool>> boolweights;
        std::vector<std::vector<uint8_t>> bitweights;
        std::vector<std::vector<uint8_t>> bestbitweights;
        // every neuron contains a vector of weigths with len == inputs
        const int N_INPUTS;
        const int N_NEURONS;
    public:
        Layer(const int& n_inputs, const int& n_neurons)
            : N_INPUTS(n_inputs), N_NEURONS(n_neurons)
        {
            bitweights.resize(n_neurons);
            boolweights.resize(n_neurons);

            for (int i = 0; i < n_neurons; i++) {
                // assign it random value
                bitweights[i].resize(n_inputs, 128);
                boolweights[i].resize(n_inputs, true);
            }
            bestbitweights = bitweights;
        }

        // takes the inputs and gives an output that will have to be put in the activation function
        std::vector<int> forward(const std::vector<bool>& inputs, const bool& firstlayer = false) {
            assert(static_cast<int>(inputs.size()) == N_INPUTS && "Input size mismatch!");
            std::vector<int> outputs(N_NEURONS);
            int sum;
            for (int i = 0; i < N_NEURONS; i++) {
                sum = 0;
                for (int j = 0; j < N_INPUTS; j++) {
                    if (boolweights[i][j] && (inputs[j] || firstlayer)) {
                        sum++;
                    }
                }
                outputs.push_back(sum);
            }
            return outputs;
        }

        int neuron_n_weight(const int& index) {
            return boolweights[index].size();
        }

        void update() {
            for (int i = 0; i < N_NEURONS; i++) {
                for (int j = 0; j < N_INPUTS; j++) {
                    if (bitweights[i][j] >= 128) {
                        boolweights[i][j] = true;
                    }
                    else {
                        boolweights[i][j] = false;
                    }
                }
            }
        }

        void newbestparams() {
            bestbitweights = bitweights;
        }

        void oldparams() {
            bitweights = bestbitweights;
        }

        // randomly modifies the weights
        void tweak(const int& learningrate) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, 1);

            for (std::vector<uint8_t>& vecweights : bitweights) {
                for (uint8_t& weight : vecweights) {
                    int sign = dist(gen) == 0 ? -1 : 1;
                    int new_val = weight + learningrate * sign;

                    if (new_val < 0) new_val = 0;
                    if (new_val > 255) new_val = 255;

                    weight = static_cast<uint8_t>(new_val);
                }
            }
        }

        // gives back the bestbitweights in boolean form
        std::vector<std::vector<bool>> getweights() {
            std::vector<std::vector<bool>> vec;
            vec.resize(bestbitweights.size());
            for (size_t i = 0; i < bestbitweights.size(); i++) {
                vec[i].resize(bestbitweights[i].size());
                for (size_t j = 0; j < bestbitweights[i].size(); j++) {
                    vec[i].push_back(uint8_t_to_bool(bestbitweights[i][j]));
                }
            }
            return vec;
        }

        // load boolean weights
        void load(const std::vector<std::vector<bool>>& bool_weights) {
            boolweights = bool_weights;
            for (int i = 0; i < N_NEURONS; i++) {
                for (int j = 0; j < N_INPUTS; j++) {
                    bitweights[i][j] = 128 * bool_weights[i][j];
                }
            }
            bestbitweights = bitweights;
        }

        int size() {
            return N_NEURONS;
        }


    };

    class NN {
    private:
        std::vector<bitnn::Layer> layers;
        const int N_LAYERS;
    public:
        NN(const std::vector<int>& neurons)
            : N_LAYERS(neurons.size() - 1)
        {
            for (size_t i = 1; i < neurons.size(); i++) {
                layers.push_back(Layer(neurons[i - 1], neurons[i]));
            }
        }

        // randomly modifies the entire nn or just a single layer
        void tweak(const int& learningrate, const int& layerindex = -1) {
            if (layerindex == -1) {
                for (Layer& layer : layers) {
                    layer.tweak(learningrate);
                    layer.update();
                }
            }
            else {
                layers[layerindex].tweak(learningrate);
                layers[layerindex].update();
            }
        }

        void oldparams() {
            for (Layer& layer : layers) {
                layer.oldparams();
                layer.update();
            }
        }

        void newbestparams() {
            for (Layer& layer : layers) {
                layer.newbestparams();
            }
        }

        std::vector<float> execute(std::vector<bool> inputs) {
            inputs = bitnn::activation(layers[0].forward(inputs, true));
            for (size_t i = 1; i < layers.size() - 1; i++) {
                inputs = bitnn::activation(layers[i].forward(inputs));
            }
            return bitnn::softmax(layers[layers.size() - 1].forward(inputs));
        }

        // gives back the number of layers
        int size() {
            return N_LAYERS;
        }

        Layer getlayer(const int& index) {
            return layers[index];
        }

        // saves the boolean weights of the model, for execution only
        void save(const std::string& filename) {
            std::vector<std::vector<std::vector<bool>>> vec;
            vec.reserve(N_LAYERS);
            for (int i = 0; i < N_LAYERS; i++) {
                vec.push_back(layers[i].getweights());
            }

            std::ofstream out(filename);
            std::string metadata = "BitNNboolmodel1";
            out << metadata << "\n";

            out << vec.size() << "\n";
            for (const auto& layer : vec) {
                out << layer.size() << "\n";
                for (const auto& row : layer) {
                    out << row.size() << " ";
                    for (bool b : row) {
                        out << (b ? 1 : 0) << " ";
                    }
                    out << "\n";
                }
            }
        }


        void load(const std::string& filename) {
            std::ifstream in(filename);
            assert(in && "Failed to open file for reading");

            std::string metadata;
            std::getline(in, metadata);

            if (metadata == "BitNNboolmodel1") {
                size_t dim1;
                in >> dim1;
                // checks if it has the right amount of layers
                assert((static_cast<int>(dim1) == N_LAYERS) && "model shape is incorrect: layers");
                std::vector<std::vector<std::vector<bool>>> vec(dim1);
                for (size_t i = 0; i < dim1; ++i) {
                    size_t dim2;
                    in >> dim2;
                    // checks if every layer has the right amount of neurons
                    assert((static_cast<int>(dim2) == getlayer(i).size()) && "model shape is incorrect: neurons");
                    vec[i].resize(dim2);

                    for (size_t j = 0; j < dim2; ++j) {
                        size_t dim3;
                        in >> dim3;
                        // checks if every neuron has the right amount of weights
                        assert((static_cast<int>(dim3) == getlayer(i).neuron_n_weight(j)) && "model shape is incorrect: weights");
                        vec[i][j].resize(dim3);

                        for (size_t k = 0; k < dim3; ++k) {
                            int bit;
                            in >> bit;
                            vec[i][j][k] = (bit != 0);
                        }
                    }
                }

                for (int i = 0; i < N_LAYERS; i++) {
                    layers[i].load(vec[i]);
                }
            }
            else {
                std::cout << "failed to load model " << filename << "\n";
            }
        }




    };

    class Optimizer {
    private:
        const int EPOCHS;
        NN& network;
        const std::vector<std::vector<bool>> YS;
        const std::vector<std::vector<bool>> XS;
        const int DEBUGMODE;

    public:


        Optimizer(NN& neuralnetwork, const int& epochs, const std::vector<std::vector<bool>>& Xs, const std::vector<std::vector<bool>>& Ys, const int& debugmode = 0)
            : EPOCHS(epochs), network(neuralnetwork), YS(Ys), XS(Xs), DEBUGMODE(debugmode) {
            assert((YS.size() == XS.size()) && "Ys and Xs have different size"); // also assert if Xs[0].size() == layer0_n_inputs .. and if Ys[0].size() == layerlast_n_neurons
        }

        void randomsearch(const int& learningrate) {
            float lowestloss = 999999999.9f;
            std::cout << "starting training...\n";
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                float totalloss = 0;
                for (size_t i = 0; i < XS.size(); i++) {
                    auto out = network.execute(XS[i]);
                    float loss = bitnn::getloss(out, YS[i]);
                    totalloss += loss;
                }

                if (totalloss < lowestloss) {
                    lowestloss = totalloss;
                    network.newbestparams();
                }
                else {
                    network.oldparams();
                }

                if (DEBUGMODE > 0) {
                    std::cout << "epoch: " << epoch << " lowestloss: " << lowestloss / XS.size() << "\n";
                }
                network.tweak(learningrate);


            }
        }

    };

}

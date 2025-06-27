#include <bitset>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <numeric>


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
        constexpr float epsilon = 1e-7f;
        float loss = 0.0f;

        for (size_t i = 0; i < smaxedin.size(); i++) {
            float p = std::min(std::max(smaxedin[i], epsilon), 1.0f - epsilon);
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
        const int ninputs;
        const int nneurons;
    public:
        Layer(const int& n_inputs, const int& n_neurons)
            : ninputs(n_inputs), nneurons(n_neurons)
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
            assert(static_cast<int>(inputs.size()) == ninputs && "Input size mismatch!");
            std::vector<int> outputs;
            outputs.reserve(nneurons);
            int sum;
            for (int i = 0; i < nneurons; i++) {
                sum = 0;
                for (int j = 0; j < ninputs; j++) {
                    if (boolweights[i][j] && (inputs[j] || firstlayer)) {
                        sum++;
                    }
                }
                outputs.push_back(sum);
            }
            return outputs;
        }

        void update() {
            for (int i = 0; i < nneurons; i++) {
                for (int j = 0; j < ninputs; j++) {
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
    };

    class NN {
    private:
        std::vector<bitnn::Layer> layers;
        const int size;
    public:
        NN(const std::vector<int>& neurons)
            : size(neurons.size() - 1)
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
            return size;
        }

    };



}

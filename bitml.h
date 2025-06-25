#include <bitset>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>


bool ftob(const float& f) {
    if (f >= 0.5) return true;
    else return false;
}

float sum(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (const auto& val : vec) {
        sum += val;
    }
    return sum;
}


class Layer {
    std::vector<uint8_t> bitbiases; // 0 to 255
    std::vector<bool> boolbiases; // either 0 or 1

    std::vector<std::vector<uint8_t>> bitweights;
    std::vector<std::vector<bool>> boolweights;

    std::vector<std::vector<uint8_t>> bestbitweights;
    std::vector<uint8_t> bestbitbiases;

    int size;
    int ninputs;

public:
    Layer(const size_t& n_neurons, const size_t& n_inputs) {
        size = n_neurons;
        ninputs = n_inputs;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 255);

        bitweights.resize(n_neurons);
        boolweights.resize(n_neurons);

        for (size_t i = 0; i < n_neurons; ++i) {
            bitbiases.push_back(dist(gen));
            boolbiases.push_back(ftob(bitbiases[i] / 255.0f));
            for (size_t j = 0; j < n_inputs; ++j) {
                bitweights[i].push_back(dist(gen));
                boolweights[i].push_back(ftob(bitweights[i][j] / 255.0f));
            }
        }
    }

    void update() {
        for (int i = 0; i < size; i++) {
            boolbiases[i] = ftob(bitbiases[i] / 255.0f);
            for (int j = 0; j < ninputs; j++) {
                boolweights[i][j] = ftob(bitweights[i][j] / 255.0f);
            }
        }
    }



    std::vector<int> forward(const std::vector<bool>& inputs) {
        assert(static_cast<int>(inputs.size()) == ninputs && "Input size mismatch!");
        std::vector<int> outputs;
        outputs.reserve(size);

        for (int i = 0; i < size; ++i) {
            int sum = 0;
            for (int j = 0; j < ninputs; ++j) {
                if (boolweights[i][j]) {
                    sum += static_cast<int>(inputs[j]);
                }
            }
            sum += boolbiases[i];
            outputs.push_back(sum);
        }
        return outputs;
    }

    static std::vector<bool> activation(const std::vector<int>& outputs) {
        std::vector<bool> ac_outputs(outputs.size(), false);
        for (size_t i = 0; i < outputs.size(); i++) {
            if (outputs[i] >= 1) {
                ac_outputs[i] = true;
            }
        }
        return ac_outputs;
    }

    static std::vector<float> softmax(const std::vector<int>& outputs) {
        std::vector<float> smaxvec;
        std::vector<float> expvec;

        int maxval = *std::max_element(outputs.begin(), outputs.end()); // subtract max to prevent overflow
        for (const auto& val : outputs) {
            expvec.push_back(std::exp(static_cast<float>(val - maxval))); // use float to be safe
        }

        float sumexp = sum(expvec);
        for (const auto& val : expvec) {
            smaxvec.push_back(val / sumexp);
        }

        return smaxvec;
    }

    static float getloss(std::vector<float> smaxedin, std::vector<bool> rightout) {
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

    void newbestparams() {
        bestbitbiases = bitbiases;
        bestbitweights = bitweights;
    }

    void oldparams() {
        bitbiases = bestbitbiases;
        bitweights = bestbitweights;
    }

    void tweak(const int& learningrate) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, 1);

        for (uint8_t& bias : bitbiases) {
            int sign = dist(gen) == 0 ? -1 : 1;
            int new_val = bias + learningrate * sign;

            if (new_val < 0) new_val = 0;
            if (new_val > 255) new_val = 255;

            bias = static_cast<uint8_t>(new_val);
        }

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
    std::vector<Layer> layers;

public:
    NN(std::vector<int> neurons, const int& inputs) {
        neurons.insert(neurons.begin(), inputs);
        for (size_t i = 1; i < neurons.size(); i++) {
            layers.push_back(Layer(neurons[i], neurons[i - 1]));
        }
    }

    void tweak(const int& learningrate) {
        for (Layer& layer : layers) {
            layer.tweak(learningrate);
            layer.update();
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
        for (size_t i = 0; i < layers.size() - 1; i++) {
            inputs = Layer::activation(layers[i].forward(inputs));
        }
        return Layer::softmax(layers[layers.size() - 1].forward(inputs));
    }
};
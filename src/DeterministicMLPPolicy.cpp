#include "DeterministicMLPPolicy.h"

/* Constructor for DeterministicMLPPolicy
env_spec: the EnvSpec containing the observation and action space of the environment
hidden_sizes: the sizes of each hidden layer, (32, 32) default
hidden_nonlinearity: ReLu default
hidden_W_init: initializer for weights of hidden layers, He Uniform/Kaiming Uniform default
hidden_b_init: initiazlier for biases of hidden layers, default is Constant(0.)
output_nonlinearity: non linearity of output layer, default is tanh
output_W_init: initializer for weights of output layer, default is Uniform(a, b, c) where a is range, b is std (used to determine range), and c is the mean (middle point)
output_b_init: initializer for the biases of output layer, default is Uniform(a, b, c) where a is range, b is std (used to determine range), and c is the mean (middle point)
bn: if using batch_normalization or not
*/
DeterministicMLPPolicy::DeterministicMLPPolicy(EnvSpec env_spec, const std::vector<int> hidden_sizes, std::string hidden_nonlinearity, std::string hidden_W_init, std::tuple<std::string, double> hidden_b_init, std::string output_nonlinearity, std::tuple<std::string, double, double, double> output_W_init, std::tuple<std::string, double, double, double> output_b_init, bool bn): Policy(env_spec) {
    //l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim))
    

    //maybe convolutional?
    torch::nn::Linear l_obs = torch::nn::Linear(env_spec_.observation_space()->flat_dim(), hidden_sizes[0]);
    if (bn) {
        //maybe not 3d
        layers = torch::nn::Sequential(torch::nn::BatchNorm3d(env_spec_.observation_space()->flat_dim()), l_obs);
    } else {
        layers = torch::nn::Sequential(l_obs);
    }

    //add if cases for other nonlinearities
    torch::nn::init::NonlinearityType nlinearity = torch::kReLU;
    
    auto l_hidden = l_obs;
    for (size_t idx = 0; idx < hidden_sizes.size() - 1; idx++) {
        if (hidden_nonlinearity == "rectify") {
            layers->push_back(torch::nn::ReLU());
        }
        l_hidden = torch::nn::Linear(hidden_sizes[idx], hidden_sizes[idx + 1]);
        if (hidden_W_init == "HeUniform") {
            //check
            //nlinearity is relu by default, makes gain sqrt(2)
            torch::nn::init::kaiming_uniform_(l_hidden->weight, 1, torch::kFanIn, nlinearity);
        }
        if (std::get<0>(hidden_b_init) == "Constant") {
            torch::nn::init::constant_(l_hidden->bias, std::get<1>(hidden_b_init));
        }
        if (bn) {
            layers->push_back(torch::nn::BatchNorm3d(hidden_sizes[idx]));
        }
        layers->push_back(l_hidden);
    }

    //add other output nonlinearities
    if (output_nonlinearity == "tanh") {
        layers->push_back(torch::nn::Tanh());
    }
    auto l_output = torch::nn::Linear(hidden_sizes[hidden_sizes.size() - 1], env_spec_.action_space()->flat_dim());
    if (std::get<0>(output_W_init) == "Uniform") {
        double range = -(std::get<3>(output_W_init) - (sqrt(3) * std::get<2>(output_W_init)));
        torch::nn::init::uniform_(l_output->weight, -range, range);
    }
    if (std::get<0>(output_b_init) == "Uniform") {
        double range = -(std::get<3>(output_b_init) - (sqrt(3) * std::get<2>(output_b_init)));
        torch::nn::init::uniform_(l_output->bias, -range, range);
    }
    layers->push_back(l_output);
    
    output_layers_->push_back(layers);

    //don't need compile function, can just use layers->forward(input)
};

torch::Tensor DeterministicMLPPolicy::get_action(const torch::Tensor& observation) {
    return layers->forward(observation);
}

torch::Tensor DeterministicMLPPolicy::get_actions(const torch::Tensor& observations) {
    return layers->forward(observations);
}


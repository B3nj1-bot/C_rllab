#include "ContinuousMLPQFunction.h"
ContinuousMLPQFunction::ContinuousMLPQFunction(EnvSpec env_spec, const std::vector<int> hidden_sizes, std::string hidden_nonlinearity, std::string hidden_W_init, std::tuple<std::string, double> hidden_b_init, int action_merge_layer, std::string output_nonlinearity, std::tuple<std::string, double, double, double> output_W_init, std::tuple<std::string, double, double, double> output_b_init, bool bn) {
    torch::nn::Linear l_obs(torch::nn::LinearOptions(env_spec.observation_space()->flat_dim(), hidden_sizes[0]).bias(false));
    torch::nn::Linear l_action(torch::nn::LinearOptions(env_spec.action_space()->flat_dim(), hidden_sizes[0]).bias(false));
    size_t n_layers = hidden_sizes.size() + 1;
    if (n_layers > 1) {
        action_merge_layer = ((action_merge_layer % n_layers) + n_layers) % n_layers;
    } else {
        action_merge_layer = 1;
    }

    if (bn) {
        //maybe not 3d
        layers = torch::nn::Sequential(torch::nn::BatchNorm3d(env_spec.observation_space()->flat_dim()), l_obs);
    } else {
        layers = torch::nn::Sequential(l_obs);
    }
    //add if cases for other nonlinearities
    torch::nn::init::NonlinearityType nlinearity = torch::kReLU;

    auto l_hidden = l_obs;
    for (size_t idx = 0; idx < hidden_sizes.size() - 1; idx++) {
        if (bn) {
            //might not work since it also applies batchnorm to the last hidden layer but the python version doesn't?
            layers->push_back(torch::nn::BatchNorm3d(hidden_sizes[idx]));
        }

        //FIX FIX FIX
        // if (idx == action_merge_layer) {
            
        //     l_hidden = torch::cat({l_hidden, l_action}, 1);
        // }


        if (hidden_nonlinearity == "rectify") {
            layers->push_back(torch::nn::ReLU());
        }
        l_hidden = torch::nn::Linear(hidden_sizes[idx], hidden_sizes[idx + 1]);
        if (hidden_W_init == "HeUniform") {
            torch::nn::init::kaiming_uniform_(l_hidden->weight, 1, torch::kFanIn, nlinearity);
        }
        if (std::get<0>(hidden_b_init) == "Constant") {
            torch::nn::init::constant_(l_hidden->bias, std::get<1>(hidden_b_init));
        }
        layers->push_back(l_hidden);
    }
}
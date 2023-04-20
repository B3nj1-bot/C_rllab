#include "ContinuousMLPQFunction.h"
ContinuousMLPQFunction::ContinuousMLPQFunction(EnvSpec env_spec, const std::vector<int> hidden_sizes, std::string hidden_nonlinearity, std::string hidden_W_init, std::tuple<std::string, double> hidden_b_init, int action_merge_layer, std::string output_nonlinearity, std::tuple<std::string, double, double, double> output_W_init, std::tuple<std::string, double, double, double> output_b_init, bool bn) {
    //torch::nn::Linear l_obs(torch::nn::LinearOptions(env_spec.observation_space()->flat_dim(), hidden_sizes[0]));
    
    //FIX, FIND REPLACEMENT FOR l_action SINCE IT ISNT REALLY AN INPUT LAYER
    //torch::nn::Linear l_action(torch::nn::LinearOptions(env_spec.action_space()->flat_dim(), hidden_sizes[0]));

    size_t n_layers = hidden_sizes.size();
    if (n_layers > 1) {
        action_merge_layer = ((action_merge_layer % n_layers) + n_layers) % n_layers;
    } else {
        action_merge_layer = 1;
    }

    //add if cases for other nonlinearities
    torch::nn::init::NonlinearityType nlinearity = torch::kReLU;

    std::vector<int> sizes = hidden_sizes;
    sizes.insert(sizes.begin(), env_spec.observation_space()->flat_dim());
    for (size_t idx = 0; idx < hidden_sizes.size() - 1; idx++) {
        //FIX FIX FIX, use two sets of sequentials? before and after
        // if (idx == action_merge_layer) {
        //     l_hidden = torch::cat({l_hidden, l_action}, 1);
        // }
        
        auto l_hidden = torch::nn::Linear(hidden_sizes[idx], hidden_sizes[idx + 1]);
        if (hidden_W_init == "HeUniform") {
            torch::nn::init::kaiming_uniform_(l_hidden->weight, 1, torch::kFanIn, nlinearity);
        }
        if (std::get<0>(hidden_b_init) == "Constant") {
            torch::nn::init::constant_(l_hidden->bias, std::get<1>(hidden_b_init));
        }
        layers->push_back(l_hidden);

        if (bn) {
            //might not work since it also applies batchnorm to the last hidden layer but the python version doesn't?
            layers->push_back(torch::nn::BatchNorm3d(hidden_sizes[idx]));
        }
        if (hidden_nonlinearity == "rectify") {
            layers->push_back(torch::nn::ReLU());
        }
    }
}
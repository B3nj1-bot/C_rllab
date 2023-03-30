#pragma once
#include <map>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <tuple>
#include "tensor_utils.h"
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/core/parameterized.py
//cant use cache approach because variables aren't shared
class Parameterized {
    public:
        Parameterized() = default;
        //tags is like regularizble = true
        //Placeholder is Theano shared variables/expressions that contains tensors
        virtual torch::nn::ParameterList get_params_internal(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        torch::nn::ParameterList get_params(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        std::vector<caffe2::TypeMeta> get_param_dtypes(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        std::vector<torch::IntArrayRef> get_param_shapes(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        std::vector<long long> get_param_sizes(const std::map<std::string, bool>& tags);
        torch::Tensor get_param_values(const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        void set_param_values(const torch::Tensor& flattened_params, const std::map<std::string, bool>& tags = std::map<std::string, bool>());
        //not used anywhere
        //std::vector<torch::Tensor> flat_to_params(const torch::Tensor& flattened_params, const std::map<std::string, bool>& tags);

        //__getstate__
        //__setstate__
    private:

        
};
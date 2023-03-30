#include "Parameterized.h"

torch::nn::ParameterList Parameterized::get_params(const std::map<std::string, bool>& tags) {
    // Get the list of parameters, filtered by the provided tags.
    // Some common tags include 'regularizable' and 'trainable'
    return get_params_internal(tags); 

}
std::vector<caffe2::TypeMeta> Parameterized::get_param_dtypes(const std::map<std::string, bool>& tags) {
    std::vector<caffe2::TypeMeta> dtypes;
    //dict with name of parameter ("weight", "bias") and the tensor
    for (auto p: *get_params(tags)) {
        dtypes.push_back(p->dtype());
    }
    return dtypes;
}
std::vector<torch::IntArrayRef> Parameterized::get_param_shapes(const std::map<std::string, bool>& tags) {
    std::vector<torch::IntArrayRef> shapes;
    //dict with name of parameter ("weight", "bias") and the tensor
    for (auto p: *get_params(tags)) {
        shapes.push_back(p->sizes());
    }
    return shapes;
}
std::vector<long long> Parameterized::get_param_sizes(const std::map<std::string, bool>& tags) {
    std::vector<long long> sizes;
    //dict with name of parameter ("weight", "bias") and the tensor
    for (auto p: *get_params(tags)) {
        sizes.push_back(p->numel());
    }
    return sizes;
}
torch::Tensor Parameterized::get_param_values(const std::map<std::string, bool>& tags) {
    std::vector<torch::Tensor> tensors;
    //dict with name of parameter ("weight", "bias") and the tensor
    for (auto p: *get_params(tags)) {
        tensors.push_back(p.value());
    }
    return flatten_tensors(tensors);
}
void Parameterized::set_param_values(const torch::Tensor& flattened_params, const std::map<std::string, bool>& tags) {
    bool debug;
    if (tags.count("debug") > 0) {
        debug = tags.at("debug");
    }
    std::vector<torch::Tensor> param_values = unflatten_tensors(flattened_params, get_param_shapes(tags), get_param_sizes(tags));
    auto params = get_params(tags);
    auto dtypes = get_param_dtypes(tags);
    for (size_t i = 0; i < param_values.size(); i++) {
        params[i].data() = param_values[i].to(dtypes[i]);
        if (debug) {
            std::cout << "Setting value of " << params[i].name() << '\n';
        }
    }

}
// std::vector<torch::Tensor> Parameterized::flat_to_params(const torch::Tensor& flattened_params, const std::map<std::string, bool>& tags) {
//     return unflatten_tensors(flattened_params, get_param_shapes(tags), get_param_sizes(tags));
// }

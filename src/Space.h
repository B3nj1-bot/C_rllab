#pragma once
#include <torch/torch.h>
//Source: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/spaces/base.py
class Space  {
    // Provides a classification state spaces and action spaces,
    // so you can write generic code that applies to any Environment.
    // E.g. to choose a random action.

    public:
        //Uniformly randomly sample a random element of this space
        virtual torch::Tensor sample(int seed = 0);

        //Return boolean specifying if x is a valid member of this space
        virtual bool contains(const torch::Tensor& x);

        virtual torch::Tensor flatten(const torch::Tensor& x);
        virtual torch::Tensor unflatten(const torch::Tensor& x);

        virtual torch::Tensor flatten_n(const torch::Tensor& xs);
        virtual torch::Tensor unflatten_n(const torch::Tensor& xs);

        //  The dimension of the flattened vector of the tensor representation, i.e. np.prod(shape)
        virtual std::size_t flat_dim();
        virtual torch::Tensor new_tensor_variable(const unsigned int& extra_dims);

        //return lower_bound, lb
        virtual torch::Tensor lower_bound();
        //return upper_bound, ub
        virtual torch::Tensor upper_bound();
        //return shape of space
        virtual std::vector<int64_t> shape();
        
};
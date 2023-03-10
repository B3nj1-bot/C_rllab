#include <torch/torch.h>
#include <map>
#include <string>
#include <assert.h>
#include <cstdlib>
class SimpleReplayPool {
    public:
        SimpleReplayPool(const int& max_pool_size, const int& observation_dim, const int& action_dim);
        void add_sample(const torch::Tensor& observation, const torch::Tensor& action, const torch::Tensor& reward, const torch::Tensor& terminal);
        std::map<std::string, torch::Tensor> random_batch(const int& batch_size);
        int size() {return size_;}
    private:
        int max_pool_size_;
        int observation_dim_;
        int action_dim_;
        torch::Tensor observations_;
        torch::Tensor actions_;
        torch::Tensor rewards_;
        torch::Tensor terminals_;
        int bottom_;
        int top_;
        int size_;
};
#include "SimpleReplayPool.h"
SimpleReplayPool::SimpleReplayPool(const int& max_pool_size, const int& observation_dim, const int& action_dim) {
    max_pool_size_ = max_pool_size;
    observation_dim_ = observation_dim;
    action_dim_ = action_dim;
    observations_ = torch::zeros({max_pool_size, observation_dim});
    actions_ = torch::zeros({max_pool_size, action_dim});
    rewards_ = torch::zeros(max_pool_size);
    terminals_ = torch::zeros(max_pool_size, torch::TensorOptions().dtype(torch::kUInt8));

}
void SimpleReplayPool::add_sample(const torch::Tensor& observation, const torch::Tensor& action, const torch::Tensor& reward, const torch::Tensor& terminal) {
    observations_[top_] = observation;
    actions_[top_] = action;
    rewards_[top_] = reward;
    terminals_[top_] = terminal;
    top_ = (top_ + 1) % max_pool_size_;
    if (size_ >= max_pool_size_) {
        bottom_ = (bottom_ + 1) % max_pool_size_;
    } else {
        size_++;
    }

}
std::map<std::string, torch::Tensor> SimpleReplayPool::random_batch(const int& batch_size) {
    assert(size_ > batch_size);
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor indices = torch::zeros(batch_size, options);
    torch::Tensor transition_indices = torch::zeros(batch_size, options);
    int count = 0;
    while (count < batch_size) {
        int index = (rand() % (size_) + bottom_) % max_pool_size_;
        // make sure that the transition is valid: if we are at the end of the pool, we need to discard
        // this sample
        if (index == (size_ - 1) && size_ <= max_pool_size_) {
            continue;
        }
        int transition_index = (index + 1) % max_pool_size_;
        indices[count] = index;
        transition_indices[count] = transition_index;
        count++;
    }
        
    std::map<std::string, torch::Tensor> returning = {{"observations", observations_[indices]}, {"actions", actions_[indices]}, {"rewards", rewards_[indices]}, {"terminals", terminals_[indices]}, {"next_observations", observations_[transition_indices]}};
    return returning;
}
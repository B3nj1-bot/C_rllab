#pragma once
#include "SimpleReplayPool.h"
#include "Placeholder.h"
#include "DeterministicMLPPolicy.h"
#include "Policy.h"
#include "NormalizedEnv.h"
//Deep Deterministic Policy Gradient.
class DDPG {

    public:
        DDPG(Env* env, Policy policy, Placeholder qf, Placeholder es, const int& batch_size = 32, const int& n_epochs = 200, const int& epoch_length = 1000, const int& min_pool_size = 10000, const int& replay_pool_size = 100000, const float& discount = 0.99, const int& max_path_length=250, const float& qf_weight_decay=0.0, const std::string& qf_update_method = "adam", const float& qf_learning_rate = 0.001, const int& policy_weight_decay=0, const std::string& policy_update_method = "adam", const float& policy_learning_rate = 0.0001, const int& eval_samples=10000, const bool& soft_target = true, const float& soft_target_tau = 0.001, const int& n_updates_per_sample = 1, const float& scale_reward = 1.0, const bool& include_horizon_terminal_transitions = false, const bool& plot = false, const bool& pause_for_plot = false);
    private:
        int something;
        
};
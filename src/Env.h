#include "EnvSpec.h"
#include "Placeholder.h"
#include <string>
#include <tuple>
#include <map>
//SOURCE: https://gitlab.engr.illinois.edu/jundayu2/nsa-nfm/-/blob/master/src/rllab/rllab/envs/base.py

//TO-DO:
//REQUIRED: Find the type of observation and action
//REQUIRED: Find type of "paths"
//Find type of "params"
//FInd type of "horizon"
class Env  {

    public:
        struct Step {
            Placeholder observation;
            float reward;
            bool done;
            std::map<std::string, std::string> info;
        };
        // Run one timestep of the environment's dynamics. When end of episode
        // is reached, reset() should be called to reset the environment's internal state.
        // Input
        // -----
        // action : an action provided by the environment
        // Outputs
        // -------
        // (observation, reward, done, info)
        // observation : agent's observation of the current environment
        // reward [Float] : amount of reward due to the previous action
        // done : a boolean, indicating whether the episode has ended
        // info : a dictionary containing other diagnostic information from the previous action

        virtual Step step(Placeholder action); //most likely torch::tensor? 

        // Resets the state of the environment, returning an initial observation.
        // Outputs, will be same type as the type of observations
        // -------
        // observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        virtual Placeholder reset();

        //returns action space
        virtual Space* action_space();

        //returns observation space
        virtual Space* observation_space();

        std::size_t action_dim() {
            return (*action_space()).flat_dim();
        }
        void render() {}
        //Placeholder should be the type of whatever a path is
        void log_diagnostics(std::vector<Placeholder> paths) {}
        EnvSpec spec() {
            return EnvSpec(observation_space(), action_space());
        }
        //not sure what this returns. Used in GymEnv
        //i.e. self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        virtual Placeholder horizon();

        //"clean up operation"
        void terminate() {}

        //no idea where these are used
        Placeholder get_param_values() {
            return Placeholder();
        }
        void set_param_values(Placeholder params) {}

        
    
        
        
};
#include "Space.h"
class EnvSpec  {
    public:
        EnvSpec(Space *observation_space, Space *action_space): observation_space_(observation_space), action_space_(action_space) {};
        Space* observation_space() {
            return observation_space_;
        }
        Space* action_space() {
            return action_space_;
        }
    private:
        Space* observation_space_;
        Space* action_space_;
    
        
        
};
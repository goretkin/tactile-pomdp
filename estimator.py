import numpy as np

import estimator_observation_model
import estimator_belief


class Estimator(object):
    def __init__(self, state_space):
        self.state_space = state_space
        self.displacement_observation_distribution = estimator_observation_model.displacement_observation_distribution_factory(self.state_space)
        self.force_observation_distribution = estimator_observation_model.force_observation_distribution_factory(self.state_space)
        self.belief_class = estimator_belief.belief_factory(state_space)


    def distribution_over_next_state_given_sao(self, state, action, force_observation, displacement_observation, duration):
        # force_observation and/or displacmenet_observation may be None
        # to not condition on them.
        b = self.belief_class()

        one_step_belief = action.propogate_state(state, dt=duration) # this isn't the prior belief, btw. I don't know a good name.
        one_step_belief.sparsify()
        for next_state in one_step_belief.nonzero_states(): #one_step_belief.nonzero_states():            
            p_o = 1
            if displacement_observation is not None:
                p_d = self.displacement_observation_distribution(state, next_state)(displacement_observation)
                p_o *= p_d

            if force_observation is not None:
                p_f = self.force_observation_distribution(next_state, action.velocity)(force_observation)
                p_o *= p_f
            
            # if p_o were one, then this would be just a transition update
            b[next_state] = p_o * one_step_belief[next_state]
        
        # you lost days thinking you should normalize over here. ugh.
        #b.normalize() 
        
        
        #if np.isnan(b.sum()):
        #    raise AssertionError((state, action, force_observation, displacement_observation, duration))

        return b


    def obs_act_update(self, old_belief, displacement_observation, force_observation, action, duration):
        Belief = self.belief_class

        # marginalize out state
        r = Belief.blend(
                [(old_belief[state], 
                  self.distribution_over_next_state_given_sao(
                        state, action, 
                        force_observation, 
                        displacement_observation, 
                        duration))
                for state in old_belief.nonzero_states()],
                check_normalized_beliefs=False
            )
        
        r.normalize()
        
        if np.isnan(r.sum()):
            raise AssertionError("numerical collapse")

        return r

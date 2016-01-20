import numpy as np

from estimator_state_space import (DirectionStateFactor, MetricStateFactor, 
                                   VoidStateFactor, ContactStateFactor, 
                                   State)


def guarded_velocity_factory(state_space, belief_class):
    Belief = belief_class

    class GuardedVelocity(object):
        def __init__(self, velocity, stuck_probability=0.1, void_probability=0.9):
            self.velocity = velocity
            #self.time_left = 1.0
            #self.finished = False
            self.stuck_probability = stuck_probability
            self.void_probability = void_probability

        def propogate_belief(self, belief, dt=0.1):
            to_blend = []
            for state in state_space.states:
                a = belief.prob(state)
                b = self.propogate_state(state, dt)
                to_blend.append( (a, b) )
            
            return Belief.blend(to_blend)

        def propogate_state(self, state, dt=0.1):
            if isinstance(state.displacement, MetricStateFactor):
                x = state_space.to_continuous(state)[0]
                xprime = x - dt*self.velocity
                
                # we need to break into cases, because even through the interpolate function 
                # does return contact/void states, we cannot let the jig tunnel through the object.
                side_of_object = np.sign(x)
                object_boundary = state_space.object_half_width * side_of_object
                
                if np.sign(x-object_boundary) == np.sign(xprime-object_boundary):
                    # on the same side of the object
                    # moving along metric surface. with probability alpha, you are stuck and don't move
                    alpha = self.stuck_probability
                    
                    affine_combo = state_space.interpolate((xprime, state.direction.d))
                    affine_combo = [(a,Belief(delta=b)) for (a,b) in affine_combo]
                    b_moved = Belief.blend(affine_combo)
                    b_got_stuck = Belief(delta=state)
                    
                    b = Belief.blend([(alpha, b_got_stuck), 
                                      (1-alpha, b_moved)])
                else:
                    # made contact, or got stuck with probability alpha 
                    alpha = self.stuck_probability
                    # TODO this would activate a guard, probably. and the termination condition would be set
                    b_contact = Belief(delta=State(state.direction, ContactStateFactor()))
                    b_got_stuck = Belief(delta=state)
                    b = Belief.blend([(alpha, b_got_stuck), 
                                      (1-alpha, b_contact)])                


            elif isinstance(state.displacement, ContactStateFactor):
                contact_direction = -state.direction.d
                if contact_direction * self.velocity >= 0:
                    #pushing into contact, so stay with same contact
                    b = Belief(delta=state)
                else:
                    #moving away from contact. with probability alpha, stay in contact
                    alpha = self.stuck_probability
                    
                    contact_position = state_space.object_half_width * -state.direction.d
                    xprime = contact_position - dt*self.velocity
                    affine_combo = state_space.interpolate((xprime, state.direction.d), snap_to_metric=-np.sign(self.velocity))
                    affine_combo = [(a,Belief(delta=b)) for (a,b) in affine_combo]
                    b_metric = Belief.blend(affine_combo)
                    b_contact = Belief(delta=state)
                    
                    b = Belief.blend([  (alpha, b_contact),
                                        (1-alpha, b_metric)])

            elif isinstance(state.displacement, VoidStateFactor):
                void_direction = -state.direction.d
                if void_direction * self.velocity <= 0:
                    #going further into void, so stay in same void
                    b = Belief(delta=state)
                else:
                    #moving toward metric. with probability alpha, stay in the void
                    alpha = self.void_probability
                    
                    void_fringe = (state_space.extent) * void_direction
                    xprime = void_fringe - dt*self.velocity
                    # make sure you don't tunnel through the object
                    side_of_object = np.sign(void_fringe)
                    object_boundary = state_space.object_half_width * side_of_object
                    stayed_on_same_side = ( np.sign(void_fringe-object_boundary) == np.sign(xprime-object_boundary) )             
                    
                    if stayed_on_same_side:
                        affine_combo = state_space.interpolate((xprime, state.direction.d), snap_to_metric=np.sign(self.velocity))
                        affine_combo = [(a,Belief(delta=b)) for (a,b) in affine_combo]
                        b_notvoid = Belief.blend(affine_combo)
                    else:
                        b_notvoid = Belief(delta=State(state.direction, ContactStateFactor()))
                        
                    b_void = Belief(delta=state)
                    b = Belief.blend([  (alpha, b_void),
                                        (1-alpha, b_notvoid)])

            return b

    return GuardedVelocity

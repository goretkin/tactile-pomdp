import numpy as np
from estimator_state_space import (DirectionStateFactor, MetricStateFactor, 
                                   VoidStateFactor, ContactStateFactor, 
                                   State)


def exp_norm_piecewise(exp_param, norm_param):
    # return a normalized log pdf that is continuous at switchpoint
    # it looks like an exponential for positive values
    # it looks like a normal for negative values
    
    def f(x):
        # return np.where(x>=0, np.exp(-x/exp_param), np.exp(-(x**2)/norm_param))
        # the first piece has area s. the second piece has area sqrt(pi)/(2 * sqrt(1/s))
        # where s is, respectively, exp_param and norm_param
        eta = exp_param + np.sqrt(np.pi)/(2*np.sqrt(1.0/norm_param))
        return np.where(x>=0, (-x/exp_param), (-(x**2)/norm_param)) - np.log(eta)
    
    return f


def displacement_observation_distribution_factory(state_space):
    def displacement_observation_distribution(from_state, to_state):
        # displacement is expressed in the object coordinate frame at 
        # the time of from_state
        #in the 1d case, the object frame doesn't rotate, this is not important?
        # return log densities
        if from_state.direction != to_state.direction:
            # not too important because the probability of this transition is zero. right?
            # at least in the 1D case.
            return lambda (x): np.zeros_like(x)
            
        from_xd = state_space.to_continuous(from_state) 
        to_xd =  state_space.to_continuous(to_state)
        delta_x = from_xd[0] - to_xd[0]
        
        def not_void(displacement):
            return isinstance(displacement, MetricStateFactor) or isinstance(displacement, ContactStateFactor)
        
        if not_void(from_state.displacement) and not_void(to_state.displacement):
            # if delta_x is positive, that means the jig moved leftward = the object moved rightward
            # this is the nominal observation.
            sd = 0.02 #2 cm std. dev
            return lambda (x): (-0.5*((x-delta_x)/(sd))**2) / (sd * np.sqrt(2*np.pi))

        metric_void_trans = None
        if (not_void(from_state.displacement) and 
            isinstance(to_state.displacement, VoidStateFactor)):
            metric_void_trans = 1
        elif (isinstance(from_state.displacement, VoidStateFactor) and 
            not_void(to_state.displacement)):
            metric_void_trans = -1
        
        assert from_state.direction == to_state.direction
        
        if metric_void_trans is not None:
            d = to_xd[1]
            # the object moved, in the from_state.direction.d, at least
            # nominally abs(delta_x). It could be a little less or a lot more.
            def f(x):
                g = exp_norm_piecewise(1.0,0.02)
                return g(metric_void_trans*(x-delta_x)/d)
            return f
        
        if (isinstance(to_state.displacement, VoidStateFactor) and 
            isinstance(from_state.displacement, VoidStateFactor)):
            # doesn't happen in 1D case for now.
            sd = 10.00 #10 m std. dev
            return lambda (x): (-0.5*((x-0.0)/(sd))**2) / (sd * np.sqrt(2*np.pi))
        
        raise ValueError("Need to handle: %s %s"%(from_state, to_state))

    return displacement_observation_distribution


def force_observation_distribution_factory(state_space):
    def force_observation_distribution(to_state, twist_action):
        # force is expressed in the object coordinate frame at 
        # the time of to_state
        # in the 1d case, the object frame doesn't rotate, this is not important?
        # return log densities
        if not isinstance(to_state.displacement, ContactStateFactor):
            # not in contact. 
            # assume there's no friction, so nominally there is zero force
            # with friction, would look at the action.
            sd = 0.5 #0.5 N std. dev
            return lambda (x): (-0.5*((x-0.0)/(sd))**2) / (sd * np.sqrt(2*np.pi))
        
        else:
            # in a contact state. depending on how we got here, we expect to feel different amounts of force.
            # if we took an action with a force guard with a threshold of t, then we expect that force nominally
            # for now, fix the force threshold.
            # and in the 1D case, we don't have a lot of directions to move in, 
            # so it is not as crucial to look at the action
            
            force_threshold = 5 #Newtons
            
            f = exp_norm_piecewise(5, 3)
            return lambda(x): f(((-to_state.direction.d*x)-force_threshold))

        raise ValueError("Need to handle: %s %s"%(twist_action, to_state))

    return force_observation_distribution

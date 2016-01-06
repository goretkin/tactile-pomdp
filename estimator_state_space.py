import numpy as np
from collections import namedtuple

# I am crazy to do this. But this prevents rogue types from being generated (like DirectionStateFactor(3.2))
# And also allows these to be used as dict keys (they are immutable)

#What it doesn't let me do is make a type hierarchy that well.

DirectionStateFactorBase = namedtuple("DirectionStateFactor", ["d"])
class DirectionStateFactor(DirectionStateFactorBase):
    def __new__(cls, d):    
        if d == 1:
            _d = 1
        elif d == -1:
            _d = -1
        else:
            raise ValueError("direction: %s"%d)

        self = super(DirectionStateFactor, cls).__new__(cls, d)
        return self

    def __repr__(self):
        return "Direction: %s"%(self.d)

DisplacementStateFactor = namedtuple("DisplacementStateFactor", []) #"abstract" type

MetricStateFactorBase = namedtuple("MetricStateFactor", ["x"])
class MetricStateFactor(MetricStateFactorBase):
    def __new__(cls, x):
        assert(x%1 == 0) #it's an index into some discretization
        self = super(MetricStateFactor, cls).__new__(cls, x)
        return self

    def __repr__(self):
        return "Metric: %s"%(self.x)

class VoidStateFactor(DisplacementStateFactor):
    def __new__(cls):
        self = super(VoidStateFactor, cls).__new__(cls)
        return self

    def __eq__(self, other):
        return self.__class__ == other.__class__
    
    def __repr__(self):
        return "Void"

class ContactStateFactor(DisplacementStateFactor):
    def __new__(cls):
        self = super(ContactStateFactor, cls).__new__(cls)
        return self

    def __eq__(self, other):
        return self.__class__ == other.__class__
    
    def __repr__(self):
        return "Contact"

#I don't want them to inherit the equality method. and I was the hashing to be unique.
assert( not (ContactStateFactor() == VoidStateFactor() ) )

StateBase = namedtuple("State", ["direction", "displacement"])
    
class State(StateBase):
    def __new__(cls, direction_factor, displacement_factor):
        assert(isinstance(direction_factor, DirectionStateFactor))
        assert(isinstance(displacement_factor, MetricStateFactor) or #this line would have been avoided if I figured out subclassing
              isinstance(displacement_factor, DisplacementStateFactor))
        self = super(State, cls).__new__(cls, direction_factor, displacement_factor)
        return self

    def __repr__(self):
        return "State: " + ("+" if self.direction.d==1 else "-") + self.displacement.__repr__()

class StateSpace(object):
    def __init__(self, extent=1.0, d_xy=0.01, object_half_width=0.0636):
        self.extent = extent
        self.d_xy = d_xy
        self.object_half_width = object_half_width
        
        self.extent_grid = int(np.ceil(self.extent/self.d_xy))
        self.object_half_width_grid = int(np.ceil(object_half_width/d_xy)) # change to floor if you want some jig points slightly penetrating the object
        
        self.discretization_free = np.concatenate(
            [np.arange(-self.extent_grid, -self.object_half_width_grid+1),
            np.arange(self.object_half_width_grid, self.extent_grid+1)]) * d_xy
        
        # negative positions have a +1 direction, and vice versa
        self.n_directions = 2
        
        self.others_displacement = [ContactStateFactor, VoidStateFactor]
        self.n_states = len(self.discretization_free) + (len(self.others_displacement))*self.n_directions
        
        self.manifold_1d = set()
        self.manifold_0d = set()
        
        self.ordered_states = []

        for d in [DirectionStateFactor(-1), DirectionStateFactor(+1)]:
            for s in [State(d, ContactStateFactor()), State(d, VoidStateFactor())]:
                self.manifold_0d.add(s)
                self.ordered_states.append(s)

        for i in range(len(self.discretization_free)):
            s =  State(DirectionStateFactor(-np.sign(self.discretization_free[i])),
                        MetricStateFactor(i))

            self.manifold_1d.add(s)
            self.ordered_states.append(s)

        self.states = set.union(self.manifold_1d, self.manifold_0d)

        self.state_to_id = {s:i for (i,s) in enumerate(self.ordered_states)}
        self.id_to_state = {i:s for (i,s) in enumerate(self.ordered_states)} # just for symmetry's sake.
            
            
    def to_continuous(self, state, frame="object"):
        # should perhaps more aptly be called "embed"
        if frame != "object":
            raise NotImplementedError()
                
        if isinstance(state.displacement, MetricStateFactor):
            return (self.discretization_free[state.displacement.x], state.direction.d)
        elif isinstance(state.displacement, VoidStateFactor):
            return ((self.extent+self.d_xy/2.0)*-state.direction.d, state.direction.d)
        elif isinstance(state.displacement, ContactStateFactor):
            return (self.object_half_width*-state.direction.d, state.direction.d)

    def nearest(self, xd, frame="object"):
        items = self.interpolate(xd, frame)
        val, state = max(items, key=lambda x: x[0])
        return state

    def interpolate(self, xd, frame="object"):
        x, d = xd # displacement and direction
        if frame != "object":
            raise NotImplementedError()

        # return affine combination of states
        # interpolating x when x is outside the extent puts all of the mass in the void.
        if np.abs(x) >= self.extent_grid * self.d_xy:
            return [(1.0, State(DirectionStateFactor(d), VoidStateFactor()) )]
        elif np.abs(x) <= self.object_half_width_grid * self.d_xy:
            #raise NotImplementedError() #This might actually just need to be an error
            return [(1.0, State(DirectionStateFactor(d), ContactStateFactor()) )]
        else:
            if np.sign(x)*d > 0:
                # the jig is up and facing into us. or down and facing into us.
                raise ValueError("this is not a state in this weird state space")
                
            i = np.searchsorted(self.discretization_free, x)
            if i == 0:
                return [ (1.0, State(DirectionStateFactor(d), MetricStateFactor(0))) ]
            elif i == len(self.discretization_free):
                return [ (1.0, State(DirectionStateFactor(d), MetricStateFactor(i-1))) ]
            else:
                a = x - self.discretization_free[i-1]
                b = self.discretization_free[i] - x
                if a<0 or b<0:
                    raise AssertionError("negative distance: %s"%((a,b)))
                
                b_state = State(DirectionStateFactor(d), MetricStateFactor(i-1))
                a_state = State(DirectionStateFactor(d), MetricStateFactor(i))
                
                if a_state in self.states and b_state in self.states:
                    return [ (b/(a+b), b_state),
                             (a/(a+b), a_state) ]
                # I guess this is a sort of clamping. 
                elif a_state in self.states:
                    return [ (1.0, a_state) ]
                else:
                    return [ (1.0, b_state) ]


hashmap = {}
for s in StateSpace().states:
    if hashmap.has_key(hash(s)):
        print("hash Collision: %s: %s"%(s, hashmap[hash(s)]))
    else:
        hashmap[hash(s)] = s


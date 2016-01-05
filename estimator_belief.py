import numpy as np
from collections import defaultdict

from estimator_state_space import State

def belief_factory(state_space):
    # I want to be able to construct Belief objects without making reference to the state space all the time
    # I should understand http://stackoverflow.com/questions/20246523/how-references-to-variables-are-resolved-in-python
    class Belief(object):
        def __init__(self, delta=None):
            # if `delta` is set, create a Belief that has all probability mass concentrated on state `delta`
            self.state_space = state_space
            
            self.p = defaultdict(lambda: 0) # keys are states

            if delta is not None:
                if isinstance(delta, State):
                    self[delta] = 1.0
                else:
                    raise ValueError(delta)
        
        def prob(self, state):
            if state not in self.state_space.states:
                raise ValueError(state)
            return self.p[state]

        def __getitem__(self, state):
            return self.prob(state)
            
        def __setitem__(self, state, value):
            if value<0 or np.isnan(value):
                raise ValueError(value)
            if state not in self.state_space.states:
                raise ValueError(state)
            self.p[state] = value
        
        def nonzero_states(self):
            # structurally nonzero
            return self.p.keys()

        @staticmethod
        def blend(affine_combo, check_normalized_beliefs=False, safe_nonnegative_affine=False):
            coeffs = np.array([a for (a, b) in affine_combo])
            
            if safe_nonnegative_affine:
                #investigating a performance issue.
                if not np.isclose(np.sum(coeffs), 1.0) or not np.all(coeffs>=0.0):
                    raise ValueError("invalid combination: %s"%(coeffs))
            
            #assume each belief is normalized
            blended = Belief()

            for (a,b) in affine_combo:
                if check_normalized_beliefs and (not np.isclose(b.sum(), 1.0)):
                    raise AssertionError("Belief is not normalized")

                for s in b.nonzero_states():
                    blended.p[s] += a*b.p[s]

            if check_normalized_beliefs and (not np.isclose(blended.sum(), 1.0)):
                raise AssertionError("sum should have come out to 1.0: %s"%(blended.sum()))
            
            return blended
                
        def something(self):
            for s in self.state_space.manifold_0d:
                self.p[s] = np.random.rand()
                self.p[s] = np.random.rand()
            
            for s in self.state_space.manifold_1d:
                self.p[s] = 0.1 * np.random.rand()
        
            self.normalize()

        def sum(self):
            return sum(self.p.values())
           
        def normalize(self):
            eta = 1.0/(self.sum())

            if np.isnan(eta):
                raise AssertionError("nan")
            for s in self.nonzero_states():
                self.p[s] *= eta

        def embed_metric_grid(self):
            # return something you can then plot and it has the right shape.
            x = []
            y = []
            
            #for s in self.nonzero_states(): # this doesn't work because we want to plot the zero values.
            for s in self.state_space.states:
                if s in self.state_space.manifold_1d:
                    translation, orientation = self.state_space.to_continuous(s)
                    x.append(translation)
                    y.append(self[s])
            
            x = np.array(x)
            y = np.array(y)
            
            sorter = np.argsort(x)
            x = x[sorter]
            y = y[sorter]
            return x, y
            
            
        def plot(self, ax, frame="object", all_kwargs={}, p_metric_kwargs=dict(color='grey',linestyle="")):
            belief = self
        
            def merge_two_dicts(x, y):
                '''Given two dicts, merge them into a new dict as a shallow copy.
                y overrides x'''
                z = x.copy()
                z.update(y)
                return z

            artists=[]

            def plot_single_stem(ax, x, y, 
                                 marker_kwargs=dict(markersize=5.0, alpha=1.0, color='darkblue'), 
                                 stemline_kwargs=dict(linewidth=4.0, alpha=0.5, color=p_metric_kwargs['color'])):
                a = [] #store artists
                a.extend(
                    ax.plot([x, x], [0.0, y], **merge_two_dicts(all_kwargs, stemline_kwargs))) # should use only formal parameters (not all_kwargs) to this function
                
           
                a.extend(
                    ax.plot([x], [y], marker='s', **merge_two_dicts(all_kwargs, marker_kwargs)))
                return a
            
            for s in self.state_space.manifold_0d:
                x, d = self.state_space.to_continuous(s)
        
                artists.extend(
                    plot_single_stem(ax, x, 
                                     belief.p[s]) )
                artists.extend(
                    plot_single_stem(ax, x, 
                                     belief.p[s]) )
                    
     
            x, p = self.embed_metric_grid()
                
            artists.extend(
                ax.plot(x, p, marker=".", 
                        **merge_two_dicts(all_kwargs, p_metric_kwargs) ) )

            #draw straight lines between these points. for now, to make it easier to see beliefs.
            # however, don't connect through the object.
            gaps = np.where(np.diff(1*(np.diff(x) <= 2*state_space.d_xy)))[0]
            gaps += 1
            
            sections = [0] + list(gaps) + [len(x)-1]
            
            if not len(sections)%2==0:
                raise AssertionError(sections)
                
            for i1, i2 in zip(sections[0::2], sections[1::2]):
                i2+=1
            
                artists.extend(
                    ax.plot(x[i1:i2], p[i1:i2], marker="", 
                            **merge_two_dicts(dict(color=p_metric_kwargs['color'], linestyle="--", alpha=0.7),
                                         all_kwargs) ) )
            return artists

        def metric_mean(self):
            # The graphical notion of a mean only really makes sense in the object frame

            # relevant states
            states = set.intersection(set(self.nonzero_states()), self.state_space.manifold_1d)
                
            s = sum([self.p[s] for s in states])
        
            if s == 0:
                raise ValueError("Metric manifold has zero probability")
            if s>1 + 1e-10:
                raise ValueError("Metric manifold had %s>1 probability"%(s))
            
            m = 0
            for state in states:
                m += self.p[state]/s * self.state_space.discretization_free[state.displacement.x]
            
            return m
        
        def metric_variance(self):
            m = self.metric_mean()

            # relevant states
            states = set.intersection(set(self.nonzero_states()), self.state_space.manifold_1d)
            s = sum([self.p[s] for s in states])
            
            v = 0
                
            for state in states:
                v += self.p[state]/s * (self.state_space.discretization_free[state.displacement.x]-m)**2
            
            return v
        
        def sparsify(self, threshold=0.0):
            # in-place make non-structural zeros be structural zeros.
            # consider normalizing if threshold > 0
            for s in self.nonzero_states():
                if self.p[s] <= threshold:
                    del self.p[s]

        @staticmethod 
        def diff(b1, b2):
            # doesn't actually return a Belief...
            b = Belief()
            for s in set.union(set(b1.nonzero_states()), set(b2.nonzero_states())):
                b.p[s] = b1.p[s] - b2.p[s]
            return b

    return Belief
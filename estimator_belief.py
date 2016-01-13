import numpy as np
from collections import defaultdict
import heapq

import json

from estimator_state_space import State

from quietlog import quiet_log, silent_division_by_zero
from scipy.misc import logsumexp

def belief_factory(state_space):
    # I want to be able to construct Belief objects without making reference to the state space all the time
    # I should understand http://stackoverflow.com/questions/20246523/how-references-to-variables-are-resolved-in-python
    class Belief(object):
        def __init__(self, delta=None):
            # if `delta` is set, create a Belief that has all probability mass concentrated on state `delta`
            self.state_space = state_space
            
            self.clear()

            if delta is not None:
                if isinstance(delta, State):
                    self[delta] = 1.0
                else:
                    raise ValueError(delta)

        def clear(self):
            self.p = defaultdict(lambda: -np.inf) # keys are states

        def prob(self, state):
            if state not in self.state_space.states:
                raise ValueError(state)
            return np.exp(self.p[state])

        def __getitem__(self, state):
            return self.prob(state)
            
        def __setitem__(self, state, value):
            if value<0 or np.isnan(value):
                raise ValueError(value)
            if state not in self.state_space.states:
                raise ValueError(state)
            self.p[state] = quiet_log(value)
        
        def nonzero_states(self):
            # structurally nonzero probabilities
            return self.p.keys()

        @staticmethod
        def blend(affine_combo, check_normalized_beliefs=False, safe_nonnegative_affine=False):
            if safe_nonnegative_affine:
                coeffs = np.array([a for (a, b) in affine_combo])
                #investigating a performance issue.
                if not np.isclose(np.sum(coeffs), 1.0) or not np.all(coeffs>=0.0):
                    raise ValueError("invalid combination: %s"%(coeffs))
            
            #assume each belief is normalized
            blended = Belief()

            for (a,b) in affine_combo:
                if check_normalized_beliefs and (not np.isclose(b.sum(), 1.0)):
                    raise AssertionError("Belief is not normalized")

                for s in b.nonzero_states():
                    #blended[s] += a*b[s]
                    blended.p[s] = np.logaddexp(blended.p[s], b.p[s]+np.log(a))

            if check_normalized_beliefs and (not np.isclose(blended.sum(), 1.0)):
                raise AssertionError("sum should have come out to 1.0: %s"%(blended.sum()))
            
            return blended

        @staticmethod
        def logblend(affine_combo, check_normalized_beliefs=False, safe_nonnegative_affine=False):
            # how to avoid copying but maintain performance
            if safe_nonnegative_affine:
                coeffs = np.array([np.exp(a) for (a, b) in affine_combo])
                #investigating a performance issue.
                if not np.isclose(np.sum(coeffs), 1.0) or not np.all(coeffs>=0.0):
                    raise ValueError("invalid combination: %s"%(coeffs))
            
            #assume each belief is normalized
            blended = Belief()

            for (a,b) in affine_combo:
                if check_normalized_beliefs and (not np.isclose(b.sum(), 1.0)):
                    raise AssertionError("Belief is not normalized")

                for s in b.nonzero_states():
                    # in non-log-probability space, this is all we're computing: 
                    # blended[s] += a*b[s]
                    # see: https://en.wikipedia.org/wiki/Log_probability
                    # blended.p[s] += np.log1p(np.exp((a) + b.p[s] - blended.p[s]))
                    # it is critical to use a proper implementation of logaddexp
                    # and to not try to use the += operator
                    # because the default value of the dictionary is -inf
                    # and you can't += anything to that reasonably.
                    blended.p[s] = np.logaddexp(blended.p[s], b.p[s]+(a))

            if check_normalized_beliefs and (not np.isclose(blended.sum(), 1.0)):
                raise AssertionError("sum should have come out to 1.0: %s"%(blended.sum()))
            
            return blended
                
        def something(self):
            for s in self.state_space.manifold_0d:
                self[s] = np.random.rand()
                self[s] = np.random.rand()
            
            for s in self.state_space.manifold_1d:
                self[s] = 0.1 * np.random.rand()
        
            self.normalize()

        def sum(self):
            return np.exp(self.logsum())

        def logsum(self):
            return logsumexp(self.p.values())
           
        def normalize(self):
            eta = self.logsum()

            if np.isinf(eta):
                raise ZeroDivisionError()
            for s in self.nonzero_states():
                self.p[s] -= eta

        def embed_metric_grid(self, logprob=False):
            # return something you can then plot and it has the right shape.
            # return probabilities by default, otherwise log-space probabilities
            x = []
            y = []
            
            #for s in self.nonzero_states(): # this doesn't work because we want to plot the zero values.
            for s in self.state_space.states:
                if s in self.state_space.manifold_1d:
                    translation, orientation = self.state_space.to_continuous(s)
                    x.append(translation)
                    y.append(self.p[s] if logprob else self[s])
            
            x = np.array(x)
            y = np.array(y)
            
            sorter = np.argsort(x)
            x = x[sorter]
            y = y[sorter]
            return x, y
            
            
        def plot(self, ax, frame="object", all_kwargs={}, p_metric_kwargs=dict(color='grey',linestyle=""),
            logprob=False):
            belief = self
            beliefplot =  belief.p if logprob else belief

            logmin = np.min(belief.p.values()) #this shouldn't be nan, but it may be -np.inf
            if logmin == -np.inf:
                logmin = -10000.0 # arbitrary low number.

            h0 = logmin if logprob else 0.0 #horizontal origin

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
                    ax.plot([x, x], [h0, y], **merge_two_dicts(all_kwargs, stemline_kwargs))) # should use only formal parameters (not all_kwargs) to this function
                
           
                a.extend(
                    ax.plot([x], [y], marker='s', **merge_two_dicts(all_kwargs, marker_kwargs)))
                return a
            
            for s in self.state_space.manifold_0d:
                x, d = self.state_space.to_continuous(s)
        
                artists.extend(
                    plot_single_stem(ax, x, 
                                     beliefplot[s]) )
                artists.extend(
                    plot_single_stem(ax, x, 
                                     beliefplot[s]) )
                    
     
            x, p = self.embed_metric_grid(logprob=logprob)
                
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
                
            s = sum([self[s] for s in states])
        
            if s == 0:
                raise ValueError("Metric manifold has zero probability")
            if s>1 + 1e-10:
                raise ValueError("Metric manifold had %s>1 probability"%(s))
            
            m = 0
            for state in states:
                m += self[state]/s * self.state_space.discretization_free[state.displacement.x]
            
            return m
        
        def metric_variance(self):
            m = self.metric_mean()

            # relevant states
            states = set.intersection(set(self.nonzero_states()), self.state_space.manifold_1d)
            s = sum([self[s] for s in states])
            
            v = 0
                
            for state in states:
                v += self[state]/s * (self.state_space.discretization_free[state.displacement.x]-m)**2
            
            return v
        
        def sparsify(self, threshold=0.0):
            # in-place make non-structural zeros be structural zeros.
            # consider normalizing if threshold > 0
            for s in self.nonzero_states():
                if self.p[s] <= quiet_log(threshold):
                    del self.p[s]

        def max_p(self):
            return self.max()[1]

        def max(self):
            return max(self.p.iteritems(), key=lambda x: x[1])

        def max_n(self, n=1):
            return heapq.nlargest(n, self.p.iteritems(), key=lambda x: x[1])

        @staticmethod 
        def diff(b1, b2):
            # doesn't actually return a Belief...
            b = Belief()
            for s in set.union(set(b1.nonzero_states()), set(b2.nonzero_states())):
                b[s] = b1[s] - b2[s]
            return b

        def to_json_string(self):
            # store log probabilities
            d = {self.state_space.state_to_id[s]:self.p[s] for s in self.p.keys()}
            return json.dumps(d)

        def from_json_string(self, s):
            self.clear()
            d = json.loads(s)

            for k in d.keys():
                self.p[self.state_space.id_to_state[int(k)]] = d[k]


        def __repr__(self):
            _str = "Belief\n"
            for s in self.state_space.ordered_states:
                if self.p.has_key(s):
                    _str += s.__repr__() + ": " + str(self.p[s]) + "\n"
            return _str

        @staticmethod
        def from_state_interpolation(affine_combo):
            #this is what the StateSpace interpolator spits out
            return Belief.blend([(a,Belief(delta=b)) for (a,b) in affine_combo])



    return Belief
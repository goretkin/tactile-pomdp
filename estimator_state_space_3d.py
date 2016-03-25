from collections import namedtuple
import itertools
import numpy as np

class ManifoldClass(object):
    # don't instiate more than one.
    # kind of like an enum, except I don't know how to program.
    def __init__(self):
        # can't set attributes directly because overrided that.
        super(ManifoldClass, self).__setattr__("names", ["free", "bottom", "left", "corner"])
        super(ManifoldClass, self).__setattr__("manifolds", [0, 1, 2, 3])
        super(ManifoldClass, self).__setattr__("name_map", {k:v for (k,v) in zip(self.names, self.manifolds)})
        super(ManifoldClass, self).__setattr__("dimensionality", [3, 2, 2, 1])

        for i, name in enumerate(self.names):
            super(ManifoldClass, self).__setattr__(name, i)

    # prevent clobbering
    def __setattr__(self, key, val):
        raise ValueError()

    def repr(self, i):
        return self.names[i]

Manifold = ManifoldClass()


from domain import PlanarPolygonObjectInCorner, plot_obj, plot_jig_relative, jig_corner_pose_relative, Discretization, transform_between_jig_object

#half_height = 2
#shape_vertex_list = [ [1,half_height], [1,-half_height], [-1,-half_height], [-1,half_height] ]
po = PlanarPolygonObjectInCorner()


discretization = Discretization(po)
discretization.do_it_all()


__StateBaseNamedTuple = namedtuple('state', ['manifold', 'param'])

class __StateBase(__StateBaseNamedTuple):
    @staticmethod
    def _validate_input(manifold, param):
        if isinstance(manifold, basestring):
            manifold = Manifold.name_map[manifold]

        if len(param) != Manifold.dimensionality[manifold]:
            raise ValueError("parameter length is {} but dimensionality of manifold \"{}\ is {}"
                .format(len(param), Manifold.names[manifold], Manifold.dimensionality[manifold]))
        
        # ensure param field is a tuple for hashing purposes
        return manifold, tuple(param)

    def __repr__(self):
        """
        human-readable manifold
        """
        return "{}(\"{}\", {})".format(self.__class__.__name__, Manifold.names[self.manifold], self.param)

# the `param` here is a tuple of integers, specifying lattice coordinates
class State(__StateBase):
    def __new__(cls, manifold, param):
        manifold, param = cls._validate_input(manifold, param)
        return super(cls, State).__new__(cls, manifold=manifold, param=param)


# the `param` here is a tuple of real numbers, specifying coordinates

class ContinuousStateJig(__StateBase):
    """
    parameters specify the pose of the object expressed in jig frame
    """
    def __new__(cls, manifold, param):
        manifold, param = cls._validate_input(manifold, param)
        return super(cls, ContinuousStateObject).__new__(cls, manifold=manifold, param=param) # how to DRY

class StateSpace(object):
    # emulate the sort of interface we did for 1D
    # however, this isn't very general. It relies on the cached stuff, and doesn't regenerate it if
    # d_xy is different, or the extent, or anything.

    def __init__(self):
        self.extent = discretization.radius_object_frame
        self.d_xy = discretization.delta_xy
        self.d_r = discretization.delta_r
        self.object_half_width = None
        
        self.manifolds = {}

        self.manifolds["free"] = discretization.free_states_grid
        self.manifolds["left"] = discretization.left_states_grid
        self.manifolds["bottom"] = discretization.bottom_states_grid
        self.manifolds["corner"] = discretization.corner_states_grid

        self.manifolds_embedded = {}
        self.manifolds_embedded["free"] = discretization.free_states
        self.manifolds_embedded["left"] = discretization.left_edge_states
        self.manifolds_embedded["bottom"] = discretization.bottom_edge_states
        self.manifolds_embedded["corner"] = discretization.corner_states

        # useful for interpolating efficiently.
        self.manifold_regularity = {
            "free": discretization.free_regular_grid_in_frame,
            "left": discretization.left_regular_grid_in_frame,
            "bottom": discretization.bottom_regular_grid_in_frame,
            "corner": discretization.corner_regular_grid_in_frame}

        self.ordered_states = []

        for mi in Manifold.manifolds:
            for s in self.manifolds[Manifold.names[mi]]:
                self.ordered_states.append(State(mi, s))
        self.states = set(self.ordered_states)

        self.state_to_id = {s:i for (i,s) in enumerate(self.ordered_states)}
        self.id_to_state = {i:s for (i,s) in enumerate(self.ordered_states)} # just for symmetry's sake.

        self.ordered_embedding = np.concatenate([self.manifolds_embedded[k] for k in Manifold.names], axis=0)


    def to_continuous(self, state):
        # should perhaps more aptly be called "embed"
        # returns a three-vector

        if state not in self.states:
            raise ValueError("Not in this space")

        #if not (discretization.state_is == "pose_of_object_in_jig"):
        #    raise NotImplementedError()

        return self.ordered_embedding[self.state_to_id[state]]


    def nearest(self, xd, frame="object"):
        items = self.interpolate(xd, frame)
        val, state = max(items, key=lambda x: x[0])
        return state


    def interpolate(self, configuration, frame, manifold_projection, allow_extrapolation=False):
        configuration = np.array(configuration)
        manfold_d = Manifold.dimensionality[Manifold.name_map[manifold_projection]]
        # needs to be consistent with discretization.*regular_grid_in_frame
        configuration_projection_axes = {
            "free":[0,1,2],
            "left":[1,2],
            "bottom":[0,2],
            "corner":[2]}

        projection = configuration_projection_axes[manifold_projection]
        delta = np.array([self.d_xy, self.d_xy, self.d_r])

        #return an affine combination of States, in manifold_projection
        assert len(configuration) == 3
        # state coordinates 
        grid_in = self.manifold_regularity[manifold_projection]

        configuration_f = transform_between_jig_object(configuration, from_frame=frame, to_frame=grid_in)

        if grid_in == "jig":
            x_offset = discretization.xs[0]
            y_offset = discretization.ys[0]
            r_offset = discretization.rs[0]
        elif grid_in == "object":
            x_offset = discretization.xs_object[0]
            y_offset = discretization.ys_object[0]
            r_offset = discretization.rs_object[0]

        offset = np.array([x_offset, y_offset, r_offset])

        configuration_projected = configuration_f[projection]

        unrounded_lattice = (configuration_projected - offset[projection]) / delta[projection]

        assert len(unrounded_lattice) == manfold_d

        floored_lattice = np.floor(unrounded_lattice).astype(np.int)
        fractional_lattice = unrounded_lattice - floored_lattice

        rotation_translation_conversion_ratio = .5 #between [0,1]. 1 means translation has no effect on weighting

        # do multi-linear interpolation (2^n points, n = 1,2,3)
        lattice_neighbor_deltas = list(itertools.product(*[(0,1)]*manfold_d))

        # `lattice_neighbor_deltas` is, e.g. `[(0, 0), (0, 1), (1, 0), (1, 1)]` if `manifold_d == 2`

        def tuple_add(t1,t2):
            return tuple([e1+e2 for (e1,e2) in zip(t1, t2)])
        

        affine = []
        for neighbor_delta in lattice_neighbor_deltas:
            p = tuple_add(neighbor_delta, floored_lattice)
            # need to make angle lattice parameter wrap
            s = State(manifold=manifold_projection, param=p)

            if (not allow_extrapolation) and (s not in self.states):
                raise ValueError(
                    "this configuration: {} projected on manifold {} required extrapolation onto state {}".format(
                        configuration, manifold_projection, s)
                    )
            w = np.prod(np.where(neighbor_delta, fractional_lattice, 1-fractional_lattice))
            affine.append((w, s))

        return affine

    @staticmethod
    def max_interpolated(distribution):
        val, state = max(distribution, key=lambda x: x[0])
        return state







#namedtuple('bottom', ['y', 'theta'])
#namedtuple('left', ['x', 'theta'])
#namedtuple('corner', ['theta'])




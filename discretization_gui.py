import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

from domain import PlanarPolygonObjectInCorner, plot_obj, plot_jig_relative, Discretization

def setup_plot(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.axhline(0, )
    ax.axvline(0, )
    ax.set_xlim(-1,7)
    ax.set_ylim(-1,7)
    ax.set_aspect('equal')
    return ax

half_height = 2
shape_vertex_list = [ [1,half_height], [1,-half_height], [-1,-half_height], [-1,half_height] ]
po = PlanarPolygonObjectInCorner(vertex_list=shape_vertex_list)


discretization = Discretization(po)
discretization.discretize()

free_states = discretization.free_states
bottom_edge_states = discretization.bottom_edge_states
left_edge_states = discretization.left_edge_states
corner_states = discretization.corner_states


class Browse():
    def __init__(self, ax, discretization, domain):
        self.canvas = ax.figure.canvas
        self.domain = domain
        self.ax = ax
        self.discretization = discretization

        self.free_states_xy_dict = defaultdict(list)

        for state in discretization.free_states:
            t = tuple(np.round(state[0:2]/discretization.delta_xy))
            self.free_states_xy_dict[t].append(state)

        for t in self.free_states_xy_dict.keys():
            self.free_states_xy_dict[t].sort(key=lambda (x): x[2])

        self.to_remove = []
        self.current_xy_list = None
        self.index_into = None

    def redraw_domain(self):
        for item in self.to_remove:
            item.remove()

        self.to_remove = []

        self.to_remove.extend(
            plot_obj(self.domain, self.ax, {'alpha':.2,})
            )

        self.canvas.draw()

    def on_key(self, event):
        if self.current_xy_list is None or self.index_into is None:
            return

        if event.key == "q":
            i = +1
        elif event.key == "e":
            i = -1
        else:
            i = 0

        self.index_into = (self.index_into + i) % len(self.current_xy_list)

        pose = self.current_xy_list[self.index_into]
        self.domain.set_pose(pose)
        self.redraw_domain()

    def on_click(self, event):
        xydata = (event.xdata, event.ydata)
        if xydata == (None, None):
            return

        if self.canvas.manager.toolbar._active is True:
            return

        t = tuple(map(int, np.round(np.array(xydata) / discretization.delta_xy)))

        if t not in self.free_states_xy_dict:
            return

        pose = self.free_states_xy_dict[t][0]
        self.domain.set_pose(pose)
        self.redraw_domain()

        self.current_xy_list = self.free_states_xy_dict[t]
        self.index_into = 0


def make_oriented_segments(states, r):
    """
    `states` is a length-n indexable of states (x, y, theta)
    returns an n-by-2-by-2  of segments that when plotted show the states
    """
    
    oriented_segments = np.zeros((len(states), 2, 2))
    oriented_segments[:,0,:] = states[:,0:2]
    oriented_segments[:,1,0] = np.cos(states[:,2])
    oriented_segments[:,1,1] = np.sin(states[:,2])

    oriented_segments[:,1,:] *= r
    
    oriented_segments[:,1,:] += oriented_segments[:,0,:]
    return oriented_segments

# don't plot the quivers in xy positions where every direction is possible.
free_states_xy_dict = defaultdict(list)

for state in discretization.free_states:
    t = tuple(np.round(state[0:2]/discretization.delta_xy))
    free_states_xy_dict[t].append(state)

free_states_not_free_rotation = []

for t in free_states_xy_dict.keys():
    if len(free_states_xy_dict[t]) < 40:
        free_states_not_free_rotation.extend(free_states_xy_dict[t])

free_states_not_free_rotation = np.array(free_states_not_free_rotation)


oriented_segments = make_oriented_segments(free_states_not_free_rotation, discretization.delta_xy * 0.5 * .90)


ax = setup_plot()
ax.add_collection(matplotlib.collections.LineCollection(oriented_segments))
ax.plot(free_states[:,0], free_states[:,1], '.', alpha=0.2)

browse = Browse(ax, discretization, po)
cid = ax.figure.canvas.mpl_connect('button_press_event', browse.on_click)
cid = ax.figure.canvas.mpl_connect('key_press_event', browse.on_key)

ax.figure.show()

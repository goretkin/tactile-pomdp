import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict

from domain import PlanarPolygonObjectInCorner, plot_obj, plot_jig_relative, jig_corner_pose_relative, Discretization

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
    def __init__(self, ax, discretization, domain, frame="jig", otherdraw=None):
        self.canvas = ax.figure.canvas
        self.domain = domain
        self.ax = ax
        self.discretization = discretization
        self.frame = frame
        self.otherdraw = otherdraw #callback to call when pose has been updated

        self.free_states_xy_dict = defaultdict(list)

        for state in discretization.free_states:
            t = tuple(np.round(state[0:2]/discretization.delta_xy))
            self.free_states_xy_dict[t].append(state)

        for t in self.free_states_xy_dict.keys():
            self.free_states_xy_dict[t].sort(key=lambda (x): x[2])

        self.to_remove = []
        self.current_xy_list = None
        self.index_into = None

        # free configurations of the jig in object frame
        self.free_states_object = np.array(map(jig_corner_pose_relative, self.discretization.free_states))


    def update_figure(self):
        for item in self.to_remove:
            item.remove()

        self.to_remove = []

        self.to_remove.extend(
            plot_obj(self.domain, self.ax, {'color':"black",}) if self.frame=="jig" else
            plot_jig_relative(self.domain, self.ax, kwline={'color':"black",})
            )

    def redraw_domain(self):
        print("len self to remove", len(self.to_remove))
        if len(self.to_remove) == 0:
            #on first go, draw the canvas
            print("draw canvas!")
            self.canvas.draw()

        self.update_figure()
        if self.otherdraw is not None:
            self.otherdraw()
        self.canvas.draw()
        for item in self.to_remove:
            ax.draw_artist(item)
        #self.canvas.update()

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
            self.index_into = None
            return

        #we're in an axes. is it ours?
        if event.inaxes != self.ax:
            return

        if self.canvas.manager.toolbar._active is True:
            return

        t = tuple(map(int, np.round(np.array(xydata) / discretization.delta_xy)))

        if self.frame=="jig":
            # in the jig frame, the discretization is uniform, and we can find the nearest configurations
            # by rounding
            if t not in self.free_states_xy_dict:
                return

            pose = self.free_states_xy_dict[t][0]
            self.domain.set_pose(pose)
            self.redraw_domain()

            self.current_xy_list = self.free_states_xy_dict[t]
            self.index_into = 0

        elif self.frame=="object":
            # the mouse position is relative to an object centered at (0,0)
            # with theta=0

            # find the closest state (x,y)
            d = self.free_states_object[:,0:2] - xydata

            distances = np.sqrt(d[:,0]**2 + d[:,1]**2)
            closest_i = np.argmin(distances)

            # plot it
            self.domain.set_pose(self.discretization.free_states[closest_i])
            self.redraw_domain()

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
free_states_free_rotation = []

for t in free_states_xy_dict.keys():
    if len(free_states_xy_dict[t]) < 40:
        free_states_not_free_rotation.extend(free_states_xy_dict[t])
    else:
        free_states_free_rotation.extend(free_states_xy_dict[t])

free_states_not_free_rotation = np.array(free_states_not_free_rotation)
free_states_free_rotation = np.array(free_states_free_rotation)

frame = "object"

fig = plt.figure()

ax_object = fig.add_subplot(1,2,1)
ax_jig = fig.add_subplot(1,2,2)

browsers = []
# the two Browse objects share a single po object (the domain object).
# so changes to one updates the other, but it only gets redrawn if their axes are on the same figure 
# if not, you need to force a redraw of the other figure
for ax, frame in zip( [ax_object, ax_jig], ["object", "jig"]):
    ax.set_aspect('equal')

    if frame == "jig":
        # don't plot the quivers in xy positions where every direction is possible.
        oriented_segments = make_oriented_segments(free_states_not_free_rotation, discretization.delta_xy * 0.5 * .90)

        ax.add_collection(matplotlib.collections.LineCollection(oriented_segments))
        ax.plot(free_states[:,0], free_states[:,1], '.', alpha=0.2)

    elif frame == "object":
        # it doesn't matter here if the rotation is free or not. If an xy position is free rotating, then
        # it gets plotted as a circle of points in this view.
        free_states_object = np.array(map(jig_corner_pose_relative, discretization.free_states))
        oriented_segments = make_oriented_segments(free_states_object, discretization.delta_xy * 0.5 * .90)

        ax.add_collection(matplotlib.collections.LineCollection(oriented_segments))

    if frame == "jig":
        ax.axhline(0, )
        ax.axvline(0, )
        ax.set_xlim(-1,7)
        ax.set_ylim(-1,7)

    elif frame == "object":
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        po.set_pose((0, 0, 0))
        plot_obj(po, ax, {'alpha':1.0,})


    browse = Browse(ax, discretization, po, frame=frame)
    cid = ax.figure.canvas.mpl_connect('button_press_event', browse.on_click)
    cid = ax.figure.canvas.mpl_connect('key_press_event', browse.on_key)

    browsers.append(browse)

assert len(browsers) == 2
browsers[0].otherdraw = browsers[1].update_figure
browsers[1].otherdraw = browsers[0].update_figure

fig.show()

## build a QApplication before building other widgets
import pyqtgraph as pg
pg.mkQApp()

## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl

from pyqtgraph import QtCore
import PyQt4


class GGViewWidget(gl.GLViewWidget):
    mute_camera_movement_key = QtCore.Qt.Key_Space
	# pan with right button click instead.
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        
        if ev.buttons() == QtCore.Qt.LeftButton:
            self.orbit(-diff.x(), diff.y())
            #print self.opts['azimuth'], self.opts['elevation']
        elif ev.buttons() == QtCore.Qt.RightButton:
            if (ev.modifiers() & QtCore.Qt.ControlModifier):
                self.pan(diff.x(), 0, diff.y(), relative=True)
            else:
                self.pan(diff.x(), diff.y(), 0, relative=True)


    def evalKeyState(self):
        speed = 2.0

        # this should get called once when all keys are released too.
        if hasattr(self, "key_hold_callback"):
            self.key_hold_callback(self)

        if len(self.keysPressed) > 0:
            if GGViewWidget.mute_camera_movement_key not in self.keysPressed:
                for key in self.keysPressed:
                    if key == QtCore.Qt.Key_Right:
                        self.orbit(azim=-speed, elev=0)
                    elif key == QtCore.Qt.Key_Left:
                        self.orbit(azim=speed, elev=0)
                    elif key == QtCore.Qt.Key_Up:
                        self.orbit(azim=0, elev=-speed)
                    elif key == QtCore.Qt.Key_Down:
                        self.orbit(azim=0, elev=speed)
                    elif key == QtCore.Qt.Key_PageUp:
                        pass
                    elif key == QtCore.Qt.Key_PageDown:
                        pass
                    self.keyTimer.start(16)
        else:
            self.keyTimer.stop()



import numpy as np


from domain import PlanarPolygonObjectInCorner, plot_obj, plot_jig_relative, jig_corner_pose_relative, Discretization, transform_between_jig_object
po = PlanarPolygonObjectInCorner()


discretization = Discretization(po)
discretization.do_it_all()

frame_states = discretization.state_is.split("_")[-1]
frame_other_states = None
if frame_states == "object":
    frame_other_states = "jig"
elif frame_states == "jig":
    frame_other_states = "object"
else:
    raise ValueError()

view1 = GGViewWidget()
view1.show()
view1.setWindowTitle("Frame: {}".format(frame_states))


view2 = GGViewWidget()
view2.show()
view2.setWindowTitle("Frame:  {}".format(frame_other_states))


## create three grids, add each to the view
xgrid = gl.GLGridItem()
ygrid = gl.GLGridItem()
zgrid = gl.GLGridItem()
#view.addItem(xgrid)
#view.addItem(ygrid)
#view.addItem(zgrid)

## rotate x and y grids to face the correct direction
xgrid.rotate(90, 0, 1, 0)
ygrid.rotate(90, 1, 0, 0)

## scale each grid differently
xgrid.scale(0.2, 0.1, 0.1)
ygrid.scale(0.2, 0.1, 0.1)
zgrid.scale(0.1, 0.2, 0.1)

for view in [view1, view2]:
    view.opts['fov'] = 0.01
    view.opts['distance'] = 2e4


f = lambda x: transform_between_jig_object(x, from_frame=frame_states, to_frame=frame_other_states)
g = lambda x: x

cursors = {}
scale = [1, 1, 1.0/10.0]
pm = False
s = 0.0005

interpolants = {}

for view, t in zip([view1, view2], [g, f]):
    free_states = np.array(map(t, discretization.free_states))
    bottom_edge_states = np.array(map(t, discretization.bottom_edge_states))
    left_edge_states = np.array(map(t, discretization.left_edge_states))
    corner_states = np.array(map(t, discretization.corner_states))




    s1 = gl.GLScatterPlotItem(pos=free_states*scale, color=(1.0,1.0,1.0,0.4), size=2*s, pxMode=pm)
    s2 = gl.GLScatterPlotItem(pos=bottom_edge_states*scale, color=(1.0,0.5,1.0,0.4), size=10*s, pxMode=pm)
    s3 = gl.GLScatterPlotItem(pos=left_edge_states*scale, color=(1.0,1.0,0.5,0.4), size=10*s, pxMode=pm)
    s4 = gl.GLScatterPlotItem(pos=corner_states*scale, color=(1.0,0.5,0.5,0.4), size=15*s, pxMode=pm)

    for sc in [s1, s2, s3, s4]:
        view.addItem(sc)

    cursor = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(1.0,0.0,0.0,0.8), size=20*s, pxMode=pm)
    view.addItem(cursor)
    cursors[view] = cursor

    interpolant = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(0.0,0.0,1.0,0.8), size=10*s, pxMode=pm)
    view.addItem(interpolant)
    interpolants[view] = interpolant



import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_aspect(1)
plt.show()


import estimator_state_space_3d

ss = estimator_state_space_3d.StateSpace()

def plot_interpolant(configuration, frame):
    affine = ss.interpolate(configuration, frame, manifold_projection="free", allow_extrapolation=True)

    view1_points = []
    view2_points = []

    ax.clear()
    po.set_pose(configuration)
    plot_obj(po, ax)

    # plot jig
    ax.axhline(0,)
    ax.axvline(0,)

    xw = discretization.xmax - discretization.xmin
    yw = discretization.ymax - discretization.ymin
    ax.set_xlim(discretization.xmin-0.1*xw, discretization.xmax+0.1*xw)
    ax.set_ylim(discretization.ymin-0.1*xw, discretization.ymax+0.1*xw)

    for v, s in affine:
        if s not in ss.states:
            continue

        p = ss.to_continuous(s) # in frame frame_states

        view1_points.append(p)
        view2_points.append(transform_between_jig_object(p, from_frame=frame_states, to_frame=frame_other_states))

        po.set_pose(transform_between_jig_object(p, from_frame=frame_states, to_frame="jig"))
        plot_obj(po, ax, kwline={"color":"green", "alpha":v})

    ax.figure.canvas.draw()

    if len(view1_points) > 0:
        view1_points = np.array(view1_points) * scale
        view2_points = np.array(view2_points) * scale

        interpolants[view1].setData(pos=view1_points)
        interpolants[view2].setData(pos=view2_points)

        print(view1_points)
    else:
        print("no points to interp")



class StateKeyClosure(object):
    def __init__(self):
        view1.key_hold_callback = self.key_hold
        view2.key_hold_callback = self.key_hold

        for view in [view1, view2]:
            Q = QtCore.Qt
            view.noRepeatKeys.extend([Q.Key_Q, Q.Key_E, Q.Key_S, Q.Key_W, Q.Key_A, Q.Key_D, Q.Key_Q, Q.Key_E])

        self.last_keys = None

        self.state_jig = np.array([0.0,0.0,0.0])
        self.cursor_speed = 0.0
        self.cursor_max_speed = 0.05
        self.cursor_acceleration = 0.0001

    def key_hold(self, view):
        if view not in [view1, view2]:
            print("Unregistered view got key event")

        self.last_key = view.keysPressed

        Q = QtCore.Qt
        key_axes = [(Q.Key_S, Q.Key_W), (Q.Key_A, Q.Key_D), (Q.Key_Q, Q.Key_E)]
        move_delta = [-1, 1]
        v = []
        v_is_zero = True
        for axis in key_axes:
            e = 0

            relevant_keys = set.intersection(set(view.keysPressed.keys()), set(axis))

            for k in relevant_keys:
                e += move_delta[axis.index(k)]
                v_is_zero = False
            v.append(e)

        if v_is_zero:
            self.cursor_speed = 0.0
        else:
            # accelerate up to some limit
            self.cursor_speed += self.cursor_acceleration
            self.cursor_speed = min(self.cursor_max_speed, self.cursor_speed)
        # v contains 3-vector corresponding to key displacement
        v = np.array(v) / scale # make cursor speed appear uniform along each configuration dimension

        delta_frame = None
        if view == view1:
            delta_frame = frame_states
        elif view == view2:
            delta_frame = frame_other_states

        s = transform_between_jig_object(self.state_jig, from_frame="jig", to_frame=delta_frame)
        s += v * self.cursor_speed
        self.state_jig  = transform_between_jig_object(s, from_frame=delta_frame, to_frame="jig")

        p1 = transform_between_jig_object(self.state_jig, from_frame="jig", to_frame=frame_states)
        p2 = transform_between_jig_object(self.state_jig, from_frame="jig", to_frame=frame_other_states)
        p1 = np.array([p1]) * scale
        p2 = np.array([p2]) * scale
        cursors[view1].setData(pos=p1)
        cursors[view2].setData(pos=p2)

        view1.opts["center"] = PyQt4.QtGui.QVector3D(*p1[0])
        view2.opts["center"] = PyQt4.QtGui.QVector3D(*p2[0])

        self.update()

    def update(self):
        plot_interpolant(self.state_jig, "jig")


skc = StateKeyClosure()

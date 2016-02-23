## build a QApplication before building other widgets
import pyqtgraph as pg
pg.mkQApp()

## make a widget for displaying 3D objects
import pyqtgraph.opengl as gl

from pyqtgraph import QtCore
class GGViewWidget(gl.GLViewWidget):
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

view = GGViewWidget()
view.show()

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


view.opts['fov'] = 0.01
view.opts['distance'] = 2e4

import numpy as np


from domain import PlanarPolygonObjectInCorner, plot_obj, plot_jig_relative, jig_corner_pose_relative, Discretization
po = PlanarPolygonObjectInCorner()


discretization = Discretization(po)
discretization.do_it_all()

f = jig_corner_pose_relative
free_states = np.array(map(f, discretization.free_states))
bottom_edge_states = np.array(map(f, discretization.bottom_edge_states))
left_edge_states = np.array(map(f, discretization.left_edge_states))
corner_states = np.array(map(f, discretization.corner_states))


scale = [1, 1, 1.0/10.0]
pm = False
s = 0.0005

s1 = gl.GLScatterPlotItem(pos=free_states*scale, color=(1.0,1.0,1.0,0.4), size=2*s, pxMode=pm)
s2 = gl.GLScatterPlotItem(pos=bottom_edge_states*scale, color=(1.0,0.5,1.0,0.4), size=10*s, pxMode=pm)
s3 = gl.GLScatterPlotItem(pos=left_edge_states*scale, color=(1.0,1.0,0.5,0.4), size=10*s, pxMode=pm)
s4 = gl.GLScatterPlotItem(pos=corner_states*scale, color=(1.0,0.5,0.5,0.4), size=15*s, pxMode=pm)

for s in [s1, s2, s3, s4]:
	view.addItem(s)
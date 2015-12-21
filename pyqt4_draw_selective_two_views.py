from pyqt4_framework_two_views import *

class PyQt4DrawSelective(Pyqt4Draw):

    def ManualDraw(self):
        """
        This implements code normally present in the C++ version,
        which calls the callbacks that you see in this class (DrawSegment,
        DrawSolidCircle, etc.).
        
        This is implemented in Python as an example of how to do it, and also
        a test.
        """
        """
        we re-implement this method so that we can, for example, selectively display
        the manipulandum.
        """
        colors = {
            'active'    : b2Color(0.5, 0.5, 0.3),
            'static'    : b2Color(0.5, 0.9, 0.5), 
            'kinematic' : b2Color(0.5, 0.5, 0.9), 
            'asleep'    : b2Color(0.6, 0.6, 0.6), 
            'default'   : b2Color(0.9, 0.7, 0.7), 
        }

        settings=self.test.settings
        world=self.test.world
        if self.test.selected_shapebody:
            sel_shape, sel_body=self.test.selected_shapebody
        else:
            sel_shape=None

        if settings.drawShapes:
            for body in world.bodies:
                if (settings.hideManipulandum and 
                    body == self.test.dynamics.manipuland_body):
                    continue #don't plot 

                transform=body.transform
                for fixture in body.fixtures:
                    shape=fixture.shape

                    if not body.active: color=colors['active']
                    elif body.type==b2_staticBody: color=colors['static']
                    elif body.type==b2_kinematicBody: color=colors['kinematic']
                    elif not body.awake: color=colors['asleep']
                    else: color=colors['default']
                    
                    self.DrawShape(fixture.shape, transform, color, (sel_shape==shape))


        if settings.drawJoints:
            for joint in world.joints:
                self.DrawJoint(joint)

        # if settings.drawPairs
        #   pass

        if settings.drawAABBs:
            color=b2Color(0.9, 0.3, 0.9)
            cm=world.contactManager
            for body in world.bodies:
                if not body.active:
                    continue
                transform=body.transform
                for fixture in body.fixtures:
                    shape=fixture.shape
                    for childIndex in range(shape.childCount):
                        self.DrawAABB(shape.getAABB(transform, childIndex), color)

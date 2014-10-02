#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
# 
# Implemented using the pybox2d SWIG interface for Box2D (pybox2d.googlecode.com)
# 
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

from framework import *

import numpy as np
import random
import pygame

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon

class AngleJoint:
    def __init__(self,revolute_joint):
        self.revolute_joint = revolute_joint
        self.target = 0
        self.controlled_body = revolute_joint.bodyB

    def solve(self):
        err = self.controlled_body.angle-self.target

        self.revolute_joint.maxMotorTorque = 3000.0
        self.revolute_joint.motorEnabled = True
        #grasp_body.angularDamping = 30.0
        self.revolute_joint.motorSpeed = (err) * (-4.0) + self.controlled_body.angularVelocity * (-0.0)

def mpl_plot_b2Body(ax,body,mpl_kwargs={}):
    vertices = [tuple(body.transform*v) for v in body.fixtures[0].shape.vertices]
    vertices = np.array(vertices)
    p = mplPolygon(vertices,**mpl_kwargs)
    ax.add_patch(p)

def mpl_plot_b2Fixture(ax,fixture,transform,mpl_kwargs={}):
    vertices = [tuple(transform*v) for v in fixture.shape.vertices]
    vertices = np.array(vertices)
    p = mplPolygon(vertices,**mpl_kwargs)
    ax.add_patch(p)


class Filter():
    def __init__(self,simulator,world_dynamics,n_particles=100):
        self.n_particles = n_particles

        self.particles = np.zeros(shape=(self.n_particles,6))
        self.weights = np.ones(shape=(self.n_particles,)) * 1.0/self.n_particles
        
        self.observation = np.zeros(shape=(self.n_particles,3))

        self.simulator = simulator
        self.world_dynamics=world_dynamics

        for i in range(self.n_particles):
            t = self.simulator.sample_manipuland_in_hand()
            self.particles[i,0:2] = t.position
            self.particles[i,2] = t.angle

        self.sim_timeStep = 1.0/60.0
        self.sim_velocityIterations = 500
        self.sim_positionIterations = 500

    def plot(self):
        fig = plt.gcf()
        ax = fig.add_subplot(111, aspect='equal')
        mpl_plot_b2Body(ax,self.simulator.ground_body)
        ax.plot(self.particles[:,0],self.particles[:,1],'o',alpha=.5)

        for i in range(self.n_particles):
            pos = self.particles[i,0:2]
            ang = self.particles[i,2]

            t = Box2D.b2Transform(pos,Box2D.b2Rot(ang))

            mpl_plot_b2Fixture(ax,self.simulator.manipuland_fixture,t,
                {'alpha':0.1,'facecolor':'none'})

            x1,y1 = self.particles[i,0:2]
            x2,y2 = self.particles[i,0:2] + 1.0 * self.observation[i,0:2]

            ax.plot([x1,x2],[y1,y2], 'y-', alpha=0.5)

        mpl_plot_b2Body(ax,self.simulator.grasp_body,{'alpha':0.3,'facecolor':'red'})
        x,y = self.simulator.grasp_body.position

        ax.plot(x,y,'r.')

    def update_particle_validity(self):
        c = 0
        for i in range(self.n_particles):
            valid = self.simulator.is_valid_manipuland_state(self.particles[i,:])
            if not valid:
                self.weights[i] = 0.0
                c += 1
        return c

    def apply_transition(self):
        for i in range(self.n_particles):
            self.simulator.grasp_body.linearVelocity = self.world_dynamics.grasp_body.linearVelocity.__copy__()
            self.simulator.grasp_body.angularVelocity = self.world_dynamics.grasp_body.angularVelocity
            self.simulator.grasp_body.position = self.world_dynamics.grasp_body.position.__copy__()
            self.simulator.grasp_body.angle = self.world_dynamics.grasp_body.angle


            self.simulator.manipuland_body.linearVelocity = self.particles[i,3:5]
            self.simulator.manipuland_body.angularVelocity = self.particles[i,5]
            self.simulator.manipuland_body.position = self.particles[i,0:2]
            self.simulator.manipuland_body.angle = self.particles[i,2]

            self.simulator.create_grasp_slip_joint()

            self.simulator.step_grasp() #we might find that the manipuland is no longer in the hand.

            self.simulator.world.Step(self.sim_timeStep, self.sim_velocityIterations, self.sim_positionIterations)
            self.simulator.world.ClearForces()

            self.particles[i,3:5] = self.simulator.manipuland_body.linearVelocity
            self.particles[i,5] = self.simulator.manipuland_body.angularVelocity
            self.particles[i,0:2] = self.simulator.manipuland_body.position
            self.particles[i,2] = self.simulator.manipuland_body.angle

            if self.simulator.grasp_slip_joint:
                self.observation[i,0:2] = self.simulator.grasp_slip_joint.GetReactionForce(1.0)
                self.observation[i,2] = self.simulator.grasp_slip_joint.GetReactionTorque(1.0)
            else:
                self.observation[i,:] = np.nan






class Dynamics():
    def __init__(self,world=None,gripper_direct_control=False):
        self.gripper_direct_control = gripper_direct_control

        if world is None:
            self.world = Box2D.b2World(gravity=(0,-10), doSleep=True)
            self.world.warmStarting = False
        else:
            self.world = world

        # Initialize all of the objects
        self.world.gravity = (0,-100)

        # And a static body to hold the ground shape
        self.ground_body=self.world.CreateStaticBody(
            position=(0.0,0.0),
            shapes=b2.polygonShape(box=(5.0,1.0)),
            )

        self.grasp_pivot_body = self.world.CreateDynamicBody(position=(0.0,5.0))
        self.grasp_pivot_body.fixedRotation = True

        self.grasp_body = self.world.CreateDynamicBody(position=(0.0,5.0), angle=0.0)
        self.grasp_fixture = self.grasp_body.CreatePolygonFixture(box=(.2,.2), density=1.0) #density is required so that torques make the body rotate.
        self.grasp_fixture.sensor = True #it doesn't collide with anything

        if self.gripper_direct_control:
            self.grasp_body.gravityScale = 0.0 #we will velocity/position/force control this ourselves
            #forces on the order of manipuland weight
            #don't move this body because it's so massive and viscous.
            
            self.grasp_body.mass = 1e10
            self.grasp_body.inertia = 1e10

            #if we do get it going with some velocity, make sure it dies fast
            #self.grasp_body.linearDamping = 1e5
            #self.grasp_body.angularDamping = 1e5
        else:
            #translation controller for the gripper
            self.gripper_translation_control = self.world.CreateMouseJoint(
                bodyA=self.ground_body,
                bodyB=self.grasp_pivot_body, 
                target=self.grasp_body.position,
                maxForce=1000.0*self.grasp_body.mass
                )

            self.gripper_translation_control.dampingRatio = 5

            self.grasp_pivot_joint = self.world.CreateRevoluteJoint(
                bodyA=self.grasp_pivot_body,
                bodyB=self.grasp_body,
                anchor=self.grasp_pivot_body.worldCenter)

            self.gripper_rotation_control = AngleJoint(self.grasp_pivot_joint)

        # Create a dynamic body
        self.manipuland_body=self.world.CreateDynamicBody(position=(0.0,5.0), angle=0)

        # And add a box fixture onto it (with a nonzero density, so it will move)
        self.manipuland_fixture=self.manipuland_body.CreatePolygonFixture(box=(.8,.4), density=.1, friction=3.3)

        #self.grasp_rotate_joint = self.world.CreateRevoluteJoint(bodyA=self.grasp_body, bodyB=self.manipuland_body, anchor=self.grasp_body.worldCenter)
        self.grasp_slip_joint = None

        self.create_grasp_slip_joint()
        self.grasp_center = None

        self.sample_manipuland_in_hand_rejection_iterations = None

    def update_sample_manipuland_in_hand_rejection(self,iterations):
        if self.sample_manipuland_in_hand_rejection_iterations is None:
            self.sample_manipuland_in_hand_rejection_iterations = iterations
        else:
            self.sample_manipuland_in_hand_rejection_iterations = (
                .98 * self.sample_manipuland_in_hand_rejection_iterations +
                .02 * iterations)

    def step_grasp(self):
        if self.grasp_slip_joint is not None:
            if Box2D.b2TestOverlap(self.grasp_fixture.shape,0,
                                    self.manipuland_fixture.shape,0,
                                    self.grasp_body.transform,
                                    self.manipuland_body.transform):

                intersection = Box2D.Vec2Vector()


                Box2D.b2Fixture.findIntersectionOfFixtures(self.grasp_fixture,self.manipuland_fixture, intersection)
                #print intersection.size()
                if intersection.size() >= 3: 
                    self.grasp_center = Box2D.b2Fixture.CenterOfFixtureIntersection(self.grasp_fixture,self.manipuland_fixture)
                else:
                    #it overlaps, but the polygon intersection is degenerate.
                    #just keep the last grasp center. the object will probably slip out soon
                    pass
            else:
                self.grasp_center = None

            if self.grasp_center is None:
                self.destroy_grasp_slip_joint()
            else:
                self.grasp_slip_joint.SetAnchor(self.grasp_center)


    def is_valid_manipuland_state(self,state):
        assert len(state)==6

        pos = (state[0],state[1])
        ang = state[2]

        t = Box2D.b2Transform(pos,Box2D.b2Rot(ang))
        #t dictates the world-relative position of the box

        ground_fixture = self.ground_body.fixtures[0]
        accept = ( Box2D.b2TestOverlap(manipuland_shape,0,
                                    gripper_shape,0,
                                    manipuland_proposal_transform,
                                    b2I)
                    and not
                    #if there are more obstacles in the world, this doesn't handle it yet
                    Box2D.b2TestOverlap(manipuland_shape,0,
                        ground_fixture.shape,0,
                        t,
                        self.ground_body.transform 
                        ) )
        return accept

    def sample_manipuland_in_hand(self,not_collide_ground=True):
        gripper_shape = self.grasp_fixture.shape

        manipuland_shape = self.manipuland_fixture.shape


        manipuland_radius = np.max(
                                np.linalg.norm( 
                                    np.array(self.manipuland_fixture.shape.vertices) - [0,0],
                                    axis=1))

        b2I = Box2D.b2Transform()
        b2I.SetIdentity()

        #grow out AABB for gripper by radius of manipuland
        aabb = self.grasp_fixture.shape.getAABB(b2I,0)

        aabb.lowerBound += (-manipuland_radius,-manipuland_radius)
        aabb.upperBound += (manipuland_radius,manipuland_radius)


        delta = (aabb.upperBound - aabb.lowerBound)

        for i in xrange(1000):
            xy = aabb.lowerBound +  (delta[0] * random.random(), delta[1] * random.random())
            theta = Box2D.b2Rot(random.random() * 2*np.pi)

            manipuland_proposal_transform = Box2D.b2Transform(xy,theta)


            t2 = self.grasp_body.transform
            t1 = manipuland_proposal_transform

            pos = t1.position+t2.position
            ang = t1.angle + t2.angle

            t = Box2D.b2Transform(pos,Box2D.b2Rot(ang))
            #t dictates the world-relative position of the box

            ground_fixture = self.ground_body.fixtures[0]
            accept = ( Box2D.b2TestOverlap(manipuland_shape,0,
                                        gripper_shape,0,
                                        manipuland_proposal_transform,
                                        b2I)
                        and not
                        #if there are more obstacles in the world, this doesn't handle it yet
                        Box2D.b2TestOverlap(manipuland_shape,0,
                            ground_fixture.shape,0,
                            t,
                            self.ground_body.transform 
                            ) )

            if accept:
                #t * vec first rotates by t.angle then translates by t.position

                #compose the two transforms together
                #t = self.grasp_body.transform * manipuland_proposal_transform 


                self.update_sample_manipuland_in_hand_rejection(i)

                return t


        raise AssertionError('Rejection Sampling Failed')



    def create_grasp_slip_joint(self):
        if self.grasp_slip_joint is not None:
            self.destroy_grasp_slip_joint()
        assert self.grasp_slip_joint is None
        #return world.CreateRevoluteJoint(bodyA=self.grasp_body,bodyB=self.manipuland_body anchor=grasp_body.worldCenter+(0,0))
        #return world.CreateWeldJoint(bodyA=self.grasp_body,bodyB=self.manipuland_body)
        a = self.world.CreateFrictionJoint(bodyA=self.grasp_body,bodyB=self.manipuland_body)
        a.maxForce = 50.0
        a.maxTorque = 50.0
        self.grasp_slip_joint = a

    def destroy_grasp_slip_joint(self):
        self.world.DestroyJoint(self.grasp_slip_joint)
        self.grasp_slip_joint = None


class Empty(Framework):
    """You can use this class as an outline for your tests.

    """
    name = "Empty" # Name of the class to display
    description="The description text goes here"
    def __init__(self):
        """ 
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(Empty, self).__init__()

        self.dynamics = Dynamics(world=self.world,gripper_direct_control=False)


        self.memory = {}

        # x,y,theta,vx,vy,omega
        self.particles = np.zeros((100,6))



    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """

        keymove = {Keys.K_w:(0,0.05),Keys.K_s:(0,-0.05),Keys.K_a:(-0.05,0),Keys.K_d:(0.05,0)}

        if key in keymove:
            dx,dy = keymove[key]
            self.dynamics.gripper_translation_control.target += (dx,dy)
        
        elif key == Keys.K_g:
            if self.dynamics.grasp_slip_joint is None:
                self.dynamics.create_grasp_slip_joint() 
            else:
                self.dynamics.destroy_grasp_slip_joint()

        elif key == Keys.K_q:
            self.dynamics.gripper_rotation_control.target += .1
        elif key == Keys.K_e:
            self.dynamics.gripper_rotation_control.target -= .1

        body_store = [self.dynamics.manipuland_body]
        #body_store = self.world.bodies
        if key == Keys.K_p:
            for body in body_store:
                self.memory[body] = {}
                self.memory[body]['pos']=body.position.__copy__()
                self.memory[body]['linvel']=body.linearVelocity.__copy__()
                self.memory[body]['angle']=body.angle
                self.memory[body]['angvel']=body.angularVelocity

            if self.dynamics.grasp_slip_joint is not None:
                self.memory['grasp_slip_lin'] = self.dynamics.grasp_slip_joint.GetLinearImpulse()
                self.memory['grasp_slip_ang'] = self.dynamics.grasp_slip_joint.GetAngularImpulse()

        elif key == Keys.K_o:
            for body in body_store:
                if body in self.memory:
                    body.position = self.memory[body]['pos']
                    body.linearVelocity = self.memory[body]['linvel']
                    body.angle = self.memory[body]['angle']
                    body.angularVelocity = self.memory[body]['angvel']
                else:
                    print 'not in memory: ', body

            if self.dynamics.grasp_slip_joint is not None:
                self.dynamics.grasp_slip_joint.SetLinearImpulse( self.memory['grasp_slip_lin'])
                self.dynamics.grasp_slip_joint.SetAngularImpulse( self.memory['grasp_slip_ang'])

        elif key == Keys.K_t:
            if self.dynamics.grasp_slip_joint is not None:
                self.dynamics.grasp_slip_joint.SetAngularImpulse(.5)

        elif key == Keys.K_f:
            if self.dynamics.grasp_slip_joint is not None:
                self.dynamics.grasp_slip_joint.SetAngularImpulse(
                    -1.0 * self.dynamics.grasp_slip_joint.GetAngularImpulse())

        elif key == Keys.K_m:
            self.dynamics.grasp_body.linearVelocity += b2Vec2(0,1.0)
        elif key == Keys.K_n:
            self.dynamics.grasp_body.linearVelocity += b2Vec2(0,-1.0)
         
    def Step(self, settings):
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """

        phy2pix = self.world.renderer.to_screen
        timeStep = 1.0/settings.hz

        if not self.dynamics.gripper_direct_control:
            self.dynamics.gripper_rotation_control.solve()
        self.dynamics.step_grasp()

        #t = self.dynamics.sample_manipuland_in_hand()

        #self.dynamics.manipuland_body.transform = (t.position, t.angle )
        #self.dynamics.manipuland_body.position = t.position.copy()
        #self.dynamics.manipuland_body.angle = t.angle

        #self.dynamics.manipuland_body.velocity = (0,0)
        #self.dynamics.manipuland_body.angularVelocity = 0

        #self.renderer.flags['drawShapes'] = True
        #self.renderer.flags['convertVertices']=isinstance(self.renderer, b2DrawExtended)

        #if self.renderer:
        #    self.renderer.StartDraw()

        #self.world.DrawDebugData()
        #do physics and conventional plotting
        super(Empty, self).Step(settings)

        self.Print(str(self.dynamics.sample_manipuland_in_hand_rejection_iterations))

        if self.dynamics.grasp_slip_joint is not None:
            c = self.dynamics.grasp_slip_joint.anchorA #or anchorB
            c = self.dynamics.grasp_center
            force = self.dynamics.grasp_slip_joint.GetReactionForce(timeStep)
            self.renderer.DrawPoint(phy2pix(c), 
                            2, b2Color(1,0,0) ) 


            self.renderer.DrawSegment(phy2pix(c),
                phy2pix(c + force * 50),
                b2Color(1,1,0)
                )

            torque = self.dynamics.grasp_slip_joint.GetReactionTorque(timeStep)

            torque *= 400
            s = np.linspace(0,torque,int(50*abs(torque)))

            if len(s) > 1:
                r = (30 + 3*s)
                u = np.c_[np.cos(s),np.sin(s)]
                xy = u * r[:,None] #add a singleton index to broadcast correctly

                xy += phy2pix(c)
                torque_color = (255,0,0,127) if torque >0 else (0,255,0,127)
                pygame.draw.lines(self.world.renderer.surface,
                    torque_color,
                    False, #not closed
                    xy, #point list
                    1) #width

        #draw manipuland velocity
        c = self.dynamics.manipuland_body.position

        self.renderer.DrawSegment(phy2pix(c),
                phy2pix(c + .2 * self.dynamics.manipuland_body.linearVelocity ),
                b2Color(.5,.5,0)
                )

        def add(a,b):
            return tuple( [x+y for (x,y) in zip(a,b)] )

        def vint(a):
            return tuple([int(x) for x in a])

        #draw gripper setpoint
        if not self.dynamics.gripper_direct_control:
            self.world.renderer.DrawCircle(phy2pix(self.dynamics.gripper_translation_control.target),
                10.0/self.world.renderer.zoom,
                b2Color(1,1,1),
                )



            r = 10.0
            a = self.dynamics.gripper_rotation_control.target
            self.world.renderer.DrawCircle(
                vint(add(
                    phy2pix(self.dynamics.gripper_translation_control.target),
                    (r*np.cos(-a), r*np.sin(-a)) 
                    )),
                3.0/self.world.renderer.zoom,
                b2Color(0,1,1),
                )

        # Placed after the physics step, it will draw on top of physics objects
        self.Print("*** Base your own testbeds on me! ***")

    def ShapeDestroyed(self, shape):
        """
        Callback indicating 'shape' has been destroyed.
        """
        pass

    def JointDestroyed(self, joint):
        """
        The joint passed in was removed.
        """
        pass

    # More functions can be changed to allow for contact monitoring and such.
    # See the other testbed examples for more information.

if __name__=="__main__":
    #main(Empty)
    domain = Empty()
    #domain.settings.enableWarmStarting=False
    domain.setCenter(domain.dynamics.manipuland_body.worldCenter)
    #domain.run()
    domain.run_init()

    d = domain.dynamics
    s = Dynamics(gripper_direct_control=True)
    f = Filter(s,d)

    def run_from_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False


    from inputhookpygame import PyGame_InputHook


    if False or not run_from_ipython():
        while domain.running:
            domain.run_step()
            f.apply_transition()
    else:
        from IPython.lib import inputhook

        #ihm = inputhook.InputHookManager()

        #old_hook = inputhook.clear_inputhook()
        #print old_hook

        def step():
            #old_hook()
            domain.run_step()
            f.apply_transition()
            return domain.running

        pyih = PyGame_InputHook(step)

        inputhook.set_inputhook(pyih.inputhook_pygame)




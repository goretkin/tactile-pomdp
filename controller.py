from place_framework import *
import numpy as np

class Controller():
    def __init__(self,domain):
        self.domain = domain
        self.dynamics = self.domain.dynamics
        self.state = "nothing"

        self.ref_forcetorque = None
        self.stopping_forcetorque = None

        self.datalog = []

        self.desired_position = None
        self.desired_velocity = None

        self.action_state = None
        self.origin_position = None
        self.origin_time = None

        self.stopped = True

        self.desired_forcetorque = None



        self.gain_matrix = .1 * np.eye(3,3) #multiplies error in force_x,force_y,torque
        self.gain_matrix[2,:] = 0   #don't do anything with torque

        self.gravity_bias = True #applied forces biased to undo weight of manipulandum if it's grasped.

        self.damping_matrix = None 
        self.spring_matrix = None

    def reset(self):
        self.ref_forcetorque = None
        self.stopping_forcetorque = None
        self.origin_position = None
        self.origin_time
        self.datalog = []

    def stop_move(self):
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,0)
        self.dynamics.grasp_body.angularVelocity = 0
        self.desired_position = None
        self.action_state = None

    def move_down(self):
        self.engage_guard()
        self.desired_position = None
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,-1.0)

    def engage_position_control(self):
        self.desired_forcetorque = None
        self.desired_velocity = None
        self.desired_position = np.array([ self.dynamics.grasp_body.position[0],
                                            self.dynamics.grasp_body.position[1],
                                            self.dynamics.grasp_body.angle])
    def engage_guard(self):
        self.ref_forcetorque = np.array(self.domain.forcetorque_measurement)
        self.state = "guarded"
        self.datalog.append(self.ref_forcetorque)

    def guarded_damper(self):
        self.engage_guard()
        self.velocity_full()

    def senseact(self):
        """
        called by the simulation loop
        """
        if self.state == "guarded":
            this_forcetorque = np.array(self.domain.forcetorque_measurement)
            delta = this_forcetorque - self.ref_forcetorque
            self.datalog.append(this_forcetorque)

            if np.linalg.norm(delta) > 1e-2:
                self.position_full()
                self.state = "nothing"
                self.stopping_forcetorque = this_forcetorque
                print("Controller: stop")


        body = self.dynamics.grasp_body
        if self.spring_matrix is not None:
            self.position = np.zeros((3,))
            self.position[0:2] = body.position
            self.position[2] = body.angle

            self.Fspring = self.spring_matrix.dot(self.position - self.desired_position)
        else:
            self.Fspring = np.zeros((3,))

        if self.damping_matrix is not None:
            self.velocity = np.zeros((3,))
            self.velocity[0:2] = body.linearVelocity
            self.velocity[2] = body.angularVelocity

            self.Fdamping = self.damping_matrix.dot(self.velocity - self.desired_velocity)
        else:
            self.Fdamping = np.zeros((3,))

        self.FTcommand = self.Fdamping + self.Fspring 
        body.ApplyForceToCenter(self.FTcommand[0:2], True)
        body.ApplyTorque(self.FTcommand[2], True)


    def position_full(self,set_current=True):
        if set_current:
            self.desired_position = np.zeros((3,))
            self.desired_position[0:2] = self.dynamics.grasp_body.position
            self.desired_position[2] = self.dynamics.grasp_body.angle

        m = self.dynamics.grasp_body.mass

        # m in kg
        # x in m
        # k in N/m = kg/s^2
        # c in N*s/m = kg/s

        # mx'' = - k * x - c * x'      

        self.desired_velocity = np.zeros((3,))
        self.damping_matrix = np.eye(3,3)
        self.spring_matrix = np.eye(3,3)

        freq = 10 #in 1/s
        damp_ratio = .7

        k = freq**2 * m
        c = damp_ratio * 2 * np.sqrt(k*m)

        self.damping_matrix[0:2,0:2] *= -c
        self.spring_matrix[0:2,0:2] *= -k 

        I = self.dynamics.grasp_body.inertia

        k = freq**2 * I
        c = damp_ratio * 2 * np.sqrt(k*I)

        self.damping_matrix[2,2] *= -c
        self.spring_matrix[2,2] *= -k 

    def velocity_full(self):
        self.spring_matrix = None
        self.damping_matrix = -1. * np.eye(3,3)


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

        self.position_control = None
        self.K_pos = 1.0
        self.K_ang = 0.5

        self.action_state = None
        self.origin_position = None
        self.origin_time = None

        self.stopped = True

        self.desired_forcetorque = None
        self.desired_velocity = None

        self.gain_matrix = .1 * np.eye(3,3) #multiplies error in force_x,force_y,torque
        self.gain_matrix[2,:] = 0   #don't do anything with torque


    def reset(self):
        self.ref_forcetorque = None
        self.stopping_forcetorque = None
        self.origin_position = None
        self.origin_time
        self.datalog = []

    def stop_move(self):
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,0)
        self.dynamics.grasp_body.angularVelocity = 0
        self.position_control = None
        self.action_state = None

    def move_down(self):
        self.engage_guard()
        self.position_control = None
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,-1.0)

    def engage_position_control(self):
        self.position_control = np.array([ self.dynamics.grasp_body.position[0],
                                            self.dynamics.grasp_body.position[1],
                                            self.dynamics.grasp_body.angle])
    def engage_guard(self):
        self.ref_forcetorque = np.array(self.domain.forcetorque_measurement)
        self.state = "guarded"
        self.datalog.append(self.ref_forcetorque)


    def senseact(self):
        """
        called by the simulation loop
        """
        if self.position_control is not None:
            if self.dynamics.gripper_direct_control:
                errpos = np.array(self.dynamics.grasp_body.position) - self.position_control[0:2]
                errang = self.dynamics.grasp_body.angle - self.position_control[2]

                linearVelocity = -self.K_pos * errpos
                angularVelocity = -self.K_ang * errang                

                angularVelocity = np.clip(angularVelocity,-0.5,0.5)
                linearVelocity = np.clip(linearVelocity,-2.0,2.0)

                self.dynamics.grasp_body.linearVelocity = linearVelocity
                self.dynamics.grasp_body.angularVelocity = angularVelocity

        elif self.desired_forcetorque is not None:
            #ft sensor measures the reaction force. 
            #So if gravity is pulling down, the ft reports "up"
            this_forcetorque = np.array(self.domain.forcetorque_measurement)
            err = self.desired_forcetorque - this_forcetorque
            forcetorque_control = self.gain_matrix.dot(err)

            self.dynamics.grasp_body.linearVelocity = forcetorque_control[0:2]
            self.dynamics.grasp_body.angularVelocity = forcetorque_control[2]

            self.forcetorque_control = forcetorque_control #just to make it visible to debug

        elif self.desired_velocity is not None: 
            vl = self.dynamics.grasp_body.linearVelocity
            va = self.dynamics.grasp_body.angularVelocity

            this_velocity = np.array([vl[0],vl[1],va])
            err = self.desired_velocity - this_velocity
            forcetorque_control = self.gain_matrix.dot(err)
            
            mass = self.dynamics.grasp_body.mass
            inertia = self.dynamics.grasp_body.inertia

            self.dynamics.grasp_body.ApplyForceToCenter(forcetorque_control[0:2] * mass,True)
            self.dynamics.grasp_body.ApplyTorque(forcetorque_control[2] * inertia,True)
            
            self.forcetorque_control = forcetorque_control #just to make it visible to debug

        else:
            pass 

        if self.state == "nothing":
            pass
        elif self.state == "guarded":
            this_forcetorque = np.array(self.domain.forcetorque_measurement)
            delta =  this_forcetorque - self.ref_forcetorque
            self.datalog.append(this_forcetorque)

            if np.linalg.norm(delta) > 1e-2:
                self.stop_move()
                self.state = "nothing"
                self.stopping_forcetorque = this_forcetorque
                print("Controller: stop")
                self.position_control = None
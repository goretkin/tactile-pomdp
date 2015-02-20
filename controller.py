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

        self.gravity_bias = True #applied forces biased to undo weight of manipulandum if it's grasped.


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

    def senseact(self):
        """
        called by the simulation loop
        """
        if self.desired_position is not None:
            if self.dynamics.gripper_direct_control:
                errpos = np.array(self.dynamics.grasp_body.position) - self.desired_position[0:2]
                errang = self.dynamics.grasp_body.angle - self.desired_position[2]

                linvel = self.dynamics.grasp_body.linearVelocity
                angvel = self.dynamics.grasp_body.angularVelocity

                #control velocity directly
                #linearVelocity = -self.K_pos * errpos
                #angularVelocity = -self.K_ang * errang                

                #angularVelocity = np.clip(angularVelocity,-0.5,0.5)
                #linearVelocity = np.clip(linearVelocity,-2.0,2.0)

                #self.dynamics.grasp_body.linearVelocity = linearVelocity
                #self.dynamics.grasp_body.angularVelocity = angularVelocity

                #control force
                m = self.dynamics.grasp_body.mass

                # m in kg
                # x in m
                # k in N/m = kg/s^2
                # c in N*s/m = kg/s

                # mx'' = - k * x - c * x'

                freq = 10 #in 1/s
                damp_ratio = .7

                k = freq**2 * m
                c = damp_ratio * 2 * np.sqrt(k*m)

                commanded_force = - k * errpos + -c * linvel
                self.dynamics.grasp_body.ApplyForceToCenter(commanded_force, True)

                I = self.dynamics.grasp_body.inertia

                k = freq**2 * I
                c = damp_ratio * 2 * np.sqrt(k*I)

                commanded_torque = - k * errang + -c * angvel
                self.dynamics.grasp_body.ApplyTorque(commanded_torque, True)

        elif self.desired_velocity is not None: 
            vl = self.dynamics.grasp_body.linearVelocity
            va = self.dynamics.grasp_body.angularVelocity

            this_velocity = np.array([vl[0],vl[1],va])
            err = this_velocity - self.desired_velocity
            forcetorque_control = -self.gain_matrix.dot(err)
            
            inertia = self.dynamics.grasp_body.inertia

            if self.gravity_bias:
                forcetorque_control[0:2] -= self.dynamics.world.gravity * self.dynamics.grasp_body.mass

                if self.dynamics.grasp_slip_joint is not None:
                    forcetorque_control[0:2] -= self.dynamics.world.gravity * self.dynamics.manipuland_body.mass

            self.dynamics.grasp_body.ApplyForceToCenter(forcetorque_control[0:2] ,True)
            self.dynamics.grasp_body.ApplyTorque(forcetorque_control[2],True)
            
            self.forcetorque_control = forcetorque_control #just to make it visible to debug


        elif self.desired_forcetorque is not None:
            #ft sensor measures the reaction force. 
            #So if gravity is pulling down, the ft reports "up"
            this_forcetorque = np.array(self.domain.forcetorque_measurement)
            err = self.desired_forcetorque - this_forcetorque
            forcetorque_control = -self.gain_matrix.dot(err)

            #self.dynamics.grasp_body.linearVelocity = forcetorque_control[0:2]
            #self.dynamics.grasp_body.angularVelocity = forcetorque_control[2]

            self.dynamics.grasp_body.ApplyForceToCenter(forcetorque_control[0:2],True)
            self.dynamics.grasp_body.ApplyTorque(forcetorque_control[2],True)
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
                self.desired_position = None
from place_framework import *


class Controller():
    def __init__(self,domain):
        self.domain = domain
        self.dynamics = self.domain.dynamics
        self.state = "nothing"

        self.ref_forcetorque = None
        self.stopping_forcetorque = None

        self.datalog = []

    def reset(self):
        self.ref_forcetorque = None
        self.stopping_forcetorque = None
        self.datalog = []

    def stop_move(self):
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,0)
        self.dynamics.grasp_body.angularVelocity = 0



    def move_down(self):
        self.ref_forcetorque = np.array(self.domain.forcetorque_measurement)
        self.dynamics.grasp_body.linearVelocity = b2Vec2(0,-1.0)
        self.state = "guarded"
        self.datalog.append(self.ref_forcetorque)


    def senseact(self):
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
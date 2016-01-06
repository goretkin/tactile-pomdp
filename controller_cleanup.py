import numpy as np

def pose_of_body(body):
    p = np.zeros((3,))
    p[0:2] = body.position
    p[2] = body.angle
    return p

def set_pose_of_body(body, pose):
    body.position = pose[0:2]
    body.angle = pose[2]

class Controller():
    def __init__(self,domain):
        self.domain = domain
        self.dynamics = self.domain.dynamics
        self.state = "nothing"

        self.ref_forcetorque = None
        self.stopping_forcetorque = None

        self.datalog = []

        self.action_state = None
        self.origin_position = None
        self.origin_time = None

        self.stopped = True

        self.desired_forcetorque = None

        self.desired_position = np.zeros((3,))

        self.spring_matrix = .1 * np.eye(3,3) #multiplies error in force_x,force_y,torque
        self.spring_matrix[2,:] = 0   #don't do anything with torque
        self.damping_matrix = None

        self.Fconstant = np.zeros((3,)) # force bias.

        # spring constants to use for critically damped behavior
        # for the case when there is no friction.
        m = self.dynamics.grasp_body.mass
        # have to multiply by m because Box2D damping term isn't the coefficient b in mx'' = - bx'
        # but instead the coefficient in x'' = -bx'
        ld = m*self.dynamics.grasp_body.linearDamping
        self.linear_k = (ld**2)/4

        I = self.dynamics.grasp_body.inertia
        ad = I*self.dynamics.grasp_body.angularDamping
        self.angular_k = (ad**2)/4 * 100
        # the *100 is a hack. it has to do with the damping value of 50 being inaccurate at dt = 1/60hz
        # the most recent version of Box2D doesn't have this problem

        self.simtime_guard = None #if set, when self.domain.simtime>=simtime_guard, then stop moving the setpoint

        self.compliance_frame = np.eye(3)

    def reset(self):
        self.ref_forcetorque = None
        self.stopping_forcetorque = None
        self.origin_position = None
        self.origin_time
        self.datalog = []

    def engage_force_guard(self):
        self.ref_forcetorque = np.array(self.domain.forcetorque_measurement)
        self.state = "guarded"
        self.datalog.append(self.ref_forcetorque)

    def set_compliance(self, is_stiff):
        # axis order is "x", "y", "theta"
        if not len(is_stiff) == 3:
            raise ValueError(is_stiff)

        mask = np.diag(is_stiff)

        setpoint_position = pose_of_body(self.dynamics.setpoint_body)
        current_position = pose_of_body(self.dynamics.grasp_body)

        # before turning on compliance in some axis, place setpoint so that there is
        # zero error along that axis.
        # (the _cf prefix means that the quantity is expressed in the compliance frame)
        current_position_cf = np.dot(self.compliance_frame, current_position)
        setpoint_position_cf = np.dot(self.compliance_frame, setpoint_position)

        new_setpoint_position_cf = (    (np.eye(3) - mask).dot(setpoint_position_cf) +
                                        mask.dot(current_position_cf) )

        new_setpoint_position = np.dot(self.compliance_frame.T, new_setpoint_position_cf)
        set_pose_of_body(self.dynamics.setpoint_body, new_setpoint_position)

        # set spring matrix
        spring_cf = np.dot(mask, np.diag([-self.linear_k, -self.linear_k, -self.angular_k]))
        spring =  np.dot(self.compliance_frame.T, spring_cf)

        self.spring_matrix[:,:] = spring

        print("mask", mask)

    def move(self, twist_velocity, duration_guard=5.0):
        self.dynamics.setpoint_body.linearVelocity = twist_velocity[0:2]
        self.dynamics.setpoint_body.angularVelocity = twist_velocity[2]
        self.simtime_guard = duration_guard + self.domain.simtime

    def stop(self):
        self.dynamics.setpoint_body.angularVelocity = 0.0
        self.dynamics.setpoint_body.linearVelocity = (0.0, 0.0)

    def senseact(self):
        """
        called by the simulation loop
        """
        body = self.dynamics.grasp_body
        setpoint = self.dynamics.setpoint_body

        self.desired_position[0:2] = setpoint.position
        self.desired_position[2] = setpoint.angle

        if self.simtime_guard is not None:
            if self.simtime_guard<=self.domain.simtime:
                self.stop()
                self.simtime_guard = None

        if self.state == "guarded":
            this_forcetorque = np.array(self.domain.forcetorque_measurement)
            delta = this_forcetorque - self.ref_forcetorque
            self.datalog.append(this_forcetorque)

            if np.linalg.norm(delta[0:2]) > 5.0 or np.linalg.norm(delta[2]) > 1e+1:
                self.stop()
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

        self.FTcommand = self.Fdamping + self.Fspring + self.Fconstant
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


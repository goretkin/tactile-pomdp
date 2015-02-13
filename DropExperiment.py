from place_framework import *

#makes sure that time, position, and gravity are all consistent.
class DropExperiment():
    def __init__(self,domain):
        #this assumes the object starts on the ground and that there is a column of free space above the object.
        self.domain = domain
        self.dynamics = self.domain.dynamics
        self.droptimes = []
        self.dropheights = []

        self.dynamics.last_contact = None
        self.release_time = None
        self.waiting_for_contact = False

        self.wait_steps = 5
    def senseact(self):
        """
        called by the simulation loop
        """
        max_drops = 100
        n_drops = len(self.droptimes)
        remaining_drops = max_drops - n_drops

        if n_drops < max_drops:
            if self.waiting_for_contact:
                if self.dynamics.last_contact:
                    #there has been a contact. assume it was the one we cared about
                    flight_time = self.dynamics.last_contact[0] - self.release_time
                    self.waiting_for_contact = False

                    self.droptimes.append(flight_time)

                else:
                    pass #waiting for object to hit ground still.
            else:
                if self.wait_steps > 0:
                    self.wait_steps -= 1
                else: 
                    #should drop object, it should currently be resting on the ground.
                    dropheight = (n_drops+1) * .1  #move in .1 meter increments
                    self.dynamics.manipuland_body.position += (0,dropheight)
                    self.dropheights.append(dropheight)
                    self.dynamics.last_contact = None
                    self.release_time = self.domain.simtime
                    self.dynamics.manipuland_body.awake = True #body won't wake up if you just set its position
                    self.waiting_for_contact = True

                    self.wait_steps = 5
        else:
            self.dropheights = np.array(self.dropheights)
            self.droptimes = np.array(self.droptimes)

            inverse_g = np.mean(self.droptimes**2 / self.dropheights) / 2.0
            
            self.g_estimate = 1/inverse_g
            pass #experiment is done








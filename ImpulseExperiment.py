from place_framework import *

#makes sure that mass, gravity, time, impulse consistent
#except that energy isn't being conserved, 

#if initial impulse is not strong enough to make the obstacle jump up, the experiment will hang
class ImpulseExperiment():
    def __init__(self,domain):
        #this assumes the object starts on the ground and that there is a column of free space above the object.
        self.domain = domain
        self.dynamics = self.domain.dynamics
        self.droptimes = []
        self.impulses = []
        self.start_heights = []

        self.dynamics.last_contact = None
        self.release_time = None
        self.waiting_for_contact = False

        self.wait_steps = 5

        #the following store time series
        self.velocities = {}
        self.heights = {}
        self.simtimes = {} 

        self.velocity = None #stores falling velocity profile for current drop
        self.height = None #ditto
        self.simtime = None

        self.experiment_done = False

    def senseact(self):
        """
        called by the simulation loop
        """
        if self.experiment_done:
            return


        max_drops = 20
        n_drops = len(self.droptimes)
        remaining_drops = max_drops - n_drops   #decrements by 1 everytime the object makes contact

        if n_drops < max_drops:
            if self.waiting_for_contact:
                self.velocity.append(self.dynamics.manipuland_body.linearVelocity[1]) #falling velocity
                self.height.append(self.dynamics.manipuland_body.position[1]) #height

                if self.dynamics.last_contact:
                    #there has been a contact. assume it was the one we cared about
                    flight_time = self.dynamics.last_contact[0] - self.release_time
                    
                    self.droptimes.append(flight_time)
                    self.velocity = None
                    self.height = None

                    self.waiting_for_contact = False

                else:
                    pass #waiting for object to hit ground still.
            else:
                if self.wait_steps > 0:
                    self.wait_steps -= 1
                else: 
                    manipuland_body = self.dynamics.manipuland_body
                    #shoot impulse object, it should currently be resting on the ground.
                    impulse = (n_drops+1) * .5  #Newton-seconds

                    #the impulse immediately changes the velocity of the body. 
                    #By the next iteration, the velocity will already change due to gravity
                    manipuland_body.ApplyLinearImpulse((0,impulse), manipuland_body.worldCenter,
                                                        True) #wake the body

                    self.impulses.append(impulse)
                    self.start_heights.append(manipuland_body.position[1])

                    self.dynamics.last_contact = None
                    self.release_time = self.domain.simtime
                    
                    
                    self.velocity = self.velocities[impulse] = []
                    self.height = self.heights[impulse] = []

                    self.wait_steps = 5

                    self.waiting_for_contact = True
        else:
            self.impulses = np.array(self.impulses)
            self.droptimes = np.array(self.droptimes)

            self.velocities = {k:np.array(self.velocities[k]) for k in self.velocities.keys()}
            self.heights = {k:np.array(self.heights[k]) for k in self.heights.keys()}

            #these are actually the velocities after one time step.
            self.initial_velocities_expected = self.impulses / self.dynamics.manipuland_body.mass - (1.0/60.0 * 100.0)
            self.initial_velocities = np.array([ self.velocities[k][0] for k in self.impulses])

            self.appogees = np.array([ np.max( self.heights[k] ) for k in self.impulses])  

            g = self.dynamics.world.gravity[1]
            m = self.dynamics.manipuland_body.mass

            self.PEmax = -m*g*(self.appogees - self.start_heights)
            self.KEmax = .5 *m*(self.impulses / self.dynamics.manipuland_body.mass)**2

            self.PEs = [ -m*g*(self.heights[k] - self.start_heights[i]) for (i,k) in enumerate(self.impulses) ]
            self.KEs = [ .5*m*(self.velocities[k])**2 for k in self.impulses ]
            self.experiment_done = True



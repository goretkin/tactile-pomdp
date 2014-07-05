import numpy as np
from sets import Set
class PlanarPolygonObjectInCorner():
    def __init__(self,vertex_list=None):
        
        if vertex_list is None:
            vertex_list = [ [1,1], [1,-1], [-1,-1], [-1,1] ]
      
        self.num_vertices = len(vertex_list)
        #self.edge_list = [(i,i+i) for i in range(self.num_vertices-1)] + [(self.num_vertices-1,0)]
    
        self.vertex_list_original = np.copy(vertex_list)
        
        self.wall_x = 0
        self.wall_y = 0
        
        self.vertex_list = None
        self.pose = None
        
        self.restore()

    def set_pose(self,pose):
        self.restore()
        self.move(*pose)
        
    def get_pose(self):
        return self.pose  
        
    def move(self,x,y,theta):
        rot = [[np.cos(theta), np.sin(theta)],
               [-np.sin(theta), np.cos(theta)]]
        
        xy_c = self.get_pose()[0:2]
        self.vertex_list = np.dot(self.vertex_list-xy_c,rot) + xy_c + np.array([x,y])
        self.pose = self.pose + np.array([x,y,theta])
        
    def restore(self):
        self.pose = np.array([0,0,0])
        self.vertex_list = np.copy(self.vertex_list_original)
                                   
    def translate_guarded(self,dx,dy):
        #translate object in direction dx dy until collision with wall
        
        left_vertex = np.argmin(self.vertex_list[:,0])
        bottom_vertex = np.argmin(self.vertex_list[:,1])
        
        left_x = self.vertex_list[left_vertex,0]
        bot_y = self.vertex_list[bottom_vertex,1]
        
        # the point that will contact the wall moves like this as a function of time increasing.
        #wall_x = left_x + t*dx
        #wall_y = bot_y + t*dy
        
        if dx != 0:
            t1 = -(left_x - self.wall_x)/dx
        else:
            t1 = np.NaN
        
        if dy != 0:
            t2 = -(bot_y - self.wall_y)/dy
        else:
            t2 = np.NaN
        
        t = np.nanmin([t1,t2]) #earliest collision time
        
        #print t,t1,t2,t*dx,t*dy
        
        if t<0:
            raise ValueError('moving in direction with no collision')
        self.move(t*dx,t*dy,0)

    def r_bounding_circle(self):
        xy = self.vertex_list
        xy_c = self.get_pose()[0:2]
        dxy = (xy - xy_c)
        
        r = (dxy[:,0]**2 + dxy[:,1]**2)**.5
        return np.max(r) #max over the radius of all vertices
        
    def rotate_guarded(self,theta):
        if self.contact_torque() * theta < 0:
            return #trying to move into a contact.
        
        #xc,yc is center of object
        #dx_i,dy_i is the displacement from rotation center of vertex i
        
        xy = self.vertex_list
        xy_c = self.get_pose()[0:2]
        dxy = (xy - xy_c)
        
        r = self.r_bounding_circle()
        
        vertical_distance = xy_c[1]/r
        angle = np.arcsin(-vertical_distance)
        
        angles_intersection = np.array([angle, (np.pi/2-angle)+np.pi/2 ])
        
        angle_vertices = np.arctan2(dxy[:,1],dxy[:,0])
        
        def circle_dist_eq(frm,to,direction):
            d = circle_dist(frm,to,direction)
            if d < 1e-8:
                d = 2*np.pi
            return d
            
        if theta>0:
            dangles = [circle_dist_eq(frm,to,1) for frm in angle_vertices for to in angles_intersection]
            
        if theta<0:
            dangles = [-circle_dist_eq(frm,to,-1) for frm in angle_vertices for to in angles_intersection]
            
        #plot the path that the corners trace
        if False:
            plt.plot(xy_c[0],xy_c[1],'ko') #center
            
            for a in angles_intersection:
                plt.plot([xy_c[0], xy_c[0]+r*np.cos(a)],
                         [xy_c[1], xy_c[1]+r*np.sin(a)],lw=2)
            
            circ_patch =  matplotlib.patches.Circle(xy_c,r,fc='none')
            plt.gca().add_patch(circ_patch)
        
        t = np.array(dangles)/theta
        
        tp = t[t>=0]
        if len(tp) == 0:
            raise ValueError('no collision')
        t0 = np.nanmin(tp)
        
        self.move(0,0,theta*t0)
        
    def contact_vertices(self):
        contact_x = np.abs(self.vertex_list[:,0] - self.wall_x) < 1e-8
        contact_y = np.abs(self.vertex_list[:,1] - self.wall_y) < 1e-8
        
        #left-wall contacts, bottom-wall contacts
        return np.where(contact_x)[0], np.where(contact_y)[0]

    def contact_torque(self):
        leftcontact, bottomcontact = self.contact_vertices()
        xy_c = self.get_pose()[0:2]
        
        moment = np.float64(0.0)
        
        arms = self.vertex_list[bottomcontact,:] - xy_c
        for moments_due_to_bottom in arms[:,0]:
            moment += moments_due_to_bottom
        
        arms = self.vertex_list[leftcontact,:] - xy_c
        for moments_due_to_left  in arms[:,1]:
            moment += -moments_due_to_left
        
        return moment

    def intersects_bottom(self):
        left_vertex = np.argmin(self.vertex_list[:,0])
        bottom_vertex = np.argmin(self.vertex_list[:,1])
        
        left_x = self.vertex_list[left_vertex,0]
        bot_y = self.vertex_list[bottom_vertex,1]
        return bot_y < 0
        
def normang(a):
    while(a>np.pi):
        a = a - 2*np.pi
    while (a<-np.pi):
        a = a + 2*np.pi
    return a

def circle_dist(a1,a2,direction=1):
    a1 = normang(a1)
    a2 = normang(a2)
    
    while a1*direction > a2*direction:
        a2 += 2*np.pi*direction
    #make a2 come after a1 in the appropriate direction
    return (a2-a1)*direction
        

def plot_obj(obj,ax=None,kwline={},kwcontact={}):
    if ax is None:
        ax = plt.gca()
    
    i = range(obj.num_vertices) + [0]
    
    ax.plot(obj.vertex_list[i,0],obj.vertex_list[i,1],'b-',**kwline)
    
    ax.plot(obj.vertex_list[0,0],obj.vertex_list[0,1],'k*',**kwline) #to mark orientation
    
    leftcontact, bottomcontact = obj.contact_vertices()
    
    contacts = Set.union(Set(leftcontact),Set(bottomcontact))
    
    i = list(contacts)
    ax.plot(obj.vertex_list[i,0],obj.vertex_list[i,1],'r.',**kwcontact)
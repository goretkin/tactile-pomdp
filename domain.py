import numpy as np
from sets import Set

"""
we left-multiply points, so the rotation matrix might be transpose of what you expect
"""
def rot_SO2(theta):
    rot = [[np.cos(theta), np.sin(theta)],
       [-np.sin(theta), np.cos(theta)]]
    return np.array(rot)

class PlanarPolygonObjectInCorner():
    def __init__(self,vertex_list=None):
        
        if vertex_list is None:
            #counter-clockwise
            vertex_list = [ [1,1], [1,-1], [-1,-1], [-1,1] ]
      
        self.num_vertices = len(vertex_list)
        #self.edge_list = [(i,i+i) for i in range(self.num_vertices-1)] + [(self.num_vertices-1,0)]
    
        self.vertex_list_original = np.copy(vertex_list)
        #edge 0-1, 1-2, 2-3, 3-0
        self.edge_angles_original = np.array([np.deg2rad(180), np.deg2rad(270), np.deg2rad(0), np.deg2rad(90)])
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
        """
        incremental move
        """
        rot = [[np.cos(theta), np.sin(theta)],
               [-np.sin(theta), np.cos(theta)]]
        
        xy_c = self.get_pose()[0:2]
        self.vertex_list = np.dot(self.vertex_list-xy_c,rot) + xy_c + np.array([x,y])
        self.pose = self.pose + np.array([x,y,theta])

    def get_posed_vertices(self, pose):
        """
        don't change state. just return position of would-be vertices after a set_pose
        """
        x, y, theta = pose
        rot = [[np.cos(theta), np.sin(theta)],
               [-np.sin(theta), np.cos(theta)]]

        return np.dot(self.vertex_list_original,rot) + np.array([x,y])

    def transform_object_point_to_jig_frame(self, pose, points):
        x, y, theta = pose
        rot = rot_SO2(theta)

        if len(points)==0:
            #interpret as an empty list of points
            return np.zeros(shape=(0,2))

        points_np = np.array(points)
        # should make sure points is not a jagged array (dtype would be object)
        if points_np.ndim == 1:
            #there is a single point
            if not points_np.shape == (2,):
                raise ValueError("point is not 2 dim")

        elif points_np.dim == 2:
            if not points_np.shape[1] == 2:
                raise ValueError("points are not 2 dim")
        else:
            raise ValueError("not a point and not a list of points")
        return np.dot(points_np,rot) + np.array([x,y])
        
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
        """
        returns indices into self.vertex_list indicating which vertices are in contact
        with which jig edges
        """
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
        return self.intersects_jig_which()[0]
        
    def intersects_jig(self):
        return any(self.intersects_jig_which())

    def intersects_jig_which(self):
        left_vertex = np.argmin(self.vertex_list[:,0])
        bottom_vertex = np.argmin(self.vertex_list[:,1])
        
        left_x = self.vertex_list[left_vertex,0]
        bot_y = self.vertex_list[bottom_vertex,1]
        return [bot_y < 0+1e-10, left_x < 0+1e-10]

    def get_pose_grounded_vertex(self, manipulandum_vertex_i, grounding_point, angle):
        contact_vertex = self.vertex_list_original[manipulandum_vertex_i,:]
        contact_vertex_to_centroid = np.array([0,0])-contact_vertex
        
        contact_vertex_to_centroid_rotated = np.dot(contact_vertex_to_centroid, rot_SO2(angle))
        centroid = contact_vertex_to_centroid_rotated + grounding_point

        return np.array([centroid[0], centroid[1], angle])
        #cw_vertex_i = (manipulandum_vertex_i+1)%self.num_vertices
        #ccw_vertex_i = (manipulandum_vertex_i-1)%self.num_vertices
        #contact_vertex_to_cw_vertex = self.vertex_list_original[cw_vertex_i,:] - contact_vertex
        #contact_vertex_to_ccw_vertex = self.vertex_list_original[ccw_vertex_i,:] - contact_vertex

    def get_pose_against_edge(self, jig_edge_i, angle, displacement_along_edge):
        if jig_edge_i == 0:
            #bottom edge
            projected_centroid = np.array([0, displacement_along_edge])
            jig_edge_normal = np.array([0,1])
            jig_edge_tangent = np.array([1,0])
        elif jig_edge_i == 1:
            #bottom edge
            projected_centroid = np.array([displacement_along_edge, 0])
            jig_edge_normal = np.array([1,0])
            jig_edge_tangent = np.array([0,1])
        else:
            raise ValueError("jig edge invalid: %s"%jig_edge_i)

        rotated = self.get_posed_vertices((0, 0, angle))
        contact_vertex_i = np.argmin(np.dot(rotated, jig_edge_normal))
        contact_vertex_to_centroid_rotated = np.array([0,0]) - rotated[contact_vertex_i,:]

        #find the displacement from the contact vertex to the centroid, projected along the jig edge we're contacting
        component_along_edge = np.dot(jig_edge_tangent, contact_vertex_to_centroid_rotated)

        ground_point = np.array([0,0]) + jig_edge_tangent*(displacement_along_edge - component_along_edge)

        return self.get_pose_grounded_vertex(contact_vertex_i, ground_point, angle)

    def get_pose_against_corner(self, jig_corner_i, angle):
        if jig_corner_i == 0:
            #bottom edge
            jig_edge1_normal = np.array([0,1])
            jig_edge1_tangent = np.array([1,0])
            jig_edge2_normal = np.array([1,0])
            jig_edge2_tangent = np.array([0,1])
        else:
            raise ValueError("jig corner invalid: %s"%jig_corner_i)

        #total hack that works for this simple jig.
        arbitrary=-10
        pose1 = self.get_pose_against_edge(jig_edge_i=0, angle=angle, displacement_along_edge=arbitrary)
        pose2 = self.get_pose_against_edge(jig_edge_i=1, angle=angle, displacement_along_edge=arbitrary)

        pose = np.array([pose2[0], pose1[1], angle])
        return pose

    def loose_theta(self, constraint_axis, center=(0,0)):
        """
        center is relative to object
        constraint_axis is relative to jig, and goes through the center
        constraint_axis is the guarded direction to move in.
        """
        jig_relative_center_before_translate = self.transform_object_point_to_jig_frame(self.pose, center)
        self.translate_guarded(*constraint_axis)
        jig_relative_center = self.transform_object_point_to_jig_frame(self.pose, center)
        left_contacts, bottom_contacts = self.contact_vertices()
        all_contacts = list(left_contacts) + list(bottom_contacts)
        if len(all_contacts) == 0:
            raise AssertionError("Took a guarded move but there are no contacts")

        #easiest to think about the case where there is a single active contact.
        #TODO incorporate other contacts. see Box2D code for example

        moment_arms = self.vertices[all_contacts] - jig_relative_center
        dots = np.dot(moment_arms, constraint_axis)

        all_contacts_active_idxs = np.where(dots<1e-8)[0]
        moment_arms_active = moment_arms[all_contacts_active_idxs]
        vertices_contact_active = all_contacts[all_contacts_active_idxs]

        if len(moment_arms_active) == 1:
            moment_arm = moment_arms_active[0]
            active_contact_vertex = all_contacts[all_contacts_active_idxs[0]]

            if active_contact in left_contacts:
                active_edge = "left"
            elif active_contact in bottom_contacts:
                active_edge = "bottom"
            else:
                raise AssertionError("which edge?")

            if np.dot(moment_arm, constraint_axis) < 1e-8:
                AssertionError("Moved in direction: %s, but contact moment arm is: %s, making a negative dot product."%
                    constraint_axis, moment_arm, np.dot(moment_arm, constraint_axis))
            torque_to_rotate = np.cross(constraint_axis, moment_arm)
            direction_sign = np.sign(torque_to_rotate)
            if direction_sign == 0:
                raise AssertionError("wow amazing tie")

            #vertices are listed counter-clockwise
            vertex_index_neighbor_contact =  np.mod((all_contacts_active_idxs[0] - direction_sign), self.num_vertices)
            if direction_sign < 0:
                edge_contact = all_contacts_active_idxs[0]
            elif direction_sign > 0:
                edge_contact = np.mod((all_contacts_active_idxs[0] -1), self.num_vertices)
            self.edge_angles_original
            if all_contacts_active_idxs[0] in left_contacts:
                pass # ???

            if active_edge == "bottom":
                resting_angle = -self.edge_angles_original[edge_contact]
                jig_edge_i = 0
            elif active_edge == "left":
                resting_angle = -self.edge_angles_original[edge_contact] - np.deg2rad(90)
                jig_edge_i = 1

            arbitrary = 10
            p = self.get_pose_against_edge(jig_edge_i, resting_angle, displacement_along_edge=arbitrary)
            new_center_arbitrary = self.transform_object_point_to_jig_frame(p, center)
   


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
    return ax.plot(obj.vertex_list[i,0],obj.vertex_list[i,1],'r.',**kwcontact)

def plot_jig_relative(obj, ax, obj_pose=(0,0,0), kwline={}, kwcontact={}):
    """
    obj_pose is the pose that the object is made to appear at, and the jig ploted relative to it
    """
    jig_vertices = np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 5.0]])
    x, y, angle = obj.get_pose()

    jig_vertices_transformed = np.dot(jig_vertices  - [x, y], rot_SO2(-angle))

    return ax.plot(jig_vertices_transformed[:,0], jig_vertices_transformed[:,1], **kwline)


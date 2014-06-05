from pyhull import qvoronoi
import pyhull.halfspace
import pyhull.voronoi
import numpy as np

class VoronoiH():
    def __init__(self,points,bounds=None):
        #points is list of points with same dimension D
        #bounds is 2-by-D dimension array where [0,:] are lower bounds and [1:] are upper bounds (hypercube)
        self.points = points
        self.v = pyhull.voronoi.VoronoiTess(points,False)
        
        self.v.vertices = np.array(self.v.vertices)
        
        self.dimension = len(points[0])
        d = self.dimension
        
        out = qvoronoi("Fo Fi FD", points )

        n_outer = int(out[0])

        outer_planes = np.zeros( (n_outer,d+1))
        outer_pairs = []

        for i,line in enumerate(out[1:1+n_outer]):
            outer_pairs.append([int(x) for x in line.split()[1:3]])
            outer_planes[i,:] = [float(x) for x in line.split()[3:]]

        n_inner = int(out[1+n_outer])

        inner_planes = np.zeros( (n_inner,d+1))
        inner_pairs = []

        for i,line in enumerate(out[2+n_outer:]):
            inner_pairs.append([int(x) for x in line.split()[1:3]])    
            inner_planes[i,:] = [float(x) for x in line.split()[3:]]
            
        self.n_outer = n_outer
        self.outer_planes = outer_planes
        self.outer_pairs = outer_pairs
        
        self.n_inner = n_inner
        self.inner_planes = inner_planes
        self.inner_pairs = inner_pairs

        #make axis-aligned planes from the bounds given
        
        self.bounds = bounds
        if bounds is not None:
            self.bounds = np.array(bounds)
            bounds_planes = np.zeros( (2*self.dimension, self.dimension+1) )
            for i in range(self.dimension):
                low = bounds[0][i]
                upp = bounds[1][i]
                
                plane = np.zeros( (self.dimension+1,))
                plane[i] = 1
                plane[-1] = -upp
                
                bounds_planes[2*i,:] = plane
                
                plane[i] = -1
                plane[-1] = low
                
                bounds_planes[2*i+1,:] = plane

            self.bounds_planes = bounds_planes
        else:
            self.bounds_planes = np.zeros( (0,self.dimension+1))
            
    def ridges_of(self,i):
        #returns dictionary. 
        #keys are indices into v.points
        #values are lists of indices into v.vertices

        #in 2D, ridges are line segments and rays.
        #in 3D, ridges are polygons (possibly unbounded)
        ridges = {}
        for key in self.v.ridges:
            if key[0] == i:
                ridges[key[1]] = self.v.ridges[key]
            if key[1] == i:
                ridges[key[0]] = self.v.ridges[key]
        return ridges

    def vrep_voronoi_cell(self,i):
        #returns set of indices into v.vertices.
        #these points are the V-representation of the voronoi cell of v.points[i]
        s = set()
        for v in self.ridges_of(i).values():
            for j in v:
                s.add(j)
        return s
    
    def bbox_voronoi_cell(self,i):
        vi = self.vrep_voronoi_cell(i)
        
        if 0 in vi and self.bounds is None:
            raise ValueError('Voronoi cell is unbounded')
            
        if 0 not in vi:
            extreme_points = self.v.vertices[list(vi)]

            low = np.amin(extreme_points,axis=0)
            upp = np.amax(extreme_points,axis=0)

            if self.bounds is not None:
                #the lower bound is the tightest of mi and the lower bound on the bbox.
                low = np.maximum(low,self.bounds[0,:])
                upp = np.minimum(upp,self.bounds[1,:])

        else:
            assert self.in_voronoi_cell(self.points[i], i) 
            #this cell is unbounded, but we must compute the extreme points of the halfplane intersections including the bbox.
            halfspaces = []
            
            for plane in self.hrep_voronoi_cell_all(i):
                hs = pyhull.halfspace.Halfspace(plane[:-1],plane[-1])
                halfspaces.append(hs)
            
            hsi = pyhull.halfspace.HalfspaceIntersection(halfspaces,self.points[i])
            
            extreme_points = np.array(hsi.vertices)
            
            low = np.amin(extreme_points,axis=0)
            upp = np.amax(extreme_points,axis=0)
            

        bbox = np.zeros((2,self.dimension))
        bbox[0,:] = low
        bbox[1,:] = upp

        return bbox            
            
    def uniform_sample_cell(self,i,n):
        samples = []
        
        bbox = self.bbox_voronoi_cell(i)
        bbox_size = bbox[1,:] - bbox[0,:]
        
        while len(samples)<n:
            x = np.random.rand(self.dimension)
            x = x * bbox_size + bbox[0,:]
            
            #x is now uniform inside bbox
            
            if self.in_voronoi_cell(x,i):
                samples.append(x)
                
        return np.array(samples)            
    
    def in_voronoi_cell(self,point,i):
        planes = self.hrep_voronoi_cell_all(i)

        #point = point.reshape( (-1, self.dimension) )
        #p = np.tensordot(planes[:,:-1] ,point,axes=[ [1,], [1,]])
        #assert(p.shape == ( len(planes),len(point) ) )
        
        #p = np.dot(planes[:,:-1],point)
        #test = (p + planes[:,-1]) < 0
        #return np.all(test)
        
        
        #return np.logical_and(test,axis=0)
        #assert(test.shape == ( len(planes),len(point) ) )
        #return np.prod(test,axis=1)
        
        inside = True
        for plane in planes:
            if np.dot(plane[:-1],point) + plane[-1] > 0 :
                inside = False
                break
        return inside
                        
    def hrep_voronoi_cell(self,i):
        inner_i = []
        inner_sign = []
        #the plane points to the point in pair[0], so if cell appeared in pair[1], flip the sign

        for j in range(self.n_inner):
            pair = self.inner_pairs[j]
            if pair[0]==i:
                inner_i.append(j)
                inner_sign.append(1)
            elif pair[1]==i:
                inner_i.append(j)
                inner_sign.append(-1)


        outer_i = []
        outer_sign = []

        for j in range(self.n_outer):
            pair = self.outer_pairs[j]
            if pair[0]==i:
                outer_i.append(j)
                outer_sign.append(1)
            elif pair[1]==i:
                outer_i.append(j)
                outer_sign.append(-1)

        cell_outer_planes = self.outer_planes[outer_i,:] * np.array(outer_sign).reshape((-1,1))
        cell_inner_planes = self.inner_planes[inner_i,:] * np.array(inner_sign).reshape((-1,1))

        return cell_outer_planes, cell_inner_planes

    def hrep_voronoi_cell_all(self,i):
        #returns array of planes. A plane is such that
        #  np.dot(plane[0:-1] , point) + plane[-1] < 0 is true for a point in the voronoi cell.
        outer,inner = self.hrep_voronoi_cell(i)
        return np.r_[outer,inner,self.bounds_planes]

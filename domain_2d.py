import numpy as np
import collections
import itertools


duplo_unit = 32.0 * 1e-3 # length of a 2-by-2 duplo block, in meters

# pose.x is non-negative. It is the position of the object centroid relative to the jig frame
Pose = collections.namedtuple("Pose", ("x", "theta"))

class Domain2D(object):
	def __init__(self):
		self.object_height = 2 * duplo_unit
		self.object_width = 3 * duplo_unit

		# w, h, listed in clockwise order. coordinates of vertices in object frame
		self.vertices = np.array([(1,1), (1, -1), (-1, -1), (-1, 1)]) * [self.object_width, self.object_height]

		self.n_vertices = len(self.vertices)
		cyclic_idx = list(itertools.islice(itertools.cycle(range(self.n_vertices)), 0, self.n_vertices+1))

		# edge_0 (and corresponding normal_0) is made from vertex_0 and vertex_1
		self.edge_tangents = np.diff(self.vertices[cyclic_idx, :], axis=0)
		self.edge_out_normals = self.edge_tangents[:, [1,0]] * [-1, +1] # rotate the vector +90 degrees, (x,y)->(-y,+x)

		self.edge_out_unit_normals = self.edge_out_normals / (np.linalg.norm(self.edge_out_normals,axis=1)).reshape((-1, 1))


	def active_vertex(self, normal):
		"""
		normal is a 2-vector. If it points down, that means the top of the rectangle is against the jig
		"""
		if len(normal) != 2:
			raise ValueError()

		# taking dot product with edge tangents is like cross product with edge normals
		projs = np.dot(self.edge_tangents, normal)
		projs_cylic = np.array(
			list(itertools.islice(itertools.cycle(
				projs
				), 0, self.n_vertices+1))
			)

		D = np.diff(np.sign(projs_cylic))
		edges_pre_transition = np.where(D>0)[0]

		# there should only be one element except for when normal is one of the normals of the polygon.
		# in which case, just choose the first (least positive angle) vertex.
		if len(edges_pre_transition) == 0:
			raise ValueError("normal vector is zero")

		edge_pre_transition = edges_pre_transition[0]

		return (edge_pre_transition + 1) % self.n_vertices


	def pose_contact(self, normal):
		"""
		pose, where normal is normal of jig (pointing in direction of the object) measured in the object frame.
		"""
		v_i = self.active_vertex(normal)
		v = self.vertices[v_i] # this is a point, but also the vector from object origin to the point
		
		x = np.dot(-v, normal/np.linalg.norm(normal)) # project vertex displacement onto jig unit normal.
		theta = np.arctan2(-normal[1], -normal[0])
		return Pose(x=x, theta=theta)

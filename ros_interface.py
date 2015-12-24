import rospy
import tf

import rosgraph_msgs.msg as rosgraph_msg
import visualization_msgs.msg as viz_msg
import geometry_msgs.msg as geom_msg
import std_msgs.msg as std_msg

import numpy as np


def init_publisher_node():
    return rospy.init_node("box2d_python_simulation")


def box2d_body_to_tf_pose(body):
    translation = [0,0,0]
    translation[0:2] = body.position
    orientation = tf.transformations.quaternion_about_axis(body.angle, (0, 0, 1))
    return (translation, orientation)


def planar_shape_to_point3(shape):
    # shape is assumed to be not closed
    shape3 = [geom_msg.Point(x, y, 0) for (x, y) in shape]
    shape3.append(shape3[0])
    return shape3


class DomainPublisher(object):
    def __init__(self, domain, controller=None):
        self.domain = domain
        self.controller = controller

        self.clock_publisher = rospy.Publisher("/clock", rosgraph_msg.Clock, queue_size=1)
        self.viz_markers_publisher = rospy.Publisher("/ggrdf/fixtures", viz_msg.Marker, queue_size=5)
        self.forcetorque_publisher = rospy.Publisher("/force_torque", geom_msg.WrenchStamped, queue_size=1)

        self.tf_broadcaster = tf.TransformBroadcaster()

        self.world_frame_name = "domain_base"


    def get_domain_sim_time(self):
        time_in_seconds = self.domain.simtime
        return rospy.Time.from_seconds(time_in_seconds)


    def publish_simtime(self):
        msg = rosgraph_msg.Clock(self.get_domain_sim_time())
        self.clock_publisher.publish(msg)
        return msg


    def send_transform(self, box2d_body, frame_name, parent_frame_name, orientation_override=None):
        translation, orientation = box2d_body_to_tf_pose(box2d_body)
        if orientation_override is not None:
            orientation = orientation_override

        translation  = [ e*(1.0/self.domain.dynamics.simulation_scale) for e in translation ]

        self.tf_broadcaster.sendTransform(
            translation, orientation,
            self.get_domain_sim_time(),
            frame_name, parent_frame_name)


    def publish_tf(self):
        self.send_transform(self.domain.dynamics.setpoint_body,
            "setpoint",
            self.world_frame_name)

        self.send_transform(self.domain.dynamics.grasp_body,
            "grasp",
            self.world_frame_name)

        self.send_transform(self.domain.dynamics.manipuland_body,
            "object",
            self.world_frame_name)

        # if you want to plot a vector expressed in world coordinates,
        # but you want the base somewhere else.
        self.send_transform(self.domain.dynamics.setpoint_body,
            "setpoint_translation",
            self.world_frame_name, orientation_override=(0, 0, 0, 1))

        self.send_transform(self.domain.dynamics.grasp_body,
            "grasp_translation",
            self.world_frame_name, orientation_override=(0, 0, 0, 1))

        if self.controller:
            cf = self.controller.compliance_frame

            #homogeneous transform matrix
            mat = np.zeros((4, 4))

            mat[0:2, 0:2] = cf[2,2]
            mat[2,2] = 1
            mat[3,3] = 1

            # there is no translation in this tf because its parent
            # is the setpoint_translation frame
            self.tf_broadcaster.sendTransform(
                (0, 0, 0),
                tf.transformations.quaternion_from_matrix(mat),
                self.get_domain_sim_time(),
                "compliance_frame",
                "setpoint_translation")


    def publish_fixtures_shape(self):
        v = self.domain.dynamics.manipuland_fixture.shape.vertices
        s = float(self.domain.dynamics.simulation_scale)
        v_scaled = [(x/s, y/s) for (x,y) in v]
        points = planar_shape_to_point3(v_scaled)

        marker = viz_msg.Marker()
        marker.header.frame_id = "object"
        marker.header.stamp = self.get_domain_sim_time()
        marker.ns = "box2d_fixtures"
        marker.id = 0 
        marker.type = viz_msg.Marker.LINE_STRIP
        marker.action = viz_msg.Marker.ADD
        marker.pose.orientation.w = 1.0 #identity transform
        marker.scale.x = 0.005

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = points
        marker.lifetime = rospy.Duration(0.1)

        self.viz_markers_publisher.publish(marker)


    def publish_forcetorque(self):
        ft_w = self.domain.forcetorque_measurement
        f_w = ft_w[0:2]
        # this is ft measured in the physics simulation frame.
        # however, in real life, the sensor is mounted on the grasp frame.
        # so project the force into the rotated frame.

        a = self.domain.dynamics.grasp_body.angle
        m = tf.transformations.rotation_matrix(-a, [0, 0, 1])[0:2,0:2]
        f_g = np.dot(m, f_w)

        f = geom_msg.Vector3(f_g[0], f_g[1], 0)
        t = geom_msg.Vector3(0, 0, ft_w[2])

        ws = geom_msg.WrenchStamped(
            std_msg.Header(None, self.get_domain_sim_time(), "grasp"), 
            geom_msg.Wrench(f,t))

        self.forcetorque_publisher.publish(ws)


    def publish(self):
        self.publish_simtime()
        self.publish_tf()
        self.publish_forcetorque()
        self.publish_fixtures_shape()






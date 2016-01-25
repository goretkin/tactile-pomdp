import rospy

rospy.init_node("estimator")

import geometry_msgs.msg as geom_msg
import std_msgs.msg as std_msg
import tf

import time

import threading

import estimator
import estimator_state_space
import estimator_transition_model
import estimator_belief

def tf_to_mat(tf_transform):
    t, o = tf_transform
    return tf.transformations.translation_matrix(t).dot(tf.transformations.quaternion_matrix(o))

def mat_to_tf(mat):
    translation = tf.transformations.translation_from_matrix(mat)
    orientation = tf.transformations.quaternion_from_matrix(mat)

    return (translation, orientation)


def world_grasp_to_relative(grasp_old, grasp_new):
    inv = tf.transformations.inverse_matrix
    # take two poses (tf format) expressed in a base frame
    # express the transform between the two poses in the frame attached to the first pose
    # return that transform in tf format.
    displacement_in_old = inv(
        tf_to_mat(grasp_old)).dot(
        tf_to_mat(grasp_new))

    translation = tf.transformations.translation_from_matrix(displacement_in_old)
    orientation = tf.transformations.quaternion_from_matrix(displacement_in_old)

    return (translation, orientation)

def tf_transform_to_pose(tf_transform):
    translation, orientation = tf_transform
    return geom_msg.Pose(
        position=geom_msg.Vector3(*translation),
        orientation=geom_msg.Quaternion(*orientation)
        )

class AggregateObservation(object):
    def __init__(self):
        self.forcetorque_subscriber = rospy.Subscriber("/force_torque", geom_msg.WrenchStamped, self.forcetorque_cb, queue_size=1)
        self.action_twist_subscriber = rospy.Subscriber("/action_twist", geom_msg.TwistStamped, self.action_twist_cb, queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.lock = threading.Lock()

        self.action_msg = None
        self.force_torque_msg = None

        self.observation_time = None
        self.last_observation_time = None
        self.grasp_pose_in_world_frame = None
        self.last_grasp_pose_in_world_frame = None

        self.observation_duration = None
        self.displacement_observation_2d = None

        self.min_time_between_aggregated_observations = rospy.Duration(0.1) # 10 Hz

        self.update_cb = None

    def forcetorque_cb(self, msg):
        self.lock.acquire()
        self.force_torque_msg = msg
        self.combine()
        self.lock.release()

    def action_twist_cb(self, msg):
        self.lock.acquire()
        self.action_msg = msg
        self.combine()
        self.lock.release()

    def combine(self):
        if self.action_msg and self.force_torque_msg:
            self.time_error = self.action_msg.header.stamp - self.force_torque_msg.header.stamp
            trial_observation_time = self.force_torque_msg.header.stamp + self.time_error/2.0

            if self.observation_time is not None:
                trial_observation_duration = trial_observation_time - self.observation_time
                if trial_observation_duration < self.min_time_between_aggregated_observations:
                    return # wait more time

            self.last_observation_time = self.observation_time
            self.last_grasp_pose_in_world_frame = self.grasp_pose_in_world_frame

            self.observation_time  = trial_observation_time

            if self.observation_time is not None and self.last_observation_time is not None:
                self.observation_duration = self.observation_time - self.last_observation_time

                self.tf_listener.waitForTransform("domain_base", "grasp", self.observation_time, rospy.Duration(1.0))
                self.grasp_pose_in_world_frame = self.tf_listener.lookupTransform("domain_base", "grasp", self.observation_time)
                # could consider using lookupTransformFull

                if self.last_grasp_pose_in_world_frame is not None and self.grasp_pose_in_world_frame is not None:
                    #for debugging only
                    arbitrary = 0
                    last_grasp_pose_publisher.publish(
                        geom_msg.PoseStamped(
                            std_msg.Header(arbitrary, self.observation_time, "domain_base"),
                            tf_transform_to_pose(self.last_grasp_pose_in_world_frame)
                        )
                    )
                    grasp_pose_publisher.publish(
                        geom_msg.PoseStamped(
                            std_msg.Header(arbitrary, self.observation_time, "domain_base"),
                            tf_transform_to_pose(self.grasp_pose_in_world_frame)
                        )
                    )                    
                    

                    
                    displacement_in_old = world_grasp_to_relative(
                        self.last_grasp_pose_in_world_frame,
                        self.grasp_pose_in_world_frame,
                        )

                    translation, orientation = displacement_in_old

                    angle = tf.transformations.euler_from_quaternion(
                                orientation,
                                axes='sxyz'
                            )[2]

                    self.displacement_observation_2d = (translation[0], translation[1], angle)

                    # Originally this was not nested within the above conditions
                    # But now the checks should be redundant 
                    if self.update_cb is not None:
                        if (self.displacement_observation_2d is not None and
                            self.observation_duration is not None and
                            self.force_torque_msg is not None and
                            self.action_msg is not None):
                            self.update_cb(
                                displacement_obs=self.displacement_observation_2d,
                                duration=self.observation_duration,
                                force_obs=(self.force_torque_msg.wrench.force.x, self.force_torque_msg.wrench.force.y, self.force_torque_msg.wrench.torque.z),
                                action_twist=(self.action_msg.twist.linear.x, self.action_msg.twist.linear.y, self.action_msg.twist.angular.z),
                                observation_timestamp=self.observation_time
                                )

                    #print(self.observation_duration, self.displacement_observation_2d)

                self.action_msg = None
                self.force_torque_msg = None

state_space = estimator_state_space.StateSpace()
Belief = estimator_belief.belief_factory(state_space)
estimator_object = estimator.Estimator(state_space)
obs_act_update = estimator_object.obs_act_update
GuardedVelocity = estimator_transition_model.guarded_velocity_factory(state_space, Belief)


class RecursiveEstimator(object):
        def __init__(self):
            self.current_belief = Belief()
            self.reset()
            self.condition_force_observation = True
            self.condition_displacement_observation = True

        def reset(self):
            # susceptible to race conditions
            self.current_belief.clear()
            for state in state_space.states:
                self.current_belief[state] = 1.0

            self.current_belief.normalize()

        def reset_box(self):
            # a box of probability around the object.
            self.current_belief.clear()

            for xi in range(state_space.object_half_width_grid, state_space.object_half_width_grid*2):
                xc = state_space.d_xy * xi
                self.current_belief[state_space.nearest((xc, -1))] = 1.0
                self.current_belief[state_space.nearest((-xc, 1))] = 1.0

            for d in [-1, 1]:
                self.current_belief[estimator_state_space.State(estimator_state_space.DirectionStateFactor(d), estimator_state_space.ContactStateFactor())] = 1.0

            self.current_belief.normalize()


        def update_cb(self, displacement_obs, duration, force_obs, action_twist, observation_timestamp):
            action_v = action_twist[0]
            force_x = force_obs[0] if self.condition_force_observation else None
            # the robot spits out a displacement observation in the gripper frame.
            # it needs to get a sign flip, because the jig moves in the opposite direction
            # the sign flip should be inside the obs_act_update in the future.
            displacement_x = displacement_obs[0] if self.condition_displacement_observation else None

            duration_s = duration.to_sec()

            time_before_update = time.time()

            action = GuardedVelocity(action_v)
            action.void_probability = 0.0 # void is fully absorbing

            # this is just to debug. show the belief update before conditioning on observations
            self.current_belief_prior = obs_act_update(
                self.current_belief,
                None,
                None,
                action,
                duration_s)

            self.current_belief = obs_act_update(
                self.current_belief,
                displacement_x,
                force_x,
                action,
                duration_s)

            time_after_update = time.time()

            #rospy.loginfo(("Belief Update Time: %s ms"%(1000*(time_after_update-time_before_update)))
            belief_publisher.publish(std_msg.String(self.current_belief.to_json_string()))
            prior_belief_publisher.publish(std_msg.String(self.current_belief_prior.to_json_string()))

            header = std_msg.Header(stamp=observation_timestamp)
            # TODO add a frame_id. probably the object frame, (but think more carefully for the displacement observation)
            force_observation_publisher.publish(
                geom_msg.WrenchStamped(
                    header=header,
                    wrench=geom_msg.Wrench(
                        force=geom_msg.Vector3(
                            x=force_x))))


            displacement_observation_publisher.publish(
                geom_msg.TransformStamped(
                    header=header,
                    transform=geom_msg.Transform(
                        translation=geom_msg.Vector3(
                            x=displacement_x))))

            action_publisher.publish(
                geom_msg.TwistStamped(
                    header=header,
                    twist=geom_msg.Twist(
                        linear=geom_msg.Vector3(
                            x=action_v))))



ao = AggregateObservation()
re = RecursiveEstimator()

belief_publisher = rospy.Publisher("/estimator_belief", std_msg.String, queue_size=10, latch=True)
prior_belief_publisher = rospy.Publisher("/estimator_prior_belief", std_msg.String, queue_size=10, latch=True)

displacement_observation_publisher = rospy.Publisher("/estimator/displacement_observation", geom_msg.TransformStamped, queue_size=10, latch=True)
force_observation_publisher = rospy.Publisher("/estimator/force_observation", geom_msg.WrenchStamped, queue_size=10, latch=True)
action_publisher = rospy.Publisher("/estimator/action", geom_msg.TwistStamped, queue_size=10, latch=True)

last_grasp_pose_publisher = rospy.Publisher("/estimator/last_grasp_pose_in_world_frame", geom_msg.PoseStamped, queue_size=10, latch=True)
grasp_pose_publisher = rospy.Publisher("/estimator/grasp_pose_in_world_frame", geom_msg.PoseStamped, queue_size=10, latch=True)


belief_publisher.publish(std_msg.String(re.current_belief.to_json_string())) #initial belief

ao.update_cb = re.update_cb
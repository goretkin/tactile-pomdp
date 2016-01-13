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

            self.tf_listener.waitForTransform("grasp", "domain_base", self.observation_time, rospy.Duration(1.0))
            self.grasp_pose_in_world_frame = self.tf_listener.lookupTransform("grasp", "domain_base", self.observation_time)
            # could consider using lookupTransformFull

            if self.last_grasp_pose_in_world_frame is not None and self.grasp_pose_in_world_frame is not None:
                displacement_in_world = tf.transformations.inverse_matrix(
                    tf_to_mat(self.last_grasp_pose_in_world_frame)).dot(
                    tf_to_mat(self.grasp_pose_in_world_frame))

                # we want to convert the displacement expressed in world to displacement expressed in grasp_{last_time}
                transformer = tf.transformations.quaternion_matrix(self.last_grasp_pose_in_world_frame[1])
                displacement_in_grasp_last_time = tf.transformations.inverse_matrix(transformer).dot(displacement_in_world)

                translation = tf.transformations.translation_from_matrix(displacement_in_grasp_last_time)
                orientation = tf.transformations.quaternion_from_matrix(displacement_in_grasp_last_time)

                #angle = tf.transformations.euler_from_quaternion(orientation, axes='sxyz')[2]

                #TODO this assumes that x,y axes of world and grasp are coplanar
                angle = tf.transformations.euler_from_quaternion(
                            tf.transformations.quaternion_from_matrix(
                                displacement_in_world),
                            axes='sxyz'
                        )[2]

                angle = - angle # ??? this is what looks right from debugging.
                self.displacement_observation_2d = (translation[0], translation[1], angle)

            if self.update_cb is not None:
                if (self.displacement_observation_2d is not None and
                    self.observation_duration is not None and
                    self.force_torque_msg is not None and
                    self.action_msg is not None):
                    self.update_cb(
                        displacement_obs=self.displacement_observation_2d,
                        duration=self.observation_duration,
                        force_obs=(self.force_torque_msg.wrench.force.x, self.force_torque_msg.wrench.force.y, self.force_torque_msg.wrench.torque.z),
                        action_twist=(self.action_msg.twist.linear.x, self.action_msg.twist.linear.y, self.action_msg.twist.angular.z)
                        )

            #print(self.observation_duration, self.displacement_observation_2d)

            self.action_msg = None
            self.force_torque_msg = None



ao = AggregateObservation()

belief_publisher = rospy.Publisher("/estimator_belief", std_msg.String, queue_size=10, latch=True)


state_space = estimator_state_space.StateSpace()
Belief = estimator_belief.belief_factory(state_space)
estimator_object = estimator.Estimator(state_space)
obs_act_update = estimator_object.obs_act_update
GuardedVelocity = estimator_transition_model.guarded_velocity_factory(state_space, Belief)


class BeliefHolderSingleton(object):
    current_belief = Belief()
    @staticmethod
    def reset():
        # susceptible to race conditions
        BeliefHolderSingleton.current_belief.clear()
        for state in state_space.states:
            BeliefHolderSingleton.current_belief[state] = 1.0

        BeliefHolderSingleton.current_belief.normalize()


belief_publisher.publish(std_msg.String(BeliefHolderSingleton.current_belief.to_json_string())) #initial belief

def cb(displacement_obs, duration, force_obs, action_twist):
    action_v = action_twist[0]
    force_x = force_obs[0]
    displacement_x = displacement_obs[0]
    duration_s = duration.to_sec()

    time_before_update = time.time()

    BeliefHolderSingleton.current_belief = obs_act_update(
        BeliefHolderSingleton.current_belief,
        displacement_x, 
        force_x, 
        GuardedVelocity(action_v),
        duration_s)

    time_after_update = time.time()

    #rospy.loginfo(("Belief Update Time: %s ms"%(1000*(time_after_update-time_before_update)))
    belief_publisher.publish(std_msg.String(BeliefHolderSingleton.current_belief.to_json_string()))

BeliefHolderSingleton.reset()
ao.update_cb = cb
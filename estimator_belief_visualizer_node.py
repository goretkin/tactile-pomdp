import rospy

import std_msgs.msg as std_msg

import estimator_belief
import estimator_state_space

import matplotlib.pyplot as plt
from threading import Lock

class BeliefVisualizer(object):
    def __init__(self, state_space, plot_prior=False):
        self.state_space = state_space
        self.Belief_class = estimator_belief.belief_factory(state_space)
        self.current_belief = self.Belief_class()

        self.fig = plt.figure()

        n_subplots = 3 if plot_prior else 2
        self.ax = self.fig.add_subplot(n_subplots,1,1)
        self.ax_logx = self.fig.add_subplot(n_subplots,1,2)

        if plot_prior:
            self.ax_prior = self.fig.add_subplot(n_subplots,1,3)

        self.ax.axhline(0.0)
        self.ax.axvline(self.state_space.object_half_width)
        self.ax.axvline(-self.state_space.object_half_width)
        self.ax.set_ylabel(r"$p$")

        self.ax_logx.set_ylabel(r"$\log p$")

        self.fig.show()

        self.plot_lock = Lock()
        # should be the final step, so that all variables are initialized.
        self.belief_subscriber = rospy.Subscriber("/estimator_belief", std_msg.String, self.belief_cb, queue_size=1)

        if plot_prior:
            self.prior_belief_subscriber = rospy.Subscriber("/estimator_prior_belief", std_msg.String, self.prior_belief_cb, queue_size=1)


    def belief_cb(self, msg):
        self.plot_lock.acquire()

        self.current_belief.from_json_string(msg.data)
        self.ax.clear()
        self.ax_logx.clear()
        self.current_belief.plot(self.ax)
        self.current_belief.plot(self.ax_logx, logprob=True)
        self.ax.set_title("Time: %04.2f"%(rospy.get_time()))
        self.fig.canvas.draw()

        self.plot_lock.release()

    def prior_belief_cb(self, msg):
        self.plot_lock.acquire()

        b = self.Belief_class()
        b.from_json_string(msg.data)
        self.ax_prior.clear()
        b.plot(self.ax_prior)
        self.ax_prior.set_title("Prior. Time: %04.2f"%(rospy.get_time()))
        self.fig.canvas.draw()

        self.plot_lock.release()

rospy.init_node ("estimator_belief_visualizer")
bv = BeliefVisualizer(estimator_state_space.StateSpace(), plot_prior=False)

#bv.fig.canvas.start_event_loop(timeout=-1)
#rospy.spin()

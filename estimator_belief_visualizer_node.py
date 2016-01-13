import rospy

import std_msgs.msg as std_msg

import estimator_belief
import estimator_state_space

import matplotlib.pyplot as plt

class BeliefVisualizer(object):
    def __init__(self, state_space):
        self.state_space = state_space
        self.current_belief = estimator_belief.belief_factory(state_space)()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(2,1,1)
        self.ax_logx = self.fig.add_subplot(2,1,2)

        self.ax.axhline(0.0)
        self.ax.axvline(self.state_space.object_half_width)
        self.ax.axvline(-self.state_space.object_half_width)
        self.ax.set_ylabel(r"$p$")

        self.fig.show()

        # should be the final step, so that all variables are initialized.
        self.belief_subscriber = rospy.Subscriber("/estimator_belief", std_msg.String, self.belief_cb, queue_size=1)


    def belief_cb(self, msg):
        self.current_belief.from_json_string(msg.data)
        self.ax.clear()
        self.ax_logx.clear()
        self.current_belief.plot(self.ax)
        self.current_belief.plot(self.ax_logx, logprob=True)
        self.ax.set_title("Time: %04.2f"%(rospy.get_time()))
        self.fig.canvas.draw()


rospy.init_node ("estimator_belief_visualizer")
bv = BeliefVisualizer(estimator_state_space.StateSpace())

#bv.fig.canvas.start_event_loop(timeout=-1)
#rospy.spin()

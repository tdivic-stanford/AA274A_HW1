import numpy as np
from utils import wrapToPi, simulate_car_dyn

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        # translate the position vector to the goal frame
        translated_pos_vector = np.array([x, y, 0]) - np.array([self.x_g, self.y_g, 0])

        # rotate the position vector to the goal frame
        R_goal = np.array([[np.cos(self.th_g), -np.sin(self.th_g), 0],
                           [np.sin(self.th_g), np.cos(self.th_g), 0],
                           [0, 0, 1]])
        pos_vector_in_goal = R_goal.dot(translated_pos_vector)

        # put theta in terms of the new frame
        th_in_goal = th - self.th_g

        # convert x, y, th to polar coordinates in the goal frame
        new_x = pos_vector_in_goal[0]
        new_y = pos_vector_in_goal[1]
        rho = np.sqrt(new_x**2 + new_y**2)
        alpha = wrapToPi(np.arctan2(new_y, new_x) - th_in_goal)
        delta = wrapToPi(alpha + th_in_goal)

        # calculate the controls
        V = self.k1 * rho * np.cos(alpha)
        om = self.k2 * alpha + self.k1 * np.sinc(alpha / np.pi) * np.cos(alpha) * (alpha + self.k3 * delta)
        
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om


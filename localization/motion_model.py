import numpy as np
import math

class MotionModel:

    def __init__(self, node):
        pass

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################

        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2]
        n = particles.shape[0]

        alpha_1 = 0.2
        alpha_2 = 0.2
        alpha_3 = 0.4
        alpha_4 = 0.07

        delta_rot_1 = np.arctan2(dy, dx)
        delta_trans = np.sqrt(dx**2 + dy**2)
        delta_rot_2 = dtheta - delta_rot_1

        delta_rot_1_hat = delta_rot_1 + np.random.normal(0, alpha_1, size=n)
        delta_trans_hat = delta_trans + np.random.normal(0, alpha_2, size=n)
        delta_rot_2_hat = delta_rot_2 + np.random.normal(0, alpha_3, size=n)

        particles[:,0] += delta_trans_hat*np.cos(particles[:, 2]+ delta_rot_1_hat)
        particles[:, 1] += delta_trans_hat*np.sin(particles[:, 2]+ delta_rot_1_hat)
        particles[:, 2] += delta_rot_1_hat + delta_rot_2_hat

        return particles

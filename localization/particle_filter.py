#!/usr/bin/env python3


import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math

VAR_POS = 0.5
VAR_ANGLE = 0.1

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        # Get the particle filter frame name.
        self.declare_parameter('particle_filter_frame', "pf_base_link")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        # Get topic names from parameters.
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', 100)  # you can adjust this

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        # Subscriptions.
        self.laser_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.laser_callback,
            1)
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            1)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.pose_callback,
            1)

        # Publisher. (Publish the estimated pose as an odometry msg on /pf/pose/odom)
        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize models.
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Number of particles.
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        # Particle set: each particle is [x, y, theta] stored as a numpy array.
        self.particles = None


    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Called when a new initial pose is received (interactive initialization).
        Initialize the particles around the given pose with a small spread.
        """
        # Extract the initial pose.
        init_x = msg.pose.pose.position.x
        init_y = msg.pose.pose.position.y

        # Convert the provided quaternion into euler (yaw)
        q = msg.pose.pose.orientation
        init_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        # Initialize particles from a gaussian distribution around [init_x, init_y, init_theta].
        self.particles = np.empty((self.num_particles, 3))
        self.particles[:, 0] = np.random.normal(init_x, VAR_POS, self.num_particles)
        self.particles[:, 1] = np.random.normal(init_y, VAR_POS, self.num_particles)
        self.particles[:, 2] = np.random.normal(init_theta, VAR_ANGLE, self.num_particles)

        self.get_logger().info("Particles initialized from /initialpose")

        # Publish the “best” (average) pose.
        self.publish_average_pose()

    def odom_callback(self, msg: Odometry):
        """
        Update particle positions according to the motion model using odometry data.
        (Note: only the twist component is provided.)
        """
        # If we have not yet initialized particles, skip.
        if self.particles is None:
            return

        # Extract the incremental motion from the twist.
        # The odom message only has the twist field.
        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        dtheta = msg.twist.twist.angular.z

        odometry = np.array([dx, dy, dtheta])

        # Update particles using the motion model.
        self.particles = self.motion_model.evaluate(self.particles, odometry)

        # Publish the updated average pose.
        self.publish_average_pose()

    def laser_callback(self, msg: LaserScan):
        """
        When a new laser scan arrives: evaluate the sensor model,
        and resample particles according to the likelihood of their
        predictions versus the actual scan.
        """
        # If particles not initialized or map not available in the sensor model, do nothing.
        if self.particles is None or not self.sensor_model.map_set:
            return

        # Convert the scan ranges into a numpy array (or list) of measurements. 
        # Here msg.ranges is already a sequence of floats.
        observation = np.array(msg.ranges)

        # Evaluate the likelihood for each particle (using the sensor model’s evaluation function).
        probabilities = self.sensor_model.evaluate(self.particles, observation)
        if probabilities is None:
            return

        # Normalize probabilities. (Avoid division by zero.)
        prob_sum = np.sum(probabilities)
        if prob_sum == 0:
            # All particles got zero weight. In this case, simply reset the weights.
            probabilities = np.ones(len(probabilities)) / len(probabilities)
        else:
            probabilities = probabilities / prob_sum

        # Resample: draw new particles (with replacement) based on the computed probabilities.
        indices = np.random.choice(
            np.arange(len(self.particles)),
            size=len(self.particles),
            replace=True,
            p=probabilities)
        self.particles = self.particles[indices]

        # Publish the “best” (average) pose.
        self.publish_average_pose()

    def publish_average_pose(self):
        """
        Compute a representative pose (average pose) from the particles.
        For x and y, simply use the mean.
        For theta, compute the mean angle (using sin+cos).
        Publish the result as an Odometry message on /pf/pose/odom.
        """
        if self.particles is None or len(self.particles) == 0:
            return

        avg_x = np.mean(self.particles[:, 0])
        avg_y = np.mean(self.particles[:, 1])
        # For angles, average using sin and cos.
        sin_sum = np.sum(np.sin(self.particles[:, 2]))
        cos_sum = np.sum(np.cos(self.particles[:, 2]))
        avg_theta = math.atan2(sin_sum, cos_sum)

        # Create an Odometry message to publish the best pose.
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"  # expected world frame
        
        # The pose field is used to store the estimated pose.
        odom_msg.pose.pose.position.x = avg_x
        odom_msg.pose.pose.position.y = avg_y
        odom_msg.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, avg_theta)
        odom_msg.pose.pose.orientation = Quaternion(
            x=quat[0],
            y=quat[1],
            z=quat[2],
            w=quat[3]
        )
        
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

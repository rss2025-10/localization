# particle_filter.py

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math


VAR_POS = 0.4
VAR_ANGLE = 0.2

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "pf_base_link")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', 100)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

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

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.pose_callback,
            1)
        #        self.marker_pub = self.create_publisher(MarkerArray, "/particle_markers", 1)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/particle_markers",
            qos_profile=rclpy.qos.QoSProfile(depth=10, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        )
        
        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models.
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = None


    def pose_callback(self, msg):

        x_0 = msg.pose.pose.position.x
        y_0 = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta_0 = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        self.particles = np.empty((self.num_particles, 3))
        self.particles[:, 0] = np.random.normal(x_0, VAR_POS, self.num_particles)
        self.particles[:, 1] = np.random.normal(y_0, VAR_POS, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta_0, VAR_ANGLE, self.num_particles)

        self.publish_average_pose()
        self.publish_particle_markers()

    def odom_callback(self, msg):
            
        if self.particles is None:
            return

        # Get the current time.
        current_time = self.get_clock().now()
        
        # If this is the first odom callback then just record time and exit.
        if not hasattr(self, "last_odom_time"):
            self.last_odom_time = current_time
            return

        # Compute the time difference (dt) in seconds.
        dt = (current_time - self.last_odom_time).nanoseconds * 1e-9
        self.last_odom_time = current_time

        # Multiply the linear and angular velocities by dt.
        dx = msg.twist.twist.linear.x * dt
        dy = msg.twist.twist.linear.y * dt
        dtheta = msg.twist.twist.angular.z * dt
        
        # Form the odometry delta vector.
        odometry = np.array([dx, dy, dtheta])
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        
        self.publish_particle_markers()
        self.publish_average_pose()

    def laser_callback(self, msg):

        if self.particles is None or not self.sensor_model.map_set:
            return

        obs = np.array(msg.ranges)
        probs = self.sensor_model.evaluate(self.particles, obs)

        if probs is None:
            return

        prob_sum = np.sum(probs)
        if prob_sum == 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / prob_sum
            
        idxs = np.random.choice(
            np.arange(len(self.particles)),
            size=len(self.particles),
            replace=True,
            p=probs)

        self.particles = self.particles[idxs]

        
        self.publish_average_pose()
        self.publish_particle_markers()
                

    def publish_average_pose(self):

        if self.particles is None or len(self.particles) == 0:
            return

        avg_x = np.mean(self.particles[:, 0])
        avg_y = np.mean(self.particles[:, 1])
        sin_sum = np.sum(np.sin(self.particles[:, 2]))
        cos_sum = np.sum(np.cos(self.particles[:, 2]))
        avg_theta = math.atan2(sin_sum, cos_sum)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"

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

    def publish_particle_markers(self):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        for i, particle in enumerate(self.particles):
            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = "map"
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = particle[0]
            marker.pose.position.y = particle[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()

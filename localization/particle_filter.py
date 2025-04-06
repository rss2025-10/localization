import numpy as np
import threading

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

# for the particle creation
var_pos = .5
var_ang = .25

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
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

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

        # Initialize the models
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
        self.particles = None
        # thread protection
        self.particle_thread_protection = threading.Lock()
        self.last_odom_time = None


    def pose_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        use_th = theta[2]

        self.particles = np.empty((self.num_particles, 3))
        self.particles[:, 0] = np.random.normal(x, var_pos, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, var_pos, self.num_particles)
        self.particles[:, 2] = np.random.normal(use_th, var_ang, self.num_particles)

        self.publish_average_pose()
        #self.pub_particle_marks()

    def odom_callback(self, msg):

        with self.particle_thread_protection:

            if self.particles is None:
                return

            current = self.get_clock().now()

            if self.last_odom_time is None:
                self.last_odom_time = current
                return

            dt = (current - self.last_odom_time).nanoseconds * 1e-9
            self.last_odom_time = current

            # Flip back to - vals if on car
            dx = -msg.twist.twist.linear.x * dt
            dy = -msg.twist.twist.linear.y * dt
            dtheta = -msg.twist.twist.angular.z * dt

            # change frames? idk this seems to help
            # wait maybe this should go in precomputation for motion model???
            theta = self.particles[:, 2]
            dx_new = dx * np.cos(theta) - dy * np.sin(theta)
            dy_new = dx * np.sin(theta) + dy * np.cos(theta)

            self.particles[:, 0] += dx_new
            self.particles[:, 1] += dy_new
            self.particles[:, 2] += dtheta


            odom = np.array([dx, dy, dtheta])
            self.particles = self.motion_model.evaluate(self.particles, odom)
            self.publish_average_pose()
            #self.pub_particle_marks()

    def laser_callback(self, msg):

        with self.particle_thread_protection:

            if self.particles is None or not self.sensor_model.map_set:
                self.get_logger().info(f"not doing anything: no particles: {self.particles is None}, map set: {not self.sensor_model.map_set}")
                return

            # i think we could downsample here also but that might be redundant
            obs = np.array(msg.ranges)
            probs = self.sensor_model.evaluate(self.particles, obs)

            if probs is None:
                return
            else:
                prob_sum = np.sum(probs)

            # if self.count == 0:
            #     # self.get_logger().info(f"particles: {self.particles}")
            #     # self.get_logger().info(f"pros: {probs}")
            #     self.get_logger().info(f"probs sum: {prob_sum}")

            if prob_sum <= 0:
                self.get_logger().info(f"particles do not sum correctly")
                probs = np.ones(len(probs)) / len(probs)
            else:

                probs = probs / prob_sum

            # if self.count == 0:
            #     self.get_logger().info(f"probs after normalizing: {probs}")
            #     self.get_logger().info(f"max val = {np.max(probs)}")

            # self.get_logger().info(f"Probs: {probs}")
            idxs = np.random.choice(np.arange(len(self.particles)),size=len(self.particles),
                replace=True,
                p=probs)

            self.particles = np.take(self.particles, idxs, axis=0)

            self.publish_average_pose()
            #self.pub_particle_marks()


    def publish_average_pose(self):

        if self.particles is None or len(self.particles) == 0:
            return

        avg_x = np.mean(self.particles[:, 0])
        avg_y = np.mean(self.particles[:, 1])

        #trying to do the circular mean thing
        # assignment suggests other calculations tho? maybe change later?
        sin_sum = np.sum(np.sin(self.particles[:, 2]))
        cos_sum = np.sum(np.cos(self.particles[:, 2]))
        avg_th = math.atan2(sin_sum, cos_sum)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"

        odom_msg.pose.pose.position.x = avg_x
        odom_msg.pose.pose.position.y = avg_y
        odom_msg.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, avg_th)

        odom_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        self.odom_pub.publish(odom_msg)

    def pub_particle_marks(self):

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

# particle_filter.py

import numpy as np
import threading

import rclpy
from rclpy.node import Node
import tf2_ros


from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math


VAR_POS = .5
VAR_ANGLE = .25

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

        self.error_pose_publisher = self.create_publisher(Float64, "/translation_error", 1)
        self.error_rotation_publisher = self.create_publisher(Float64, "/rotation_error", 1)

        self.error_buffer = tf2_ros.Buffer()
        self.error_listener = tf2_ros.TransformListener(self.error_buffer, self)
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

        self.particle_thread_protection = threading.Lock()
        self.count = 0


    def pose_callback(self, msg):

        x_0 = msg.pose.pose.position.x
        y_0 = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.get_logger().info(f"theta{theta}")
        theta_0 = theta[2]


        self.particles = np.empty((self.num_particles, 3))
        self.particles[:, 0] = np.random.normal(x_0, VAR_POS, self.num_particles)
        self.particles[:, 1] = np.random.normal(y_0, VAR_POS, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta_0, VAR_ANGLE, self.num_particles)

        self.publish_average_pose()
        self.publish_particle_markers()

    def odom_callback(self, msg):

        with self.particle_thread_protection:

            if self.particles is None:
                return

            current_time = self.get_clock().now()

            if not hasattr(self, "last_odom_time"):
                self.last_odom_time = current_time
                return

            # Compute the time difference (dt) in seconds.
            dt = (current_time - self.last_odom_time).nanoseconds * 1e-9
            self.last_odom_time = current_time

            noise = 0.0

            dx = msg.twist.twist.linear.x * dt + np.random.normal(0, noise)
            dy = msg.twist.twist.linear.y * dt + np.random.normal(0, noise)
            dtheta = msg.twist.twist.angular.z * dt + np.random.normal(0, noise)

            theta = self.particles[:, 2]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            delta_x = dx * cos_theta - dy * sin_theta
            delta_y = dx * sin_theta + dy * cos_theta

            self.particles[:, 0] += delta_x
            self.particles[:, 1] += delta_y
            self.particles[:, 2] += dtheta

            odometry = np.array([dx, dy, dtheta])
            self.particles = self.motion_model.evaluate(self.particles, odometry)
            self.publish_average_pose()
            self.publish_particle_markers()

    def laser_callback(self, msg):

        with self.particle_thread_protection:

            if self.particles is None or not self.sensor_model.map_set:
                self.get_logger().info(f"Not doing anything: no particles: {self.particles is None}, map set: {not self.sensor_model.map_set}")
                return
            else:
                self.get_logger().info(f"Doing calcs")

            obs = np.array(msg.ranges)
            probs = self.sensor_model.evaluate(self.particles, obs)

            if probs is None:
                return

            # self.get_logger().info(f"Probabilities {probs}")

            # probs = np.clip(probs, 1e-25, 1)
            prob_sum = np.sum(probs)

            if self.count == 0:
                self.get_logger().info(f"particles: {self.particles}")
                self.get_logger().info(f"pros: {probs}")
                self.get_logger().info(f"probs sum: {prob_sum}")

            if prob_sum <= 0:
                self.get_logger().info(f"particles do not sum correctly")
                probs = np.ones(len(probs)) / len(probs)
            else:

                probs = probs / prob_sum

            if self.count == 0:
                self.get_logger().info(f"probs after normalizing: {probs}")
                self.get_logger().info(f"max val = {np.max(probs)}")

            # self.get_logger().info(f"Probs: {probs}")
            idxs = np.random.choice(
                np.arange(len(self.particles)),
                size=len(self.particles),
                replace=True,
                p=probs)

            self.particles = np.take(self.particles, idxs, axis=0)

            self.publish_average_pose()
            self.count += 1
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

        try:
        # Get transform from /map to /base_link
            transform = self.error_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            average_error = np.sqrt((avg_x - translation.x)**2 + (avg_y - translation.y)**2)
            average_rot_error = np.sqrt((quat[0] - rotation.x)**2 + (quat[1] - rotation.y)**2 + (quat[2] - rotation.z)**2 + (quat[3] - rotation.w)**2)


            msg = Float64()
            msg.data = average_error
            rot_msg = Float64()
            rot_msg.data = average_rot_error
            self.error_pose_publisher.publish(msg)
            self.error_rotation_publisher.publish(rot_msg)


        except (tf2_ros.TransformException, Exception) as e:
            self.get_logger().warn(f"Could not get transform: {e}")
            return None


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

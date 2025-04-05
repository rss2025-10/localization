import numpy as np
import math
from scan_simulator_2d import PyScanSimulator2D
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid

class SensorModel:

    def __init__(self, node):

        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)
        self.lidar_scale_to_map_scale = node.get_parameter('lidar_scale_to_map_scale').get_parameter_value().double_value
        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit   = 0.74
        self.alpha_short = 0.07
        self.alpha_max   = 0.07
        self.alpha_rand  = 0.12
        self.sigma_hit   = 0.5

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = self.lidar_scale_to_map_scale * (self.table_width - 1)
        ####################################

        # Precompute the sensor model table
        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Create a simulated laser scan
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def p_hit(self, z, d):
        if z < 0 or z > self.z_max:
            return 0.0

        hit_probability = (1.0 / (math.sqrt(2 * math.pi) * self.sigma_hit)) * (-0.5 * ((z - d) / self.sigma_hit) ** 2)

        cdf_upper = 0.5 * (1 + math.erf((self.z_max - d) / (self.sigma_hit * math.sqrt(2))))
        cdf_lower = 0.5 * (1 + math.erf((0 - d) / (self.sigma_hit * math.sqrt(2))))
        integral = cdf_upper - cdf_lower

        if integral == 0:
            return hit_probability
        else:
            return 1.0 / integral * hit_probability


    def p_short(self, z, d):

        if d <= 0 or z < 0 or z > d:
            return 0.0

        return (2.0 / d) * (1 - z / d)

    def p_max(self, z):

        if abs(z - self.z_max) < (self.lidar_scale_to_map_scale / 2.0):
            return 1.0

        return 0.0

    def p_rand(self, z):

        if z < 0 or z > self.z_max:
            return 0.0

        return 1.0 / self.z_max

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        for i in range(self.table_width):
            d = i * self.lidar_scale_to_map_scale
            for j in range(self.table_width):
                z = j * self.lidar_scale_to_map_scale
                self.sensor_model_table[i, j] = (self.alpha_hit * self.p_hit(z, d) +
                     self.alpha_short * self.p_short(z, d) +
                     self.alpha_max * self.p_max(z) +
                     self.alpha_rand * self.p_rand(z))

            row_sum = np.sum(self.sensor_model_table[i, :])
            self.sensor_model_table[i, :] /= row_sum

        self.sensor_model_table = self.sensor_model_table.T

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        scans = self.scan_sim.scan(particles)
        N, num_beams = scans.shape

        obs_indices = np.array([int(round(z / self.lidar_scale_to_map_scale)) for z in observation])
        obs_indices = np.clip(obs_indices, 0, self.table_width - 1)

        probs = np.ones(N)

        for j in range(num_beams):
            pred = scans[:, j]
            pred_indices = np.array([int(round(z / self.lidar_scale_to_map_scale)) for z in pred])
            pred_indices = np.clip(pred_indices, 0, self.table_width - 1)
            beam_probs = self.sensor_model_table[pred_indices, obs_indices[j]]
            probs *= beam_probs

        return probs

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)
        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_yaw = euler_from_quaternion((origin_o.x, origin_o.y, origin_o.z, origin_o.w))[2]
        origin = (origin_p.x, origin_p.y, origin_yaw)

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")

#!/usr/bin/env python3
# sensor_model.py

import numpy as np
import math
from scan_simulator_2d import PyScanSimulator2D
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid

class SensorModel:

    def __init__(self, node):
        # Get parameters from the node.
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)
        # For the sensor model precomputation we use a fixed maximum range (in meters),
        # as given in the assignment (z_max = 10 m) and a discretization resolution.
        # (If scale==1 then the number of bins is z_max+1.)
        self.lidar_scale_to_map_scale = node.get_parameter('lidar_scale_to_map_scale').get_parameter_value().double_value
        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.node = node

        ####################################
        # Sensor model mixture parameters.
        self.alpha_hit   = 0.74
        self.alpha_short = 0.07
        self.alpha_max   = 0.07
        self.alpha_rand  = 0.12
        # For the hit model, use the sigma value from the writing assignment (0.5 m).
        self.sigma_hit   = 0.5
        # For the short reading model we use the simple triangular form below.
        ####################################

        # Set the maximum sensor range (m) from instructions.
        # Choose a table resolution: we assume scale converts meters to pixels.
        # Here we assume a scale of 1 m per bin; adjust if needed.
        self.scale = self.lidar_scale_to_map_scale
        # Compute the number of bins so that the highest index corresponds to z_max.
        self.table_width = 201
        self.z_max = self.scale * (self.table_width - 1)

        # Create the sensor model lookup table.
        self.sensor_model_table = np.empty((self.table_width, self.table_width))

        # Precompute the sensor model table.
        self.precompute_sensor_model()

        # Create a simulated laser scan.
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,      # no additional simulator noise
            0.01,   # epsilon parameter (not used in our p_max with discretization)
            self.scan_theta_discretization)

        # Subscribe to the map.
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

        node.get_logger().info("SensorModel initialized with z_max=%.2f m and table width = %d" %
                                 (self.z_max, self.table_width))

    def compute_hit_probability(self, z, z_expected):
        """
        p_hit: Gaussian probability centered around z_expected,
        normalized over the interval [0, z_max].
        Returns 0 for values of z outside [0, z_max].
        """
        if z < 0 or z > self.z_max:
            return 0.0

        # Unnormalized Gaussian probability.
        sigma = self.sigma_hit
        coeff = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        exponent = -0.5 * ((z - z_expected) / sigma) ** 2
        p_unnormalized = coeff * math.exp(exponent)

        # Compute the integral over [0, z_max] (using the error function).
        sqrt2 = math.sqrt(2)
        # Compute the cumulative probability from 0 to z_max.
        cdf_upper = 0.5 * (1 + math.erf((self.z_max - z_expected) / (sigma * sqrt2)))
        cdf_lower = 0.5 * (1 + math.erf((0 - z_expected) / (sigma * sqrt2)))
        integral = cdf_upper - cdf_lower

        # Avoid division by zero.
        if integral == 0:
            eta = 1.0
        else:
            eta = 1.0 / integral

        return eta * p_unnormalized

    def compute_short_probability(self, z, z_expected):
        """
        p_short: models unexpected short readings.
        According to instructions:
           p_short(z | x, m) = 2/d (1 - z/d) for 0 <= z <= d (and 0 if z>d or d==0)
        Here d is the expected range (z_expected).
        """
        if z_expected <= 0:
            return 0.0
        if z < 0 or z > z_expected:
            return 0.0
        return (2.0 / z_expected) * (1 - z / z_expected)

    def compute_max_probability(self, z):
        """
        p_max: a spike at the maximum range. With discretization,
        we set this to 1 if the observed measurement equals z_max, 0 otherwise.
        """
        # Here we allow a little tolerance of half a bin.
        tol = self.scale / 2.0
        if abs(z - self.z_max) < tol:
            return 1.0
        return 0.0

    def compute_rand_probability(self, z):
        """
        p_rand: a uniform random measurement probability over [0, z_max].
        """
        if z < 0 or z > self.z_max:
            return 0.0
        return 1.0 / self.z_max

    def precompute_sensor_model(self):
        """
        Precompute the sensor model lookup table.
        For each expected measurement (row: index i, with z_expected = i*scale)
        and for every discretized observed measurement (column: index j with z = j*scale),
        compute the likelihood as a weighted combination.
        Then, normalize each row so that the likelihoods sum to 1.
        """
        for i in range(self.table_width):
            # Expected measurement for this row.
            z_expected = i * self.scale
            for j in range(self.table_width):
                # Observed measurement.
                z = j * self.scale
                p_hit   = self.compute_hit_probability(z, z_expected)
                p_short = self.compute_short_probability(z, z_expected)
                p_max   = self.compute_max_probability(z)
                p_rand  = self.compute_rand_probability(z)
                # Weighted sum.
                p = (self.alpha_hit * p_hit +
                     self.alpha_short * p_short +
                     self.alpha_max * p_max +
                     self.alpha_rand * p_rand)
                self.sensor_model_table[i, j] = p

            # Normalize the row if its sum is nonzero.
            row_sum = np.sum(self.sensor_model_table[i, :])
            if row_sum > 0:
                self.sensor_model_table[i, :] /= row_sum
            else:
                # In case the row summed to zero, use a uniform distribution.
                self.sensor_model_table[i, :] = np.ones(self.table_width) / self.table_width

        self.sensor_model_table = self.sensor_model_table.T

    def evaluate(self, particles, observation):
        """
        Evaluate the sensor model likelihood for each particle.
        For each particle, obtain the predicted beam ranges via the scan simulator,
        then lookup (from the precomputed table) the likelihood corresponding to (predicted, observed)
        pair for each beam. Multiply likelihoods over all beams.
        """
        if not self.map_set:
            return

        scale = self.scale * self.resolution  # conversion from meters to table index (if scale==1, 1 m/bin)
        table_width = self.table_width

        # Obtain predicted ranges for each particle.
        scans = self.scan_sim.scan(particles)  # shape: (N, num_beams)
        N, num_beams = scans.shape

        # Discretize the observed scan measurements.
        obs_indices = np.array([int(round(z / scale)) for z in observation])
        # self.node.get_logger().info(f" obs: {obs_indices}")
        obs_indices = np.clip(obs_indices, 0, table_width - 1)

        likelihoods = np.ones(N)

        # For each beam, look up the beam probability and multiply.
        for j in range(num_beams):
            # Predicted measurements from each particle for beam j.
            pred = scans[:, j]
            # self.node.get_logger().info(f"scans: {pred}")
            pred_indices = np.array([int(round(z / scale)) for z in pred])
            # self.node.get_logger().info(f"Pred indices: {pred_indices}")
            pred_indices = np.clip(pred_indices, 0, table_width - 1)
            # Lookup: for each particle, use its predicted index (row) and the
            # corresponding observed index (column j).
            beam_likelihoods = self.sensor_model_table[obs_indices[j], pred_indices]
            likelihoods *= beam_likelihoods

        return likelihoods

    def map_callback(self, map_msg):
        """
        Convert the incoming OccupancyGrid map message into a numpy array and set the map in the scan simulator.
        """
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)
        self.resolution = map_msg.info.resolution

        # Get map origin (position and yaw).
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_yaw = euler_from_quaternion((origin_o.x, origin_o.y, origin_o.z, origin_o.w))[2]
        origin = (origin_p.x, origin_p.y, origin_yaw)

        # Set the map into the scan simulator.
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # cells with value < 0.5 are free

        self.map_set = True
        self.node.get_logger().info("Map initialized")

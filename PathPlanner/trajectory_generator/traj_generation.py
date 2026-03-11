import math
import numpy as np

class TrajGenerator:
    """
    Class for generating trajectories by resampling and interpolating a given path.
    """
    def __init__(self):
        # Basic configuration for trajectory generation
        self.time_step = 0.1  # Time step for interpolation
        self.max_acceleration = 1  # Maximum allowable acceleration
        self.max_velocity = 1  # Maximum allowable velocity
        self.omega_max = 2.8  # Maximum angular velocity
        self.min_nfe = 20  # Minimum number of trajectory points

    def resample_path(self, path):
        """
        Resample a given path into a full trajectory with detailed state information.

        Args:
            path (list): List of tuples representing the path (x, y, theta).

        Returns:
            FullStates: Resampled trajectory containing detailed states.
        """
        max_accel = self.max_acceleration
        max_velocity = self.max_velocity
        max_omega_vel = self.omega_max
        result = FullStates()  # Container for full states
        gears = [0] * len(path)  # Forward (1) or reverse (-1) gear
        stations = [0.0] * len(path)  # Cumulative distance along the path
        
        # TODO: Calculate gear and station (distance) information
        # YOUR CODE STARTS HERE
        if not path:
            return result

        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]
            segment = self.distance(prev[:2], curr[:2])
            stations[i] = stations[i - 1] + segment

            heading = math.atan2(curr[1] - prev[1], curr[0] - prev[0]) if segment > 1e-9 else math.radians(prev[2])
            orientation = math.radians(prev[2])
            direction = math.cos(heading - orientation)
            gears[i] = 1 if direction >= 0 else -1

        if len(gears) > 1:
            gears[0] = gears[1]
        else:
            gears[0] = 1
        # YOUR CODE ENDS HERE

        # Calculate the time profile for the trajectory
        # YOUR CODE STARTS HERE
        if stations[-1] <= 1e-9:
            single_state = State()
            x, y, theta = path[0]
            single_state.x = x
            single_state.y = y
            single_state.theta = math.radians(theta)
            result.states.append(single_state)
            return result
        # YOUR CODE ENDS HERE

        time_profile = [0.0] * len(gears)
        last_idx = 0
        start_time = 0.0

        for i in range(len(gears)):
            if i == len(gears) - 1 or gears[i+1] != gears[i]:
                station_segment = stations[last_idx:i+1]
                profile = self.generate_optimal_time_profile_segment(station_segment, start_time)
                time_profile[last_idx:i+1] = profile
                start_time = profile[-1]
                last_idx = i + 1

        velocity_profile = [0.0] * len(path)
        for i in range(1, len(path)):
            dt_profile = time_profile[i] - time_profile[i - 1]
            ds_profile = stations[i] - stations[i - 1]
            if dt_profile > 1e-9:
                velocity_profile[i] = gears[i] * min(max_velocity, ds_profile / dt_profile)
            else:
                velocity_profile[i] = 0.0

        if len(path) > 1:
            velocity_profile[0] = velocity_profile[1]
            velocity_profile[-1] = 0.0
        
        # Interpolate the trajectory to ensure uniform time steps
        nfe = max(self.min_nfe, int(time_profile[-1] / self.time_step))
        interpolated_ticks = np.linspace(time_profile[0], time_profile[-1], nfe)

        # Prepare for 1D interpolation
        prev_x = [p[0] for p in path]
        prev_y = [p[1] for p in path]
        prev_theta = [p[2] for p in path]

        result.tf = interpolated_ticks[-1]
        interp_x = self.interpolate_1d(time_profile, prev_x, interpolated_ticks)
        interp_y = self.interpolate_1d(time_profile, prev_y, interpolated_ticks)
        interp_theta = self.to_continuous_angle(self.interpolate_1d(time_profile, prev_theta, interpolated_ticks))
        interp_v = self.interpolate_1d(time_profile, velocity_profile, interpolated_ticks)

        dt = interpolated_ticks[1] - interpolated_ticks[0] # Time step

        # Initialize the trajectory with interpolated states
        for i in range(nfe):
            state = State()
            state.x = interp_x[i]
            state.y = interp_y[i]
            state.theta = interp_theta[i]
            state.v = interp_v[i]
            state.a = 0.0
            state.omega = 0.0
            result.states.append(state)

        # TODO: Calculate velocities and angular velocities for the trajectory
        # YOUR CODE STARTS HERE
        for i in range(1, nfe):
            dtheta = self.normalize_angle(result.states[i].theta - result.states[i - 1].theta)
            result.states[i].omega = dtheta / dt if dt > 1e-9 else 0.0

        if nfe > 1:
            result.states[0].v = result.states[1].v
            result.states[0].omega = result.states[1].omega
        # YOUR CODE ENDS HERE

        # TODO: Calculate accelerations and angular accelerations
        # YOUR CODE STARTS HERE
        for i in range(1, nfe):
            result.states[i].a = (result.states[i].v - result.states[i - 1].v) / dt if dt > 1e-9 else 0.0

        if nfe > 1:
            result.states[0].a = result.states[1].a
        # YOUR CODE ENDS HERE

        return result

    def generate_optimal_time_profile_segment(self, stations, start_time):
        """
        Generate the optimal time profile for a segment of the path.

        Args:
            stations (list): Cumulative distances along the path segment.
            start_time (float): Starting time for the segment.

        Returns:
            list: Time profile for the segment.
        """
        max_accel = self.max_acceleration
        max_decel = -max_accel
        max_velocity = self.max_velocity
        min_velocity = -max_velocity

        accel_idx = 0
        decel_idx = len(stations) - 1
        vi = 0.0 # Initial velocity
        profile = [0.0] * len(stations)
        total_distance = stations[-1] - stations[0]

        if len(stations) <= 1 or total_distance <= 1e-9:
            return [start_time] * len(stations)

        accel_distance = max_velocity * max_velocity / (2.0 * max_accel)
        decel_distance = max_velocity * max_velocity / (2.0 * abs(max_decel))

        if accel_distance + decel_distance <= total_distance:
            peak_velocity = max_velocity
            cruise_distance = total_distance - accel_distance - decel_distance
        else:
            peak_velocity = math.sqrt(
                max(
                    0.0,
                    2.0 * total_distance / ((1.0 / max_accel) + (1.0 / abs(max_decel)))
                )
            )
            accel_distance = peak_velocity * peak_velocity / (2.0 * max_accel)
            decel_distance = peak_velocity * peak_velocity / (2.0 * abs(max_decel))
            cruise_distance = 0.0

        # TODO: Implement acceleration phase
        # YOUR CODE STARTS HERE
        for i in range(len(stations) - 1):
            local_s = stations[i + 1] - stations[0]
            if local_s <= accel_distance + 1e-9:
                profile[i + 1] = min(peak_velocity, math.sqrt(max(0.0, 2.0 * max_accel * local_s)))
                accel_idx = i + 1
                vi = profile[i + 1]
            else:
                break
        # YOUR CODE ENDS HERE

        # TODO: Implement deceleration phase
        vi = 0.0
        # YOUR CODE STARTS HERE
        for i in range(len(stations) - 1, accel_idx, -1):
            remaining_s = stations[-1] - stations[i - 1]
            if remaining_s <= decel_distance + 1e-9:
                decel_speed = min(peak_velocity, math.sqrt(max(0.0, 2.0 * abs(max_decel) * remaining_s)))
                if profile[i - 1] == 0.0:
                    profile[i - 1] = decel_speed
                else:
                    profile[i - 1] = min(profile[i - 1], decel_speed)
                vi = profile[i - 1]
                decel_idx = i - 1
        # YOUR CODE ENDS HERE

        # TODO: Fill constant velocity phase
        # YOUR CODE STARTS HERE
        for i in range(accel_idx + 1, decel_idx):
            if stations[i] - stations[0] <= accel_distance + cruise_distance:
                profile[i] = peak_velocity

        if len(profile) > 0:
            profile[0] = 0.0
            profile[-1] = 0.0
        # YOUR CODE ENDS HERE

        # Time profile calculation
        time_profile = [start_time] * len(stations)
        for i in range(1, len(stations)):
            if profile[i] < 1e-6:
                time_profile[i] = time_profile[i - 1]
            else:
                time_profile[i] = time_profile[i - 1] + (stations[i] - stations[i - 1]) / profile[i]

        return time_profile

    def path_interpolation(self, path, graph, lattice_cell_size, arc_length):
        """
        Interpolate a path into finer segments based on lattice and arc primitives.

        Args:
            path (list): List of path points (row, col, angle).
            graph: The lattice graph containing arc primitives.
            lattice_cell_size (int): Size of each lattice cell.
            arc_length (int): Number of points to sample along an arc.

        Returns:
            list: Interpolated path as a list of (x, y, theta).
        """
        cell_size = 3
        sampled_path = []
        for i in range(0, len(path)-1):
            v1 = path[i]
            v2 = path[i+1]
            dir_row = lattice_cell_size * (v2[0] - v1[0])
            dir_col = lattice_cell_size * (v2[1] - v1[1])
            if path[i][2] == path[i+1][2]:
                for j in range(lattice_cell_size):
                    row = v1[0] * lattice_cell_size + j * dir_row / lattice_cell_size
                    col = v1[1] * lattice_cell_size + j * dir_col / lattice_cell_size
                    sampled_path.append((row / cell_size, col / cell_size, v1[2]))
            else:
                arc = graph.arc_primitives[(v1[2], v2[2])]
                arc = np.array(v1[:2]).reshape((2, 1)) * lattice_cell_size + arc
                for j in range(arc.shape[1]):
                    sampled_path.append((arc[0, j] / cell_size, arc[1, j] / cell_size, v1[2] + (v2[2] - v1[2]) / arc_length * j))
        return sampled_path

    def interpolate_1d(self, x, y, t):
        """
        Perform 1D linear interpolation.

        Args:
            x (list): Known x values.
            y (list): Known y values.
            t (list or float): Values to interpolate.

        Returns:
            list or float: Interpolated values.
        """
        if isinstance(t, (float, int)):
            # Single value interpolation
            if t >= x[-1]:
                return y[-1]

            index = np.searchsorted(x, t, side='left')

            if index == 0:
                return y[0]
            
            if abs(x[index] - x[index - 1]) <= 1.0e-6:
                return y[index - 1]
            r = (t - x[index - 1]) / (x[index] - x[index - 1])
            
            return y[index - 1] + r * (y[index] - y[index - 1])
            
        # Multiple values interpolation
        result = []
        for ti in t:
            result.append(self.interpolate_1d(x, y, ti))
        return result

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].

        Args:
            angle (float): Input angle in radians.

        Returns:
            float: Normalized angle.
        """
        # TODO: Normalize an angle to the range [-pi, pi]
        # YOUR CODE STARTS HERE
        return math.atan2(math.sin(angle), math.cos(angle))
        # YOUR CODE ENDS HERE

    def distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points.

        Args:
            p1 (list): First point [x, y].
            p2 (list): Second point [x, y].

        Returns:
            float: Distance between the points.
        """
        # TODO: Calculate Euclidean distance between two points
        # YOUR CODE STARTS HERE
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        # YOUR CODE ENDS HERE

    def to_continuous_angle(self, angles):
        """
        Convert a list of angles into a continuous angle representation.

        Args:
            angles (list): List of angles in radians.

        Returns:
            list: Continuous angle representation.
        """
        # TODO: Convert a list of angles into a continuous angle representation.
        # YOUR CODE STARTS HERE
        if not angles:
            return []

        continuous_angles = [math.radians(angles[0])]
        for angle_deg in angles[1:]:
            angle = math.radians(angle_deg)
            delta = self.normalize_angle(angle - continuous_angles[-1])
            continuous_angles.append(continuous_angles[-1] + delta)
        return continuous_angles
        # YOUR CODE ENDS HERE


class FullStates:
    """
    Class representing the full trajectory states for a path.

    Attributes:
        tf (float): Final time of the trajectory.
        states (list): List of individual states representing the trajectory.
    """
    def __init__(self):
        self.tf = 0.0  # Initialize final time as 0.0
        self.states = []  # Initialize an empty list to hold trajectory states


class State:
    """
    Class representing a single state in a trajectory.

    Attributes:
        x (float): X-coordinate of the state.
        y (float): Y-coordinate of the state.
        theta (float): Orientation (angle in radians) of the state.
        v (float): Linear velocity of the state.
        a (float): Linear acceleration of the state.
        omega (float): Angular velocity of the state.
    """
    def __init__(self):
        self.x = 0.0  # Initialize X-coordinate
        self.y = 0.0  # Initialize Y-coordinate
        self.theta = 0.0  # Initialize orientation
        self.v = 0.0  # Initialize linear velocity
        self.a = 0.0  # Initialize linear acceleration
        self.omega = 0.0  # Initialize angular velocity


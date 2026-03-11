import rclpy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from path_planner.lattice_planner import LatticeGraph
from path_planner.rrt_planner import RRTPlanner
from path_planner.prm_planner import PRMPlanner
from path_planner.utils import ObstaclesGrid
from path_planner.utils import *
from trajectory_generator.traj_generation import TrajGenerator
import matplotlib.pyplot as plt
from path_planner.utils import *


def scale_points(path_points, scaler):
    if not path_points:
        return [], []

    rows = [point[0] / scaler for point in path_points]
    cols = [point[1] / scaler for point in path_points]
    return rows, cols


def main(args=None):
    # Initialize the graph and trajectory generator
    graph = LatticeGraph()
    traj_generator = TrajGenerator()

    # Define the graph dimensions and scaling
    n_rows = 10
    n_cols = 10
    lattice_cell_size = 10
    map_size = (n_rows * lattice_cell_size, n_cols * lattice_cell_size)
    scaler = 3.0
    graph.initialise_graph(n_rows=n_rows, n_cols=n_cols, lattice_cell_size=lattice_cell_size)

    # Define the start and goal positions with orientation
    s_3d = (1, 8, 90)  # Start position in lattice coordinates: row, col, angle
    g_3d = (8, 2, 270)  # Goal position in lattice coordinates: row, col, angle
    s_2d = (s_3d[0] * lattice_cell_size, s_3d[1] * lattice_cell_size)  # Start position in map coordinates
    g_2d = (g_3d[0] * lattice_cell_size, g_3d[1] * lattice_cell_size)  # Goal position in map coordinates

    # Initialize obstacle grids
    obs = ObstaclesGrid(map_size=(n_rows * lattice_cell_size, n_cols * lattice_cell_size))
    obs_plot = ObstaclesGrid(map_size=(int(n_rows * lattice_cell_size / scaler), int(n_cols * lattice_cell_size / scaler)))

    # Add obstacles to the map
    obs.map[25:35, 45:56] = True
    obs.map[40:43, 48:76] = True
    obs.map[67:89, 57:76] = True
    obs.map[50:55, 60:89] = True
    obs.map[20:60, 5:35] = True

    # Add scaled obstacles for plotting
    obs_plot.map[int((25 / scaler)):int((35 / scaler)), int((45 / scaler)):int((56 / scaler))] = True
    obs_plot.map[int((40 / scaler)):int((43 / scaler)), int((48 / scaler)):int((76 / scaler))] = True
    obs_plot.map[int((67 / scaler)):int((89 / scaler)), int((57 / scaler)):int((76 / scaler))] = True
    obs_plot.map[int((50 / scaler)):int((55 / scaler)), int((60 / scaler)):int((89 / scaler))] = True
    obs_plot.map[int((20 / scaler)):int((60 / scaler)), int((5 / scaler)):int((35 / scaler))] = True

    # Update the graph with obstacle information
    graph.update_obstacles(obs)

    # Find a path from start to goal using lattice planner
    path = graph.solve(s_3d, g_3d, graph._graph._vert_list, graph._graph._adjacency_matrix, graph._graph._edge_dict)

    # Find a path from start to goal using rrt planner
    rrt = RRTPlanner(s_2d, g_2d, map_size, obs)
    path_rrt = rrt.plan()

    # Find a path from start to goal using prm planner
    prm = PRMPlanner(s_2d, g_2d, map_size, obs)
    prm.construct_roadmap()
    path_prm = prm.plan()

    if path:
        # Interpolate the path for smoothness
        path_interpolated = traj_generator.path_interpolation(path, graph, lattice_cell_size, 10)

        # Resample the interpolated path to generate a trajectory
        result = traj_generator.resample_path(path_interpolated)

        # Print the number of states in the trajectory
        print("trajectory length = ", len(result.states))

        traj_x = [state.x for state in result.states]
        traj_y = [state.y for state in result.states]
        traj_v = [state.v for state in result.states]
        traj_t = [i * traj_generator.time_step for i in range(len(result.states))]

        plt.figure("Velocity-Time")
        plt.plot(traj_t, traj_v, color='tab:blue', linewidth=2.0, label='Lattice velocity')
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity")
        plt.title("Velocity-Time Curve")
        plt.grid(True)
        plt.legend()

    # Plot the obstacle map and trajectory
    ax = plot_map(obs_plot, graph, lattice_cell_size)

    if path:
        ax.plot(traj_y, traj_x, color='green', linewidth=2.0, label='Lattice path')

    if path_rrt:
        rrt_rows, rrt_cols = scale_points(path_rrt, scaler)
        ax.plot(rrt_cols, rrt_rows, color='tab:red', linewidth=2.0, linestyle='--', label='RRT path')

    if path_prm:
        prm_rows, prm_cols = scale_points(path_prm, scaler)
        ax.plot(prm_cols, prm_rows, color='tab:orange', linewidth=2.0, linestyle='-.', label='PRM path')

    ax.scatter(s_2d[1] / scaler, s_2d[0] / scaler, color='tab:blue', s=80, marker='o', label='Start')
    ax.scatter(g_2d[1] / scaler, g_2d[0] / scaler, color='black', s=80, marker='x', label='Goal')
    ax.set_title("Path Planning Results")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend()

    plt.show()

if __name__ == '__main__':
    main()

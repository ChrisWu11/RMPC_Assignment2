from path_planner import utils
from queue import PriorityQueue
import numpy as np


class LatticeGraph:
    """
    Class representing a lattice graph for path planning.
    """

    def __init__(self):
        self._graph = utils.Graph()  # Initialize the graph structure
        self.solver = Astar()  # Use A* algorithm for solving
        self._nrows = None  # Number of rows in the grid
        self._n_cols = None  # Number of columns in the grid
        self.lattice_cell_size = None  # Size of each cell in the lattice
        self.arc_primitives = {}  # Precomputed arc primitives for curved paths

    def initialise_graph(self, n_rows=10, n_cols=10, lattice_cell_size=10):
        """
        Initialize the graph with lattice structure.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            lattice_cell_size (int): Size of each lattice cell.
        """
        self._nrows = n_rows
        self._n_cols = n_cols
        self.lattice_cell_size = lattice_cell_size
        self.generate_lattice(n_rows, n_cols, lattice_cell_size)
        self._graph.set_adjacency_matrix()

    def update_obstacles(self, obs):
        """
        Update the graph by invalidating edges that intersect with obstacles.

        Args:
            obs: Obstacle grid object to check edge validity.
        """
        for edge_key, edge_val in self._graph._edge_dict.items():
            is_valid = obs.is_edge_valid(edge_key, edge_val, self.lattice_cell_size, self.arc_primitives)

            if not is_valid:
                self._graph._edge_dict[edge_key] = np.inf

        self._graph.set_adjacency_matrix()

    def solve(self, s, g, graph_vert_list, adjacency_matrix, edge_dict):
        """
        Solve for a path from start to goal using the A* algorithm.

        Args:
            s (tuple): Start vertex (row, col, angle).
            g (tuple): Goal vertex (row, col, angle).
            graph_vert_list (list): List of graph vertices.
            adjacency_matrix (ndarray): Adjacency matrix of the graph.
            edge_dict (dict): Dictionary of edge weights.

        Returns:
            list: Path from start to goal as a list of vertices.
        """
        path = self.solver.solve_astar(s, g, graph_vert_list, adjacency_matrix, edge_dict)
        return path

    def generate_lattice(self, n_rows, n_cols, lattice_cell_size):
        """
        Generate lattice vertices and edges for the graph.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            lattice_cell_size (int): Size of each lattice cell.
        """
        for row in range(n_rows):
            for col in range(n_cols):
                for angle in [0, 90, 180, 270]:
                    v = (row, col, angle)
                    self._graph.add_vertex(v)

        # construct lattice graph
        for row in range(n_rows):
            for col in range(n_cols):
                for angle in [0, 90, 180, 270]:

                    v = (row, col, angle)

                    # top row
                    if (row - 1) >= 0 and angle == 90:
                        v_top = (row - 1, col, 90)

                    if (col - 1) >= 0 and (row - 1) >= 0 and angle == 90:
                        v_top_left = (row - 1, col - 1, 180)
                        self._graph.set_edge(v, v_top_left, np.pi)

                    if (col + 1) < n_cols and (row - 1) >= 0 and angle == 90:
                        v_top_right = (row - 1, col + 1, 0)
                        self._graph.set_edge(v, v_top_right, np.pi)

                    # buttom row
                    if (row + 1) < n_rows and angle == 270:
                        v_buttom = (row + 1, col, 270)
                        self._graph.set_edge(v, v_buttom, 1)

                    if (col - 1) >= 0 and (row + 1) < n_rows and angle == 270:
                        v_buttom_left = (row + 1, col - 1, 180)
                        self._graph.set_edge(v, v_buttom_left, np.pi)

                    if (col + 1) < n_cols and (row + 1) < n_rows and angle == 270:
                        v_buttom_right = (row + 1, col + 1, 0)
                        self._graph.set_edge(v, v_buttom_right, np.pi)

                    # left col
                    if (col - 1) >= 0 and angle == 180:
                        v_left = (row, col - 1, 180)
                        self._graph.set_edge(v, v_left, 1)

                    if (col - 1) >= 0 and (row - 1) >= 0 and angle == 180:
                        v_left_up = (row - 1, col - 1, 90)
                        self._graph.set_edge(v, v_left_up, np.pi)

                    if (col - 1) >= 0 and (row + 1) < n_rows and angle == 180:
                        v_left_down = (row + 1, col - 1, 270)
                        self._graph.set_edge(v, v_left_down, np.pi)

                    # right col
                    if (col + 1) < n_cols and angle == 0:
                        v_right = (row, col + 1, 0)
                        self._graph.set_edge(v, v_right, 1)

                    if (col + 1) < n_cols and (row - 1) >= 0 and angle == 0:
                        v_right_up = (row - 1, col + 1, 90)
                        self._graph.set_edge(v, v_right_up, np.pi)

                    if (col + 1) < n_cols and (row + 1) < n_rows and angle == 0:
                        v_right_down = (row + 1, col + 1, 270)
                        self._graph.set_edge(v, v_right_down, np.pi)

        # arcs
        npoints = int(lattice_cell_size * np.pi / 2 )
        pts_0_to_90 = np.zeros((2, npoints))
        for i in range(npoints):
            x = np.cos(float(i) / float(npoints - 1) * np.pi / 2) * lattice_cell_size - lattice_cell_size
            y = np.sin(float(i) / float(npoints - 1) * np.pi / 2) * lattice_cell_size
            pts_0_to_90[0, i] = x
            pts_0_to_90[1, i] = y

        self.arc_primitives[(0, 90)] = pts_0_to_90

        pts_0_to_270 = np.zeros((2, npoints))
        pts_0_to_270[0, :] = -1*pts_0_to_90[0, :]
        pts_0_to_270[1, :] = pts_0_to_90[1, :]
        self.arc_primitives[(0, 270)] = pts_0_to_270

        pts_270_to_180 = np.zeros((2, npoints))
        pts_270_to_180[0, :] = np.flip(pts_0_to_90[0, :]) + lattice_cell_size
        pts_270_to_180[1, :] = np.flip(pts_0_to_90[1, :]) - lattice_cell_size
        self.arc_primitives[(270, 180)] = pts_270_to_180

        pts_90_to_180 = np.zeros((2, npoints))
        pts_90_to_180[0, :] = np.flip(pts_0_to_270[0, :]) - lattice_cell_size
        pts_90_to_180[1, :] = np.flip(pts_0_to_270[1, :]) - lattice_cell_size
        self.arc_primitives[(90, 180)] = pts_90_to_180

        pts_90_to_0 = np.zeros((2, npoints))
        pts_90_to_0[0, :] = pts_90_to_180[0, :]
        pts_90_to_0[1, :] = -1*pts_90_to_180[1, :]
        self.arc_primitives[(90, 0)] = pts_90_to_0

        pts_180_to_270 = np.zeros((2, npoints))
        pts_180_to_270[0, :] = np.flip(pts_90_to_0[0, :]) + lattice_cell_size
        pts_180_to_270[1, :] = np.flip(pts_90_to_0[1, :]) - lattice_cell_size
        self.arc_primitives[(180, 270)] = pts_180_to_270

        pts_270_to_0 = np.zeros((2, npoints))
        pts_270_to_0[0, :] = -1*np.flip(pts_180_to_270[0, :]) + lattice_cell_size
        pts_270_to_0[1, :] = np.flip(pts_180_to_270[1, :]) + lattice_cell_size
        self.arc_primitives[(270, 0)] = pts_270_to_0

        pts_180_to_90 = np.zeros((2, npoints))
        pts_180_to_90[0, :] = np.flip(pts_270_to_0[0, :]) - lattice_cell_size
        pts_180_to_90[1, :] = np.flip(pts_270_to_0[1, :]) - lattice_cell_size
        self.arc_primitives[(180, 90)] = pts_180_to_90

class Astar:
    def solve_astar(self, s, g, graph_vert_list, adjacency_matrix, edge_dict):
        """
        Solve the shortest path problem using A*.

        Args:
            s (tuple): Start vertex.
            g (tuple): Goal vertex.
            graph_vert_list (list): List of graph vertices.
            adjacency_matrix (ndarray): Adjacency matrix.
            edge_dict (dict): Dictionary of edge weights.

        Returns:
            list: Path from start to goal as a list of vertices.
        """
        open_set = PriorityQueue()

        closed_set = set()
        parent_node = dict()
        distances = dict()
        costs = dict()
        open_set.put((0, s))
        distances[s] = 0.0
        costs[s] = self.calH(s, g)

        while not open_set.empty():
            _, current = open_set.get()

            if current in closed_set:
                continue

            if current == g:
                return self.traverse_path(s, g, parent_node)

            closed_set.add(current)

            for neighbor in self.get_neighbor(current, graph_vert_list, adjacency_matrix):
                if neighbor in closed_set:
                    continue

                expand_cost = self.cal_expand_cost(current, neighbor, edge_dict)
                if not np.isfinite(expand_cost):
                    continue

                tentative_distance = distances[current] + expand_cost
                if tentative_distance < distances.get(neighbor, np.inf):
                    parent_node[neighbor] = current
                    distances[neighbor] = tentative_distance
                    total_cost = tentative_distance + self.calH(neighbor, g)
                    costs[neighbor] = total_cost
                    open_set.put((total_cost, neighbor))

        return []

    def traverse_path(self, s, g, parent_node):
        """
        Reconstruct the path from start to goal using parent nodes.

        Args:
            s (tuple): Start vertex.
            g (tuple): Goal vertex.
            parent_node (dict): Dictionary mapping each vertex to its parent.

        Returns:
            list: Path from start to goal.
        """
        if s == g:
            return [s]

        if g not in parent_node:
            return []

        path = [g]
        current = g
        while current != s:
            current = parent_node[current]
            path.append(current)

        path.reverse()
        return path

    def get_neighbor(self, u, graph_vert_list, adjacency_matrix):

        row = graph_vert_list.index(u)
        is_adj = (adjacency_matrix[row, :] < np.inf) & (adjacency_matrix[row, :] > 0)

        adj_list = []
        for i, v in enumerate(graph_vert_list):
            if is_adj[i]:
                adj_list.append(v)

        return adj_list

    def cal_expand_cost(self, v1, v2, edge_dict):
        """
        Calculate the cost of expanding from one vertex to another.

        Args:
            v1 (tuple): Current vertex.
            v2 (tuple): Next vertex.
            edge_dict (dict): Dictionary of edge weights.

        Returns:
            float: Cost of expansion.
        """
        return edge_dict.get((v1, v2), np.inf)

    def calH(self, v1, v2):
        """
        Calculate the heuristic cost.

        Args:
            v1 (tuple): Current vertex.
            v2 (tuple): Goal vertex.

        Returns:
            float: Heuristic cost.
        """
        return np.hypot(v1[0] - v2[0], v1[1] - v2[1])

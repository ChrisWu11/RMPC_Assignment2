import numpy as np
import matplotlib.pyplot as plt
from path_planner.utils import ObstaclesGrid

try:
    from scipy.spatial import KDTree
except ImportError:  # pragma: no cover - optional dependency fallback
    KDTree = None

class Node:
    def __init__(self, x, y):
        """
        Represents a node in the PRM roadmap.

        Args:
            x (float): X-coordinate of the node.
            y (float): Y-coordinate of the node.
        """
        self.x = x
        self.y = y

class PRMPlanner:
    def __init__(self, start, goal, map_size, obstacles, num_samples=200, k_neighbors=10, step_size=5):
        """
        Initializes the PRM planner.

        Args:
            start (tuple): (x, y) coordinates of the start position.
            goal (tuple): (x, y) coordinates of the goal position.
            map_size (tuple): (width, height) of the environment.
            obstacles (ObstaclesGrid): Object that stores obstacle information.
            num_samples (int): Number of random samples for roadmap construction.
            k_neighbors (int): Number of nearest neighbors to connect in the roadmap.
            step_size (float): Step size used for collision checking.
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_size = map_size
        self.obstacles = obstacles
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.step_size = step_size
        self.roadmap = [] 
        self.edges = {} 

    def construct_roadmap(self):
        """
        Constructs the probabilistic roadmap by sampling nodes and connecting them.

        Returns:
        None
        """
        self.roadmap = [self.start, self.goal]
        self.edges = {self.start: [], self.goal: []}

        while len(self.roadmap) < self.num_samples + 2:
            node = self.sample_free_point()
            self.roadmap.append(node)
            self.edges[node] = []

        for node in self.roadmap:
            neighbors = self.find_k_nearest(node, self.k_neighbors)
            for neighbor in neighbors:
                if neighbor is node:
                    continue
                if not self.is_colliding(node, neighbor):
                    if neighbor not in self.edges[node]:
                        self.edges[node].append(neighbor)
                    if node not in self.edges[neighbor]:
                        self.edges[neighbor].append(node)

    def sample_free_point(self):
        """
        Samples a random collision-free point in the environment.

        Returns:
        Node: A randomly sampled node.
        """

        while True:
            x = np.random.uniform(0, self.map_size[0] - 1)
            y = np.random.uniform(0, self.map_size[1] - 1)
            if not self.obstacles.map[int(x), int(y)]:
                return Node(x, y)

    def find_k_nearest(self, node, k):
        """
        Finds the k-nearest neighbors of a node in the roadmap.

        Args:
            node (Node): The node for which neighbors are searched.
            k (int): The number of nearest neighbors to find.

        Returns:
            list: A list of k-nearest neighbor nodes.
        """

        if KDTree is not None:
            points = np.array([(roadmap_node.x, roadmap_node.y) for roadmap_node in self.roadmap])
            tree = KDTree(points)
            query_k = min(k + 1, len(self.roadmap))
            _, indices = tree.query([node.x, node.y], k=query_k)
            indices = np.atleast_1d(indices)

            neighbors = []
            for idx in indices:
                candidate = self.roadmap[int(idx)]
                if candidate is not node:
                    neighbors.append(candidate)
            return neighbors[:k]

        distances = []
        for candidate in self.roadmap:
            if candidate is node:
                continue
            dist = np.hypot(node.x - candidate.x, node.y - candidate.y)
            distances.append((dist, candidate))
        distances.sort(key=lambda item: item[0])
        return [candidate for _, candidate in distances[:k]]

    def is_colliding(self, node1, node2):
        """
        Checks if the path between two nodes collides with an obstacle.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            bool: True if there is a collision, False otherwise.
        """

        dx = node2.x - node1.x
        dy = node2.y - node1.y
        distance = np.hypot(dx, dy)
        n_steps = max(1, int(np.ceil(distance / max(self.step_size, 1e-6))))

        for i in range(n_steps + 1):
            ratio = i / n_steps
            x = node1.x + dx * ratio
            y = node1.y + dy * ratio
            xi = int(np.clip(round(x), 0, self.map_size[0] - 1))
            yi = int(np.clip(round(y), 0, self.map_size[1] - 1))
            if self.obstacles.map[xi, yi]:
                return True

        return False

    def plan(self):
        """
        Plans a path from start to goal using the constructed roadmap.

        Returns:
        list: A list of (x, y) tuples representing the path.
        """

        if not self.edges:
            self.construct_roadmap()

        open_set = [(0.0, self.start)]
        parent = {self.start: None}
        g_cost = {self.start: 0.0}
        visited = set()

        while open_set:
            open_set.sort(key=lambda item: item[0])
            _, current = open_set.pop(0)

            if current in visited:
                continue
            visited.add(current)

            if current is self.goal:
                path = []
                while current is not None:
                    path.append((current.x, current.y))
                    current = parent[current]
                path.reverse()
                return path

            for neighbor in self.edges.get(current, []):
                if neighbor in visited:
                    continue
                step_cost = np.hypot(neighbor.x - current.x, neighbor.y - current.y)
                tentative_cost = g_cost[current] + step_cost
                if tentative_cost < g_cost.get(neighbor, np.inf):
                    g_cost[neighbor] = tentative_cost
                    parent[neighbor] = current
                    heuristic = np.hypot(neighbor.x - self.goal.x, neighbor.y - self.goal.y)
                    open_set.append((tentative_cost + heuristic, neighbor))

        print("Path not found.")
        return None

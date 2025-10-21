from alg_util import timeit
import time
from tqdm import tqdm

class DepthFirstSearch:
    """
    Naive DFS-based SCC finder (O(V * (V + E))).
    Used as a baseline for comparison against Tarjan’s algorithm.
    """
    def __init__(self, graph, is_directed=True, weight=None, is_visualize=False):
        self.graph = graph
        self.isdirected = True
        self.weight = weight
        self.isvisualize = is_visualize

    def _get_transpose(self):
        """Compute the transpose of the directed graph."""
        try:
            from benchmark import Graph
        except ImportError:
            # fallback Graph class
            class Graph:
                def __init__(self):
                    self.adj = {}
                    self.nodes = set()
                def add_edge(self, u, v):
                    if u not in self.adj:
                        self.adj[u] = []
                    self.adj[u].append(v)
                    self.nodes.add(u)
                    self.nodes.add(v)

        g_transpose = Graph()
        nodes = self.graph.get_nodes()
        for node in nodes:
            g_transpose.nodes.add(node)
            if node not in g_transpose.adj:
                g_transpose.adj[node] = []

        for u in nodes:
            for v in self.graph.get_neighbors(u):
                g_transpose.add_edge(v, u)
        return g_transpose

    def _dfs_traversal(self, start_node, graph_obj):
        """Iterative DFS to find all reachable nodes."""
        reachable_nodes = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node not in reachable_nodes:
                reachable_nodes.add(node)
                for neighbor in graph_obj.get_neighbors(node):
                    if neighbor not in reachable_nodes:
                        stack.append(neighbor)
        return reachable_nodes

    @timeit
    def run(self):
        start_time = time.time()
        strongly_connected_components = []
        nodes_already_in_scc = set()

        graph_transpose = self._get_transpose()
        all_nodes = list(self.graph.get_nodes())

        with tqdm(total=len(all_nodes), desc="Naive DFS SCC", unit="node", mininterval=1.0, colour="magenta") as pbar:
            for node in all_nodes:
                if node in nodes_already_in_scc:
                    pbar.update(1)
                    continue

                # 1. Find all nodes reachable from this node
                descendants = self._dfs_traversal(node, self.graph)
                # 2. Find all nodes that can reach this node
                ancestors = self._dfs_traversal(node, graph_transpose)
                # 3. Intersection = SCC
                current_scc = list(descendants.intersection(ancestors))
                strongly_connected_components.append(current_scc)

                # 4. Mark nodes to skip in future
                nodes_already_in_scc.update(current_scc)

                pbar.update(1)

        elapsed = time.time() - start_time
        return strongly_connected_components, elapsed

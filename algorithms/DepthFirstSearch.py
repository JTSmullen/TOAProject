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
            from graph_util import Graph
        except ImportError:
            # fallback Graph class
            class Graph:
                def __init__(self):
                    self.adj = {}
                    self.nodes = set()
                def add_edge(self, u, v):
                    self.adj.setdefault(u, []).append(v)
                    self.adj.setdefault(v, [])
                    self.nodes.update([u, v])
                def get_neighbors(self, node):
                    return self.adj.get(node, [])
                def get_nodes(self):
                    return list(self.nodes)

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

    def _dfs_traversal(self, start_node, graph_obj, yield_steps=False):
        """Wrapper for DFS traversal that can return a set or a generator."""
        if yield_steps:
            return self._dfs_traversal_generator(start_node, graph_obj)
        else:
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

    def _dfs_traversal_generator(self, start_node, graph_obj):
        """Generator for visualizing DFS traversal."""
        reachable_nodes = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in reachable_nodes:
                reachable_nodes.add(node)
                yield {'visited': set(reachable_nodes), 'current_node': node}
                for neighbor in graph_obj.get_neighbors(node):
                    if neighbor not in reachable_nodes:
                        stack.append(neighbor)

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

                descendants = self._dfs_traversal(node, self.graph, yield_steps=False)
                ancestors = self._dfs_traversal(node, graph_transpose, yield_steps=False)
                current_scc = list(descendants.intersection(ancestors))
                strongly_connected_components.append(current_scc)
                nodes_already_in_scc.update(current_scc)
                pbar.update(1)

        elapsed = time.time() - start_time
        return strongly_connected_components, elapsed

def dfs_scc_generator(graph):
    """Generator for the naive DFS-based SCC algorithm."""
    strongly_connected_components = []
    nodes_already_in_scc = set()

    # Step 1: Get graph transpose
    dfs_instance = DepthFirstSearch(graph)
    g_transpose = dfs_instance._get_transpose()
    yield {
        'description': 'Step 1: Computed graph transpose',
        'visited': set(),
        'scc': [],
        'all_sccs': []
    }

    all_nodes = list(graph.get_nodes())
    for node in all_nodes:
        if node in nodes_already_in_scc:
            continue

        # Step 2: Find descendants
        descendants = set()
        traversal_gen = dfs_instance._dfs_traversal(node, graph, yield_steps=True)
        for step in traversal_gen:
            descendants = step['visited']
            yield {
                'description': f'Finding descendants of {node}',
                'visited': descendants,
                'current_node': step['current_node'],
                'scc': [],
                'all_sccs': strongly_connected_components
            }
        
        # Step 3: Find ancestors
        ancestors = set()
        traversal_gen = dfs_instance._dfs_traversal(node, g_transpose, yield_steps=True)
        for step in traversal_gen:
            ancestors = step['visited']
            yield {
                'description': f'Finding ancestors of {node}',
                'visited': ancestors,
                'current_node': step['current_node'],
                'scc': [],
                'all_sccs': strongly_connected_components
            }

        # Step 4: Intersection is the SCC
        current_scc = list(descendants.intersection(ancestors))
        strongly_connected_components.append(current_scc)
        nodes_already_in_scc.update(current_scc)
        
        yield {
            'description': f'Found SCC: {current_scc}',
            'visited': set(),
            'scc': current_scc,
            'all_sccs': strongly_connected_components
        }

    yield {
        'description': 'Algorithm finished. All SCCs highlighted.',
        'visited': set(),
        'scc': [],
        'all_sccs': strongly_connected_components,
        'phase': 4 # Using phase 4 to signify final step
    }

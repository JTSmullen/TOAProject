from alg_util import timeit
import time
from tqdm import tqdm

class Tarjan:
    """
    Tarjan’s algorithm for finding strongly connected components (O(V + E)).
    """
    def __init__(self, graph, is_directed=True, weight=None, is_visualize=False):
        self.graph = graph
        self.isdirected = is_directed
        self.weight = weight
        self.isvisualize = is_visualize

    @timeit
    def run(self):
        if self.isvisualize:
            # Add the project root to the python path to allow importing visualize_tarjan
            import sys
            import os
            # Correctly identify the project root by going up one level from the 'algorithms' directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from visualize_tarjan import tarjan_visualizer_generator, visualize_tarjan
            import matplotlib.pyplot as plt

            steps = list(tarjan_visualizer_generator(self.graph))
            
            plt.ion()
            visualize_tarjan(self.graph, steps)
            plt.ioff()
            plt.show()
            
            # The original run method returns SCCs and elapsed time.
            # The visualizer generator does not directly return these, so we need to extract the final SCCs.
            final_sccs = []
            if steps:
                # Reconstruct the SCCs from the last step or by accumulating them
                for step in steps:
                    if step['description'].startswith("Found SCC"):
                        final_sccs.append(step['scc'])

            return final_sccs, 0 # Elapsed time is not measured in visualization mode.

        start_time = time.time()

        index_counter = 0
        stack = []
        lowlinks = {}
        indices = {}
        on_stack = set()
        strongly_connected_components = []

        all_nodes = list(self.graph.get_nodes())

        with tqdm(total=len(all_nodes), desc="Tarjan SCC", unit="node", mininterval=1.0, colour="cyan") as pbar:
            for v_start in all_nodes:
                if v_start not in indices:
                    traversal_stack = [(v_start, iter(self.graph.get_neighbors(v_start)))]

                    while traversal_stack:
                        node, neighbors_iterator = traversal_stack[-1]

                        if node not in indices:
                            indices[node] = index_counter
                            lowlinks[node] = index_counter
                            index_counter += 1
                            stack.append(node)
                            on_stack.add(node)

                        processed_all_neighbors = False
                        for neighbor in neighbors_iterator:
                            if neighbor not in indices:
                                traversal_stack.append((neighbor, iter(self.graph.get_neighbors(neighbor))))
                                break
                            elif neighbor in on_stack:
                                lowlinks[node] = min(lowlinks[node], indices[neighbor])
                        else:
                            processed_all_neighbors = True

                        if processed_all_neighbors:
                            traversal_stack.pop()
                            if traversal_stack:
                                parent, _ = traversal_stack[-1]
                                lowlinks[parent] = min(lowlinks[parent], lowlinks[node])

                            if lowlinks[node] == indices[node]:
                                scc = []
                                while True:
                                    w = stack.pop()
                                    on_stack.remove(w)
                                    scc.append(w)
                                    if w == node:
                                        break
                                strongly_connected_components.append(scc)

                pbar.update(1)

        elapsed = time.time() - start_time
        return strongly_connected_components, elapsed

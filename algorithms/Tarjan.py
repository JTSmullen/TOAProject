import util
import time

timeit = util.timeit

class Tarjan:
    def __init__(self, graph, is_directed=True, weight=None, is_visualize=False):
        self.graph = graph
        self.isdirected = is_directed
        self.weight = weight
        self.isvisualize = is_visualize

    @timeit
    def run(self):
        start_time = time.time()

        index = 0
        stack = []
        lowlinks = {}
        index_mapping = {}
        on_stack = set()
        strongly_connected_components = []

        def strongconnect(node):
            nonlocal index
            index_mapping[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in self.graph.get_neighbors(node):
                if neighbor not in index_mapping:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], index_mapping[neighbor])

            if lowlinks[node] == index_mapping[node]:
                connected_component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    connected_component.append(w)
                    if w == node:
                        break
                strongly_connected_components.append(connected_component)

        for v in self.graph.get_nodes():
            if v not in index_mapping:
                strongconnect(v)

        end_time = time.time()
        elapsed = end_time - start_time
        return strongly_connected_components, elapsed
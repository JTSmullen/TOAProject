import random

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

def generate_random_graph(num_vertices, edge_probability):
    graph = Graph()
    for i in range(num_vertices):
        graph.nodes.add(i)
        graph.adj[i] = []

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and random.random() < edge_probability:
                graph.add_edge(i, j)
    return graph

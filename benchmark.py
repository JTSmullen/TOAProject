import tracemalloc
import random
import sys
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow importing from the 'algorithms' package
sys.path.append('.')

# Import the algorithm classes to be tested
# NOTE: DepthFirstSearch.py should contain the TRULY NAIVE O(V*(V+E)) SCC algorithm
from algorithms.DepthFirstSearch import DepthFirstSearch
from algorithms.Tarjan import Tarjan

# --- Graph Representation and Generation ---

class Graph:
    """A simple graph class for generating test cases."""
    def __init__(self):
        self.adj = {}
        self.nodes = set()

    def add_edge(self, u, v):
        if u not in self.adj: self.adj[u] = []
        if v not in self.adj: self.adj[v] = []
        self.adj[u].append(v)
        self.nodes.add(u)
        self.nodes.add(v)

    def get_neighbors(self, node):
        return self.adj.get(node, [])

    def get_nodes(self):
        return list(self.nodes)

def generate_scc_graph(num_nodes, num_sccs, internal_density):
    """Generates a graph with a specified number of Strongly Connected Components."""
    graph = Graph()
    if num_sccs > num_nodes:
        num_sccs = num_nodes
    partitions = [[] for _ in range(num_sccs)]
    for i in range(num_nodes):
        partitions[i % num_sccs].append(i)
    for component_nodes in partitions:
        if not component_nodes: continue
        for i in range(len(component_nodes)):
            u, v = component_nodes[i], component_nodes[(i + 1) % len(component_nodes)]
            graph.add_edge(u, v)
        max_internal_edges = len(component_nodes) * (len(component_nodes) - 1)
        num_internal_edges = int(max_internal_edges * internal_density)
        for _ in range(num_internal_edges):
            u, v = random.choice(component_nodes), random.choice(component_nodes)
            if u != v: graph.add_edge(u, v)
    for i in range(num_sccs - 1):
        if partitions[i] and partitions[i+1]:
            u, v = random.choice(partitions[i]), random.choice(partitions[i+1])
            graph.add_edge(u, v)
    return graph

# --- Measurement Function ---

def run_and_measure(algorithm_class, graph):
    """Runs a given algorithm on a graph and measures its runtime and peak memory."""
    alg_instance = algorithm_class(graph)
    tracemalloc.start()
    _, elapsed_time = alg_instance.run()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed_time, peak_mem / 1024

# --- Main Execution ---

if __name__ == '__main__':
    # --- Configuration ---
    NODE_COUNTS = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 750, 900, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 20000] 
    SCC_RATIO = 0.1
    INTERNAL_DENSITY = 0.2
    
    # --- WARM-UP SECTION ---
    # We run the entire process once on a small graph to handle any one-time
    # costs like module imports, JIT compilation, or caching.
    # The results of this run are discarded.
    print("--- Performing a warm-up run to ensure consistent measurements... ---")
    WARMUP_NODE_COUNT = 10
    warmup_graph = generate_scc_graph(WARMUP_NODE_COUNT, max(1, int(WARMUP_NODE_COUNT * SCC_RATIO)), INTERNAL_DENSITY)
    run_and_measure(DepthFirstSearch, warmup_graph)
    run_and_measure(Tarjan, warmup_graph)
    print("--- Warm-up complete. Starting main benchmark. ---\n")
    
    # --- MAIN BENCHMARK LOOP ---
    print("--- Starting Algorithm Benchmark on Multi-SCC Graphs ---")
    results = []

    for n in NODE_COUNTS:
        num_sccs = max(1, int(n * SCC_RATIO))
        print(f"Generating and testing graph with {n} nodes and {num_sccs} SCCs...")
        
        graph = generate_scc_graph(n, num_sccs, INTERNAL_DENSITY)
        
        tarjan_time, tarjan_mem = run_and_measure(Tarjan, graph)
        naive_time, naive_mem = run_and_measure(DepthFirstSearch, graph)
        
        results.append({
            'nodes': n,
            'naive_time': naive_time, 'naive_mem_kb': naive_mem,
            'tarjan_time': tarjan_time, 'tarjan_mem_kb': tarjan_mem
        })
        
        print(f"  Naive SCC: {naive_time:.6f}s, {naive_mem:.2f} KB")
        print(f"  Tarjan:    {tarjan_time:.6f}s, {tarjan_mem:.2f} KB\n")

    print("--- Benchmark Complete ---")

    # --- Plotting Results ---
    nodes = [r['nodes'] for r in results]
    naive_times = [r['naive_time'] for r in results]
    tarjan_times = [r['tarjan_time'] for r in results]
    naive_mems = [r['naive_mem_kb'] for r in results]
    tarjan_mems = [r['tarjan_mem_kb'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(nodes, naive_times, label='Naive SCC Runtime (O(V*(V+E)))', color='red')
    plt.scatter(nodes, tarjan_times, label='Tarjan Runtime (O(V+E))', color='blue')
    plt.plot(nodes, naive_times, color='red', alpha=0.5, linestyle='--')
    plt.plot(nodes, tarjan_times, color='blue', alpha=0.5, linestyle='--')
    plt.title('Runtime vs. Graph Size on Multi-SCC Graphs')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('runtime_vs_size.png')
    print("Saved runtime plot to runtime_vs_size.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(nodes, naive_mems, label='Naive SCC Memory', color='red')
    plt.scatter(nodes, tarjan_mems, label='Tarjan Memory', color='blue')
    plt.plot(nodes, naive_mems, color='red', alpha=0.5, linestyle='--')
    plt.plot(nodes, tarjan_mems, color='blue', alpha=0.5, linestyle='--')
    plt.title('Peak Memory vs. Graph Size on Multi-SCC Graphs')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Peak Memory (KB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_vs_size.png')
    print("Saved memory plot to memory_vs_size.png")
    
    plt.show()
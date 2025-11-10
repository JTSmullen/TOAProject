import os
import tracemalloc
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.ticker as ticker
import psutil
import concurrent.futures
import time
from tqdm import tqdm  # progress bar for long-running processes

# Add the project root to the Python path to allow importing from the 'algorithms' package
sys.path.append('.')

# Import your SCC algorithm implementations
from algorithms.DepthFirstSearch import DepthFirstSearch  # Naive O(V*(V+E))
from algorithms.Tarjan import Tarjan  # Efficient Tarjan's algorithm

# --- Graph Utilities ---

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


def generate_scc_graph(num_nodes, num_sccs, internal_density):
    """Generate a graph with a specific number of SCCs and edge density within components."""
    graph = Graph()
    num_sccs = min(num_sccs, num_nodes)
    partitions = [[] for _ in range(num_sccs)]

    for i in range(num_nodes):
        partitions[i % num_sccs].append(i)

    for component in partitions:
        if not component:
            continue
        # Ring to ensure SCC
        for i in range(len(component)):
            u, v = component[i], component[(i + 1) % len(component)]
            graph.add_edge(u, v)
        # Extra internal edges
        max_edges = len(component) * (len(component) - 1)
        num_edges = int(max_edges * internal_density)
        for _ in range(num_edges):
            u, v = random.choice(component), random.choice(component)
            if u != v:
                graph.add_edge(u, v)

    # Connect components linearly (not cyclically to prevent collapsing all SCCs)
    for i in range(num_sccs - 1):
        if partitions[i] and partitions[i + 1]:
            u = random.choice(partitions[i])
            v = random.choice(partitions[i + 1])
            graph.add_edge(u, v)

    return graph

# --- Benchmarking ---

def run_and_measure(algorithm_class, graph):
    """Run the algorithm and measure execution time and memory usage."""
    alg_instance = algorithm_class(graph)
    tracemalloc.start()
    start = time.perf_counter()
    alg_instance.run()
    end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return end - start, peak_mem / 1024  # Return time and memory in KB

def benchmark_case(args):
    """Benchmark wrapper for multiprocessing."""
    i, n, internal_density = args
    num_sccs = max(1, int(n * i))
    graph = generate_scc_graph(n, num_sccs, internal_density)
    
    tarjan_time, tarjan_mem = run_and_measure(Tarjan, graph)
    naive_time, naive_mem = run_and_measure(DepthFirstSearch, graph)

    return {
        'nodes': n,
        'sccs': num_sccs,
        'scc_ratio': i,
        'naive_time': naive_time,
        'naive_mem_kb': naive_mem,
        'tarjan_time': tarjan_time,
        'tarjan_mem_kb': tarjan_mem
    }

def auto_workers():
    cores = os.cpu_count() or 2
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return max(1, min(cores - 1, int(ram_gb // 1)))

# --- Main ---

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # To allow saving plots in headless environments

    # Config
    NODE_COUNTS = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000]
    SCC_RATIOS = [0.05, 0.1, 0.25, 0.5, 1.0]
    INTERNAL_DENSITY = 0.2

    # Warm-up run
    print("Warming up...")
    warmup_graph = generate_scc_graph(10, 2, INTERNAL_DENSITY)
    run_and_measure(DepthFirstSearch, warmup_graph)
    run_and_measure(Tarjan, warmup_graph)
    print("Warm-up complete.\n")

    # Create task list
    tasks = [(i, n, INTERNAL_DENSITY) for i in SCC_RATIOS for n in NODE_COUNTS]

    # Run benchmarks in parallel
    print("Starting benchmarks...")
    results = []
    num_workers = auto_workers()
    print(f"Using {num_workers} worker(s)...\n")

    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(benchmark_case, tasks), total=len(tasks)):
            results.append(result)

    print(f"\nBenchmarking complete in {time.time() - start_time:.2f} seconds.")

    # --- Plotting Results ---

    # Group results by node count
    grouped = defaultdict(list)
    for r in results:
        grouped[r['nodes']].append(r)

    nodes_sorted = sorted(grouped.keys())

    # --- Plot average runtime ---
    plt.figure(figsize=(12, 8))
    
    # --- Naive Algorithm Plotting ---
    # Extract all data points for the naive algorithm
    all_nodes_naive = [r['nodes'] for r in results]
    all_times_naive = [r['naive_time'] for r in results]

    # Scatter plot for all individual data points
    plt.scatter(all_nodes_naive, all_times_naive, alpha=0.3, label='Naive DFS Data Points')

    # Calculate and plot the mean line
    mean_times_naive = [np.mean([r['naive_time'] for r in grouped[n]]) for n in nodes_sorted]
    plt.plot(nodes_sorted, mean_times_naive, linestyle='--', marker='o', label='Naive DFS Mean Time')

    # Calculate and plot the quadratic regression line
    coeffs_naive = np.polyfit(all_nodes_naive, all_times_naive, 2)
    poly_naive = np.poly1d(coeffs_naive)
    x_poly_naive = np.linspace(min(nodes_sorted), max(nodes_sorted), 100)
    y_poly_naive = poly_naive(x_poly_naive)
    plt.plot(x_poly_naive, y_poly_naive, color='red', label='Naive DFS Quadratic Fit')

    # --- Tarjan's Algorithm Plotting ---
    # Extract all data points for Tarjan's algorithm
    all_nodes_tarjan = [r['nodes'] for r in results]
    all_times_tarjan = [r['tarjan_time'] for r in results]

    # Scatter plot for all individual data points
    plt.scatter(all_nodes_tarjan, all_times_tarjan, alpha=0.3, label="Tarjan's Data Points")

    # Calculate and plot the mean line
    mean_times_tarjan = [np.mean([r['tarjan_time'] for r in grouped[n]]) for n in nodes_sorted]
    plt.plot(nodes_sorted, mean_times_tarjan, linestyle='--', marker='x', label="Tarjan's Mean Time")

    plt.title('Average Runtime vs Graph Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('runtime_vs_size.png', dpi=300)
    print("Plot saved as runtime_vs_size.png")
    plt.close()


    # --- Plot peak memory usage ---
    plt.figure(figsize=(12, 8))

    # --- Naive Algorithm Plotting ---
    # Extract all data points for the naive algorithm
    all_nodes_naive_mem = [r['nodes'] for r in results]
    all_mem_naive = [r['naive_mem_kb'] for r in results]

    # Scatter plot for all individual data points
    plt.scatter(all_nodes_naive_mem, all_mem_naive, alpha=0.3, label='Naive DFS Data Points')

    # Calculate and plot the mean line
    mean_mem_naive = [np.mean([r['naive_mem_kb'] for r in grouped[n]]) for n in nodes_sorted]
    plt.plot(nodes_sorted, mean_mem_naive, linestyle='--', marker='o', label='Naive DFS Mean Memory')
    
    # --- Tarjan's Algorithm Plotting ---
    # Extract all data points for Tarjan's algorithm
    all_nodes_tarjan_mem = [r['nodes'] for r in results]
    all_mem_tarjan = [r['tarjan_mem_kb'] for r in results]
    
    # Scatter plot for all individual data points
    plt.scatter(all_nodes_tarjan_mem, all_mem_tarjan, alpha=0.3, label="Tarjan's Data Points")

    # Calculate and plot the mean line
    mean_mem_tarjan = [np.mean([r['tarjan_mem_kb'] for r in grouped[n]]) for n in nodes_sorted]
    plt.plot(nodes_sorted, mean_mem_tarjan, linestyle='--', marker='x', label="Tarjan's Mean Memory")

    plt.title("Peak Memory Usage vs Graph Size")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Memory Usage (KB)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('peak_memory_vs_size.png', dpi=300)
    print("Plot saved as peak_memory_vs_size.png")
    plt.close()
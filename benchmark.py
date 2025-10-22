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
import itertools
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
    NODE_COUNTS = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 35000, 50000, 75000, 100000]
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

    # Plot average runtime
    plt.figure(figsize=(10, 6))
    for algo in ['naive_time', 'tarjan_time']:
        means = [np.mean([r[algo] for r in grouped[n]]) for n in nodes_sorted]
        plt.plot(nodes_sorted, means, label=algo.replace('_', ' ').title())

    plt.title('Average Runtime vs Graph Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('runtime_vs_size.png', dpi=300)
    print("Plot saved as runtime_vs_size.png")

    plt.figure(figsize=(10, 6))
    for algo in ['naive_mem_kb', 'tarjan_mem_kb']:
        means = [np.mean([r[algo] for r in grouped[n]]) for n in nodes_sorted]
        plt.plot(nodes_sorted, means, label=algo.replace('_', ' ').title())

    plt.title("Peak Memory Usage vs Graph Size")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Memory Usage (KB)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('peak_memory_vs_size.png', dpi=300)
    print("Plot saved as peak_memory_vs_size.png")


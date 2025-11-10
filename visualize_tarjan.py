import networkx as nx
import matplotlib.pyplot as plt
import random
from graph_util import Graph, generate_random_graph
import matplotlib.colors as mcolors
from algorithms.DepthFirstSearch import DepthFirstSearch, dfs_scc_generator
from algorithms.Tarjan import Tarjan
from itertools import zip_longest
import os
import tracemalloc
import sys
import numpy as np
from collections import defaultdict
import psutil
import concurrent.futures
import time
from tqdm import tqdm

def draw_algorithm_step(ax, G, pos, step, scc_colors, node_mapping, title):
    ax.clear()
    ax.set_title(title)
    
    description = step.get('description', '')
    ax.text(0.5, 1.05, description, transform=ax.transAxes, fontsize=10, ha='center')

    node_colors_list = ['white'] * len(G.nodes())
    
    if step.get('phase') == 4 or (step.get('lowlinks') is not None and step.get('description').startswith('Algorithm finished')): # Final step
        all_sccs = step.get('all_sccs', [])
        scc_text = "Found SCCs:\n"
        for scc_idx, scc in enumerate(all_sccs):
            color = scc_colors[scc_idx % len(scc_colors)]
            scc_text += f"- {scc}\n"
            for node in scc:
                node_colors_list[node_mapping[node]] = color
        ax.text(0.0, -0.1, scc_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else: # Intermediate steps
        for node in G.nodes():
            if node in step.get('scc', []):
                node_colors_list[node_mapping[node]] = 'lightgreen'
            elif node in step.get('stack', []):
                node_colors_list[node_mapping[node]] = 'lightblue'
            elif node == step.get('current_node'):
                node_colors_list[node_mapping[node]] = 'red'
            elif node in step.get('visited', []):
                node_colors_list[node_mapping[node]] = 'gray'

    nx.draw(G, pos, with_labels=True, node_color=node_colors_list, node_size=800, ax=ax, arrows=True, arrowstyle='->', arrowsize=20)
    
    if 'lowlinks' in step:
        labels = {node: f"low={step['lowlinks'].get(node)}\nid={step['indices'].get(node)}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, verticalalignment='bottom', ax=ax)


def visualize_scc_side_by_side(graph_data, tarjan_steps, naive_steps):
    G = nx.DiGraph()
    for u in graph_data.get_nodes():
        G.add_node(u)
        for v in graph_data.get_neighbors(u):
            G.add_edge(u, v)

    pos = nx.kamada_kawai_layout(G, scale=2)
    scc_colors = list(mcolors.TABLEAU_COLORS.values())
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    for tarjan_step, naive_step in zip_longest(tarjan_steps, naive_steps, fillvalue=tarjan_steps[-1] if len(tarjan_steps) > len(naive_steps) else naive_steps[-1]):
        if tarjan_step:
            draw_algorithm_step(ax1, G, pos, tarjan_step, scc_colors, node_mapping, "Tarjan's Algorithm")
        if naive_step:
            draw_algorithm_step(ax2, G, pos, naive_step, scc_colors, node_mapping, "DFS SCC Algorithm")
        
        plt.pause(0.5)

def tarjan_visualizer_generator(graph):
    index_counter = 0
    stack = []
    lowlinks = {}
    indices = {}
    on_stack = set()
    strongly_connected_components = []
    
    all_nodes = list(graph.get_nodes())

    for v_start in all_nodes:
        if v_start not in indices:
            traversal_stack = [(v_start, iter(graph.get_neighbors(v_start)))]

            while traversal_stack:
                node, neighbors_iterator = traversal_stack[-1]

                if node not in indices:
                    indices[node] = index_counter
                    lowlinks[node] = index_counter
                    index_counter += 1
                    stack.append(node)
                    on_stack.add(node)
                    yield {
                        'description': f"Visiting node {node}",
                        'current_node': node,
                        'stack': list(stack),
                        'lowlinks': dict(lowlinks),
                        'indices': dict(indices),
                        'scc': []
                    }

                processed_all_neighbors = False
                for neighbor in neighbors_iterator:
                    yield {
                        'description': f"Checking neighbor {neighbor} of {node}",
                        'current_node': node,
                        'stack': list(stack),
                        'lowlinks': dict(lowlinks),
                        'indices': dict(indices),
                        'scc': []
                    }
                    if neighbor not in indices:
                        traversal_stack.append((neighbor, iter(graph.get_neighbors(neighbor))))
                        break
                    elif neighbor in on_stack:
                        lowlinks[node] = min(lowlinks[node], indices[neighbor])
                        yield {
                            'description': f"Updating low-link of {node} based on neighbor {neighbor}",
                            'current_node': node,
                            'stack': list(stack),
                            'lowlinks': dict(lowlinks),
                            'indices': dict(indices),
                            'scc': []
                        }
                else:
                    processed_all_neighbors = True

                if processed_all_neighbors:
                    traversal_stack.pop()
                    if traversal_stack:
                        parent, _ = traversal_stack[-1]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[node])
                        yield {
                            'description': f"Updating low-link of parent {parent} based on child {node}",
                            'current_node': node,
                            'stack': list(stack),
                            'lowlinks': dict(lowlinks),
                            'indices': dict(indices),
                            'scc': []
                        }

                    if lowlinks[node] == indices[node]:
                        scc = []
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            scc.append(w)
                            if w == node:
                                break
                        strongly_connected_components.append(scc)
                        yield {
                            'description': f"Found SCC: {scc}",
                            'current_node': node,
                            'stack': list(stack),
                            'lowlinks': dict(lowlinks),
                            'indices': dict(indices),
                            'scc': scc
                        }
    
    yield {
        'description': "Algorithm finished. All SCCs highlighted.",
        'current_node': None,
        'stack': [],
        'lowlinks': lowlinks,
        'indices': indices,
        'scc': [],
        'all_sccs': strongly_connected_components
    }

def generate_scc_graph(num_nodes, num_sccs, internal_density):
    graph = Graph()
    num_sccs = min(num_sccs, num_nodes)
    partitions = [[] for _ in range(num_sccs)]

    for i in range(num_nodes):
        partitions[i % num_sccs].append(i)

    for component in partitions:
        if not component:
            continue
        for i in range(len(component)):
            u, v = component[i], component[(i + 1) % len(component)]
            graph.add_edge(u, v)
        max_edges = len(component) * (len(component) - 1)
        num_edges = int(max_edges * internal_density)
        for _ in range(num_edges):
            u, v = random.choice(component), random.choice(component)
            if u != v:
                graph.add_edge(u, v)

    for i in range(num_sccs - 1):
        if partitions[i] and partitions[i + 1]:
            u = random.choice(partitions[i])
            v = random.choice(partitions[i + 1])
            graph.add_edge(u, v)

    return graph

def run_and_measure(algorithm_class, graph):
    alg_instance = algorithm_class(graph)
    tracemalloc.start()
    start = time.perf_counter()
    alg_instance.run()
    end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return end - start, peak_mem / 1024

def benchmark_case(args):
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

def run_benchmarks():
    NODE_COUNTS = [50, 100, 200, 500, 1000, 2000]
    SCC_RATIOS = [0.05, 0.1, 0.25, 0.5, 1.0]
    INTERNAL_DENSITY = 0.2

    print("Warming up...")
    warmup_graph = generate_scc_graph(10, 2, INTERNAL_DENSITY)
    run_and_measure(DepthFirstSearch, warmup_graph)
    run_and_measure(Tarjan, warmup_graph)
    print("Warm-up complete.\n")

    tasks = [(i, n, INTERNAL_DENSITY) for i in SCC_RATIOS for n in NODE_COUNTS]

    print("Starting benchmarks...")
    results = []
    num_workers = auto_workers()
    print(f"Using {num_workers} worker(s)...")

    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(benchmark_case, tasks), total=len(tasks)):
            results.append(result)

    print(f"\nBenchmarking complete in {time.time() - start_time:.2f} seconds.")

    grouped = defaultdict(list)
    for r in results:
        grouped[r['nodes']].append(r)

    nodes_sorted = sorted(grouped.keys())

    plt.figure(figsize=(10, 6))
    for algo in ['naive_time', 'tarjan_time']:
        all_points = [(n, r[algo]) for n in nodes_sorted for r in grouped[n]]
        x_points, y_points = zip(*all_points)
        plt.scatter(x_points, y_points, alpha=0.5, label=f'{algo.replace("_", " ").title()} data')
        
        means = [np.mean([r[algo] for r in grouped[n]]) for n in nodes_sorted]
        plt.plot(nodes_sorted, means, linestyle='--', label=f'{algo.replace("_", " ").title()} mean')

        coeffs = np.polyfit(x_points, y_points, 2)
        poly = np.poly1d(coeffs)
        fit_y = poly(nodes_sorted)
        plt.plot(nodes_sorted, fit_y, linestyle=':', label=f'{algo.replace("_", " ").title()} quadratic fit')

    plt.title('Average Runtime vs Graph Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for algo in ['naive_mem_kb', 'tarjan_mem_kb']:
        all_points = [(n, r[algo]) for n in nodes_sorted for r in grouped[n]]
        x_points, y_points = zip(*all_points)
        plt.scatter(x_points, y_points, alpha=0.5, label=f'{algo.replace("_", " ").title()} data')

        means = [np.mean([r[algo] for r in grouped[n]]) for n in nodes_sorted]
        plt.plot(nodes_sorted, means, linestyle='--', label=f'{algo.replace("_", " ").title()} mean')

        # Add quadratic fit
        coeffs = np.polyfit(x_points, y_points, 2)
        poly = np.poly1d(coeffs)
        fit_y = poly(nodes_sorted)
        plt.plot(nodes_sorted, fit_y, linestyle=':', label=f'{algo.replace("_", " ").title()} quadratic fit')

    plt.title("Peak Memory Usage vs Graph Size")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Memory Usage (KB)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    choice = input("Enter 'v' to run the visualizer or 'b' to run the benchmark: ")
    if choice.lower() == 'v':
        G = generate_random_graph(10, 0.2)
        tarjan_steps = list(tarjan_visualizer_generator(G))
        naive_steps = list(dfs_scc_generator(G))
        
        plt.ion()
        visualize_scc_side_by_side(G, tarjan_steps, naive_steps)
        plt.ioff()
        plt.show()
    elif choice.lower() == 'b':
        run_benchmarks()
    else:
        print("Invalid choice.")

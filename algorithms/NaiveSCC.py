import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from graph_util import Graph

def naive_scc_generator(graph):
    """
    A generator that yields steps of the naive SCC algorithm (Kosaraju's).
    """
    
    # Step 1: First DFS pass to get finishing times
    stack = []
    visited = set()
    
    yield {
        'description': 'Step 1: First DFS to compute finishing times',
        'visited': set(visited),
        'stack': list(stack),
        'scc': [],
        'phase': 1
    }

    for node in graph.get_nodes():
        if node not in visited:
            dfs_stack = [node]
            while dfs_stack:
                u = dfs_stack[-1]
                if u not in visited:
                    visited.add(u)
                    yield {
                        'description': f'Step 1: Visiting {u}',
                        'visited': set(visited),
                        'stack': list(stack),
                        'scc': [],
                        'phase': 1,
                        'current_node': u
                    }
                
                unvisited_neighbors = [v for v in graph.get_neighbors(u) if v not in visited]
                if unvisited_neighbors:
                    dfs_stack.append(unvisited_neighbors[0])
                else:
                    dfs_stack.pop()
                    if u not in stack:
                        stack.append(u)
                        yield {
                            'description': f'Step 1: Finished {u}, adding to stack',
                            'visited': set(visited),
                            'stack': list(stack),
                            'scc': [],
                            'phase': 1,
                            'current_node': u
                        }

    # Step 2: Transpose the graph
    transposed_graph = Graph()
    for u in graph.get_nodes():
        transposed_graph.adj[u] = []
    for u in graph.get_nodes():
        for v in graph.get_neighbors(u):
            transposed_graph.add_edge(v, u)
            
    yield {
        'description': 'Step 2: Transposed the graph',
        'visited': set(),
        'stack': list(stack),
        'scc': [],
        'phase': 2
    }

    # Step 3: Second DFS pass on the transposed graph
    visited.clear()
    strongly_connected_components = []
    while stack:
        node = stack.pop()
        if node not in visited:
            scc = []
            component_stack = [node]
            while component_stack:
                u = component_stack.pop()
                if u not in visited:
                    visited.add(u)
                    scc.append(u)
                    yield {
                        'description': f'Step 3: Finding SCC, visiting {u}',
                        'visited': set(visited),
                        'stack': list(stack),
                        'scc': list(scc),
                        'phase': 3,
                        'current_node': u
                    }
                    for v in transposed_graph.get_neighbors(u):
                        if v not in visited:
                            component_stack.append(v)
            strongly_connected_components.append(scc)
            yield {
                'description': f'Step 3: Found SCC: {scc}',
                'visited': set(visited),
                'stack': list(stack),
                'scc': list(scc),
                'phase': 3
            }
            
    yield {
        'description': 'Algorithm finished. All SCCs highlighted.',
        'visited': set(visited),
        'stack': [],
        'scc': [],
        'all_sccs': strongly_connected_components,
        'phase': 4
    }

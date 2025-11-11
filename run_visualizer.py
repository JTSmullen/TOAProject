from algorithms.Tarjan import Tarjan
from graph_util import generate_random_graph

if __name__ == "__main__":
    # Create a random graph with 10 vertices and a 20% chance of an edge
    G = generate_random_graph(10, 0.2)
    
    # Instantiate Tarjan with the graph and the visualization flag
    tarjan = Tarjan(G, is_visualize=True)
    
    # Run the algorithm
    tarjan.run()

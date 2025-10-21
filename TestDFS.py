import unittest
import time
from unittest.mock import Mock

# Assuming the DepthFirstSearch class is in a file named DFSTraversal.py
from algorithms.DepthFirstSearch import DepthFirstSearch

class SimpleGraph:
    """
    A simple graph implementation for testing purposes.
    """
    def __init__(self, edges):
        self.adj = {}
        for u, v in edges:
            if u not in self.adj:
                self.adj[u] = []
            if v not in self.adj:
                self.adj[v] = []
            self.adj[u].append(v)

    def get_neighbors(self, node):
        return self.adj.get(node, [])

    def get_nodes(self):
        return list(self.adj.keys())

class DFSTest(unittest.TestCase):
    """
    Unit tests for the DepthFirstSearch class.
    """

    def test_empty_graph(self):
        """
        Test with an empty graph.
        """
        graph = SimpleGraph([])
        dfs = DepthFirstSearch(graph)
        traversal_order, elapsed = dfs.run()
        self.assertEqual(traversal_order, [])

    def test_single_node_graph(self):
        """
        Test with a graph containing a single node.
        """
        graph = SimpleGraph([('A', 'A')])
        dfs = DepthFirstSearch(graph)
        traversal_order, elapsed = dfs.run()
        self.assertEqual(traversal_order, ['A'])

    def test_disconnected_graph(self):
        """
        Test with a disconnected graph.
        """
        edges = [('A', 'B'), ('B', 'C'), ('D', 'E')]
        graph = SimpleGraph(edges)
        dfs = DepthFirstSearch(graph)
        traversal_order, elapsed = dfs.run()
        self.assertIn('A', traversal_order)
        self.assertIn('B', traversal_order)
        self.assertIn('C', traversal_order)
        self.assertIn('D', traversal_order)
        self.assertIn('E', traversal_order)
        self.assertEqual(len(set(traversal_order)), 5)

    def test_directed_graph(self):
        """
        Test with a directed graph.
        """
        edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]
        graph = SimpleGraph(edges)
        dfs = DepthFirstSearch(graph, is_directed=True)
        traversal_order, elapsed = dfs.run()
        # The exact traversal order can vary, so we check for correctness
        # of the traversal from the starting node 'A'.
        self.assertTrue(
            traversal_order == ['A', 'B', 'D', 'C', 'E'] or
            traversal_order == ['A', 'C', 'E', 'B', 'D']
        )

    def test_undirected_graph(self):
        """
        Test with an undirected graph.
        """
        edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
        graph = SimpleGraph(edges)
        # For an undirected graph, we add reverse edges
        for u, v in edges:
            if v not in graph.adj:
                graph.adj[v] = []
            graph.adj[v].append(u)

        dfs = DepthFirstSearch(graph, is_directed=False)
        traversal_order, elapsed = dfs.run()
        self.assertEqual(len(set(traversal_order)), 3)
        self.assertIn('A', traversal_order)
        self.assertIn('B', traversal_order)
        self.assertIn('C', traversal_order)

    def test_with_mocked_graph(self):
        """
        Test using a mock graph object.
        """
        mock_graph = Mock()
        mock_graph.get_nodes.return_value = ['A', 'B', 'C']
        mock_graph.get_neighbors.side_effect = lambda node: {
            'A': ['B', 'C'],
            'B': ['C'],
            'C': []
        }[node]

        dfs = DepthFirstSearch(mock_graph)
        traversal_order, elapsed = dfs.run()

        self.assertEqual(traversal_order, ['A', 'B', 'C'])
        mock_graph.get_nodes.assert_called_once()
        self.assertGreaterEqual(mock_graph.get_neighbors.call_count, 2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
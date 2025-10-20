import unittest

from Tarjan import Tarjan


class SimpleGraph:
	"""A minimal graph implementation that provides get_nodes and get_neighbors
	which the Tarjan implementation expects.
	"""
	def __init__(self, adjacency=None):
		# adjacency: dict[node, list[neighbor]]
		self.adj = adjacency or {}

	def get_nodes(self):
		# ensure we include isolated nodes
		return list(self.adj.keys())

	def get_neighbors(self, node):
		return list(self.adj.get(node, []))


class TestTarjanAlgorithm(unittest.TestCase):

	def assertSCCsEqualAsSets(self, actual, expected):
		# compare list of components ignoring order and ordering within components
		actual_sets = {tuple(sorted(comp)) for comp in actual}
		expected_sets = {tuple(sorted(comp)) for comp in expected}
		self.assertEqual(actual_sets, expected_sets)

	def test_empty_graph(self):
		g = SimpleGraph({})
		t = Tarjan(g)
		scc, _ = t.run()
		self.assertEqual(scc, [])

	def test_single_node(self):
		g = SimpleGraph({"A": []})
		t = Tarjan(g)
		scc, _ = t.run()
		self.assertSCCsEqualAsSets(scc, [["A"]])

	def test_two_node_cycle(self):
		g = SimpleGraph({"A": ["B"], "B": ["A"]})
		t = Tarjan(g)
		scc, _ = t.run()
		self.assertSCCsEqualAsSets(scc, [["A", "B"]])

	def test_multiple_sccs(self):
		# Graph with three SCCs: {A,B}, {C}, {D,E,F}
		g = SimpleGraph({
			"A": ["B"],
			"B": ["A", "C"],
			"C": [],
			"D": ["E"],
			"E": ["F"],
			"F": ["D"],
		})
		t = Tarjan(g)
		scc, _ = t.run()
		expected = [["A", "B"], ["C"], ["D", "E", "F"]]
		self.assertSCCsEqualAsSets(scc, expected)

	def test_linear_no_cycle(self):
		g = SimpleGraph({"1": ["2"], "2": ["3"], "3": []})
		t = Tarjan(g)
		scc, _ = t.run()
		expected = [["1"], ["2"], ["3"]]
		self.assertSCCsEqualAsSets(scc, expected)


if __name__ == "__main__":
	unittest.main()


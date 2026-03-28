# W2/py/tests/test_graph.py
import pytest
import networkx as nx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph import build_territory_graph, round_node

def _simple_assigned():
    """Two segments: A(10m) → B(8m) → C(6m), all territory=1"""
    return [
        {'pts': [(0,0,10.0),(5,0,8.0)], 'geom': None, 'territory': 1},
        {'pts': [(5,0,8.0),(10,0,6.0)], 'geom': None, 'territory': 1},
    ]

def test_edge_direction_high_to_low():
    segs = _simple_assigned()
    G = build_territory_graph(segs, territory_id=1)
    A = round_node(0, 0)
    B = round_node(5, 0)
    C = round_node(10, 0)
    assert G.has_edge(A, B)   # 10m → 8m (downhill)
    assert G.has_edge(B, C)   # 8m → 6m (downhill)
    assert not G.has_edge(B, A)
    assert not G.has_edge(C, B)

def test_sink_is_lowest_node():
    segs = _simple_assigned()
    G = build_territory_graph(segs, territory_id=1)
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    C = round_node(10, 0)
    assert sinks == [C]

def test_returns_empty_graph_for_wrong_territory():
    segs = _simple_assigned()
    G = build_territory_graph(segs, territory_id=99)
    assert G.number_of_nodes() == 0

def test_gap_heal_connects_nearby_nodes():
    """Two nearby but disconnected nodes within connect_tol are healed."""
    # Segment A→B, and C disconnected from B by 0.3m (within default tol=0.5)
    # C should get connected to B via gap-heal
    segs = [
        {'pts': [(0,0,10.0),(5,0,8.0)], 'geom': None, 'territory': 1},
        # C is at (5.3, 0) — 0.3m from B at (5, 0)
        {'pts': [(5.3,0,8.0),(10,0,6.0)], 'geom': None, 'territory': 1},
    ]
    G = build_territory_graph(segs, territory_id=1, connect_tol=0.5)
    # The graph should be connected (B and C should be healed together)
    assert nx.is_weakly_connected(G)

def test_gap_heal_does_not_create_cycle():
    """Gap-heal must not create directed cycles — topological_sort must succeed."""
    # Triangle arrangement: A→B, B→C, C near A (gap-heal would close cycle A→B→C→A)
    segs = [
        {'pts': [(0,0,10.0),(5,0,8.0)], 'geom': None, 'territory': 1},  # A→B
        {'pts': [(5,0,8.0),(10,0,6.0)], 'geom': None, 'territory': 1},  # B→C
        # D at (0.3, 0) close to A — gap-heal would try D→A or A→D
        {'pts': [(10,0,6.0),(0.3,0,5.5)], 'geom': None, 'territory': 1},  # C→D
    ]
    G = build_territory_graph(segs, territory_id=1, connect_tol=0.5)
    # Must be a DAG — topological_sort must not raise
    try:
        list(nx.topological_sort(G))
        is_dag = True
    except nx.NetworkXUnfeasible:
        is_dag = False
    assert is_dag, "Gap-heal introduced a directed cycle"

def test_seg_pts_reversed_for_uphill_to_downhill_edge():
    """When segment goes uphill (e > s elevation), edge is reversed and seg_pts too."""
    # Segment defined start-to-end as uphill: (0,0,5) → (5,0,10)
    segs = [
        {'pts': [(0,0,5.0),(5,0,10.0)], 'geom': None, 'territory': 1},
    ]
    G = build_territory_graph(segs, territory_id=1)
    high = round_node(5, 0)
    low  = round_node(0, 0)
    # Edge should go high→low: (5,0)→(0,0)
    assert G.has_edge(high, low)
    assert not G.has_edge(low, high)
    # seg_pts should be reversed so first point is the high end
    edge_pts = G.edges[high, low]['seg_pts']
    assert edge_pts[0][2] >= edge_pts[-1][2], "seg_pts[0] should be the high end"


def test_bfs_orients_edge_through_local_dip():
    """
    Road: A(10m) - B(5m) - C(8m). Outfall at C.
    High->low would make B a dead-end sink (C is higher than B).
    BFS from C should produce A->B->C (both edges flow toward C).
    """
    segs = [
        {'pts': [(0,0,10.0),(5,0,5.0)], 'geom': None, 'territory': 1},
        {'pts': [(5,0,5.0),(10,0,8.0)], 'geom': None, 'territory': 1},
    ]
    A = round_node(0, 0); B = round_node(5, 0); C = round_node(10, 0)
    G = build_territory_graph(segs, territory_id=1, outfall_xy=(10.0, 0.0))
    assert G.has_edge(A, B)
    assert G.has_edge(B, C)          # uphill road, but pipe flows toward outfall
    assert not G.has_edge(C, B)
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    assert sinks == [C]              # C is the unique sink


def test_bfs_removes_road_disconnected_nodes():
    """Nodes with no road path to the outfall are removed from G."""
    segs = [
        {'pts': [(0,0,10.0),(5,0,8.0)],   'geom': None, 'territory': 1},
        {'pts': [(100,0,7.0),(105,0,6.0)], 'geom': None, 'territory': 1},
    ]
    G = build_territory_graph(segs, territory_id=1, outfall_xy=(5.0, 0.0))
    A = round_node(0,0); B = round_node(5,0)
    D = round_node(100,0); E = round_node(105,0)
    assert A in G.nodes() and B in G.nodes()
    assert D not in G.nodes() and E not in G.nodes()

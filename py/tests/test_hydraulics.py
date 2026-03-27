# W2/py/tests/test_hydraulics.py
import pytest
import networkx as nx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hydraulics import (route_topdown, prune_to_feasibility,
                         excess_cover, find_bottleneck, find_guilty_branch,
                         collect_nodes_to_prune)
from hydraulics import check_gravity_connectivity
from hydraulics import pool_reassignment_loop

def _make_graph(nodes, edges):
    """nodes: {key: ground_elev}, edges: [(u,v,length)]"""
    G = nx.DiGraph()
    for k, e in nodes.items():
        G.add_node(k, ground_elev=e, x=0.0, y=0.0)
    for u, v, L in edges:
        G.add_edge(u, v, length=L, seg_pts=None)
    return G

def test_simple_chain_inverts():
    """A(10) -> B(8) -> C(6), length=100m each, min_slope=0.001"""
    G = _make_graph(
        {'A': 10.0, 'B': 8.0, 'C': 6.0},
        [('A','B',100.0), ('B','C',100.0)]
    )
    inverts, status = route_topdown(G, outfall_node='C',
                                    I_outfall=5.9,
                                    min_slope=0.001, min_cover=0.0)
    # A is source: I_A = 10.0
    # slope A→B: (10-8)/100 = 0.02 > 0.001 → s_pipe=0.02, I_B = 10.0 - 0.02*100 = 8.0
    # slope B→C: (8-6)/100 = 0.02 → I_C = 8.0 - 2.0 = 6.0
    # I_C=6.0 >= I_outfall=5.9 → PASS
    assert inverts['A'] == pytest.approx(10.0, abs=0.01)
    assert inverts['B'] == pytest.approx(8.0,  abs=0.01)
    assert inverts['C'] == pytest.approx(6.0,  abs=0.01)
    assert status == 'PASS'

def test_junction_rule_deepest_wins():
    """
    Two branches A and B both feed C.
    A(20) → C, B(10) → C, length=100m each.
    I_A_start=20.0, I_B_start=10.0
    slope A→C: (20-5)/100=0.15, I_from_A = 20 - 0.15*100 = 5.0
    slope B→C: (10-5)/100=0.05, I_from_B = 10 - 0.05*100 = 5.0
    min(5.0, 5.0) = 5.0
    """
    G = _make_graph(
        {'A': 20.0, 'B': 10.0, 'C': 5.0},
        [('A','C',100.0), ('B','C',100.0)]
    )
    inverts, status = route_topdown(G, outfall_node='C',
                                    I_outfall=0.0,
                                    min_slope=0.001, min_cover=0.0)
    I_from_A = 20.0 - max((20.0-5.0)/100.0, 0.001)*100.0
    I_from_B = 10.0 - max((10.0-5.0)/100.0, 0.001)*100.0
    assert inverts['C'] == pytest.approx(min(I_from_A, I_from_B), abs=0.01)

def test_feasibility_fail_when_too_deep():
    """Network where invert at outfall < I_outfall → FAIL."""
    G = _make_graph(
        {'A': 5.0, 'B': 4.9},
        [('A','B', 10000.0)]   # 10 km flat — min_slope forces 5m drop
    )
    # I_A=5.0, slope=min_slope=0.0005, drop=5.0m → I_B=0.0 < I_outfall=4.5
    inverts, status = route_topdown(G, outfall_node='B',
                                    I_outfall=4.5,
                                    min_slope=0.0005, min_cover=0.0)
    assert status == 'FAIL'


def test_prune_single_bad_branch_passes():
    """
    DEEP_SRC → A → OF: too deep (long flat branch)
    After pruning DEEP_SRC subtree, remaining graph A→OF is checked.
    If A→OF alone is feasible, result is PASS.
    """
    # DEEP_SRC→A is 5000m flat → pipe drops 2.5m → arrives way below outfall
    # A→OF is 10m, ground drops 1m → feasible alone
    G = _make_graph(
        {'DEEP': 10.0, 'A': 10.0, 'OF': 9.0},
        [('DEEP', 'A', 5000.0), ('A', 'OF', 10.0)]
    )
    inverts, pruned, status = prune_to_feasibility(
        G, outfall_node='OF', I_outfall=6.6,
        min_slope=0.0005, min_cover=0.0)
    assert status == 'PASS'
    assert 'DEEP' in pruned


def test_prune_preserves_feasible_sibling_branch():
    """
    DEEP_SRC → A → OF  (deep, should be pruned)
    SHALLOW_SRC → A → OF  (feasible, must be preserved)
    After pruning DEEP_SRC, A and SHALLOW_SRC must remain.
    """
    G = _make_graph(
        {'DEEP': 5.0, 'SHALLOW': 15.0, 'A': 4.9, 'OF': 4.8},
        [('DEEP', 'A', 5000.0),    # long flat — drops 2.5m below OF
         ('SHALLOW', 'A', 10.0),   # short steep — arrives high
         ('A', 'OF', 5.0)]
    )
    inverts, pruned, status = prune_to_feasibility(
        G, outfall_node='OF', I_outfall=4.5,
        min_slope=0.0005, min_cover=0.0)
    # DEEP must be pruned; SHALLOW and A must survive
    assert 'DEEP' in pruned
    assert 'SHALLOW' not in pruned
    assert 'A' not in pruned


def test_outfall_node_never_pruned():
    """outfall_node must never appear in pruned_nodes."""
    G = _make_graph(
        {'A': 5.0, 'OF': 4.9},
        [('A', 'OF', 10000.0)]  # absurdly long — will definitely fail
    )
    _, pruned, _ = prune_to_feasibility(
        G, outfall_node='OF', I_outfall=4.8,
        min_slope=0.0005, min_cover=0.0)
    assert 'OF' not in pruned


def test_excess_cover_arithmetic():
    """excess_cover = (ground - invert) - min_cover"""
    G = _make_graph({'A': 10.0}, [])
    inverts = {'A': 7.0}
    ec = excess_cover(G, inverts, min_cover=0.5)
    assert ec['A'] == pytest.approx(2.5, abs=1e-9)  # (10-7)-0.5


def test_requires_review_when_unfixable():
    """Single-segment network that is always infeasible → REQUIRES_REVIEW."""
    # Outfall invert = 100m (impossibly high) — no graph can satisfy this
    G = _make_graph(
        {'A': 5.0, 'OF': 4.9},
        [('A', 'OF', 10.0)]
    )
    _, _, status = prune_to_feasibility(
        G, outfall_node='OF', I_outfall=100.0,
        min_slope=0.0005, min_cover=0.0)
    assert status == 'REQUIRES_REVIEW'


def test_gravity_connectivity_excludes_local_sink_nodes():
    """
    Graph:  SOURCE_A → OUTFALL
            SOURCE_B → LOCAL_SINK   (no path to outfall)
    Only SOURCE_A and OUTFALL should be reachable.
    """
    G = _make_graph(
        {'SOURCE_A': 10.0, 'SOURCE_B': 9.0,
         'LOCAL_SINK': 5.0, 'OUTFALL': 4.0},
        [('SOURCE_A', 'OUTFALL', 50.0),
         ('SOURCE_B', 'LOCAL_SINK', 50.0)]
    )
    reachable = check_gravity_connectivity(G, outfall_node='OUTFALL')
    assert 'SOURCE_A' in reachable
    assert 'OUTFALL'  in reachable
    assert 'SOURCE_B'    not in reachable
    assert 'LOCAL_SINK'  not in reachable

def test_gravity_connectivity_all_connected():
    """All nodes drain to outfall → all reachable."""
    G = _make_graph(
        {'A': 10.0, 'B': 8.0, 'C': 6.0},
        [('A','B',50.0), ('B','C',50.0)]
    )
    reachable = check_gravity_connectivity(G, outfall_node='C')
    assert reachable == {'A', 'B', 'C'}


def test_pool_channel_accepted_by_second_territory():
    """
    Channel P (territory=None, blacklisted from T1): eligible for T2.
    G2 has a node at coordinate (30,0). Pool seg connects (40,0)→(30,0).
    After routing, G2 PASS → pool seg assigned to T2.
    """
    from graph import round_node, build_territory_graph

    # G1: uses string node keys (string-key graphs are valid for blacklist test)
    # Very long flat segment → routing would FAIL if pool seg were added
    G1 = _make_graph(
        {'A': 5.001, 'OF1': 5.0},
        [('A', 'OF1', 10000.0)]
    )

    # G2: uses coordinate node keys to match round_node behavior
    N_B   = round_node(30.0, 0.0)   # (30.0, 0.0)
    N_OF2 = round_node(35.0, 0.0)   # (35.0, 0.0)
    G2 = nx.DiGraph()
    G2.add_node(N_B,   x=30.0, y=0.0, ground_elev=30.0)
    G2.add_node(N_OF2, x=35.0, y=0.0, ground_elev=20.0)
    G2.add_edge(N_B, N_OF2, length=100.0, seg_pts=None)

    # Pool seg: (40,0)→(30,0), connects to G2 at N_B
    N_C = round_node(40.0, 0.0)   # (40.0, 0.0)
    pool_seg = {
        'pts':      [(N_C[0], N_C[1], 35.0),
                     (N_B[0], N_B[1], 30.0)],
        'geom':      None,
        'territory': None,
        'blacklist': {1},   # already rejected by T1
    }

    # T2 existing assigned seg
    t2_seg = {
        'pts':      [(N_B[0], N_B[1], 30.0),
                     (N_OF2[0], N_OF2[1], 20.0)],
        'geom':      None,
        'territory': 2,
        'blacklist': set(),
    }

    graphs     = {1: G1, 2: G2}
    of_inverts = {1: 4.5, 2: 19.0}
    assigned   = [
        {'pts': [(0, 0, 5.001), (5, 0, 5.0)], 'geom': None, 'territory': 1, 'blacklist': set()},
        t2_seg,
        pool_seg,
    ]

    pool_reassignment_loop(assigned, graphs, of_inverts,
                           min_slope=0.0005, min_cover=0.0)

    # Pool channel should be accepted by T2
    assert pool_seg['territory'] == 2
    assert 1 in pool_seg['blacklist']  # T1 pre-blacklisted before call — must be preserved

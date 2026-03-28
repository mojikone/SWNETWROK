"""hydraulics.py — Top-down invert routing + cover-violation pruning.

Physical model
--------------
Pipe inverts are computed top-down (sources → outfall):

    Source:  I = ground_source − min_cover
    Edge U→D: s_pipe = max((ground_U − ground_D) / L, min_slope)
              I_D_candidate = I_U − s_pipe × L
    Junction: I_D = min(all candidates)  [deepest arriving pipe sets the level]

For a monotone path this gives:
    I_arrived = I_outfall  (exactly)

Feasibility check:
    I_arrived ≥ I_outfall

I_outfall is a physical constraint (outfall structure invert).  Monotone paths
pass exactly.  Non-monotone BFS paths (uphill sections or sources below outfall
elevation) arrive deeper than I_outfall and are pruned.

The outfall node's DISPLAY invert is set to ground_outfall (the physical
discharge elevation).  The approach pipe arrives min_cover below that — normal
for an open-end headwall structure.
"""
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from graph import round_node, build_territory_graph


def route_topdown(G, outfall_node, I_outfall, min_slope=0.0005, min_cover=0.0):
    """
    Route pipe inverts top-down (sources → outfall).

    Rules
    -----
    - Source nodes (in_degree=0): I = ground − min_cover
    - Edge U→D: target = I_outfall if D is outfall, else ground_D − min_cover
                s_recovery = (I_U − target) / L
                s_pipe = max(s_recovery, min_slope)   [grade adjustment]
                I_D_candidate = I_U − s_pipe × L  →  arrives at target
                                                      when s_recovery ≥ min_slope
    - Junction:  I_D = min(candidates)   [deepest wins]
    - Feasibility: I_at_outfall ≥ I_outfall

    Returns
    -------
    inverts : dict { node: float }
    status  : 'PASS' | 'FAIL'
    """
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return {}, 'FAIL'

    inverts = {}

    for node in topo:
        g     = G.nodes[node]['ground_elev']
        preds = list(G.predecessors(node))

        if not preds:
            inverts[node] = g - min_cover
        else:
            # Target invert for this node:
            #   outfall node → I_outfall (physical fixed constraint)
            #   all other nodes → g - min_cover (minimum cover requirement)
            target = I_outfall if node == outfall_node else (g - min_cover)

            candidates = []
            for pred in preds:
                if pred not in inverts:
                    continue
                I_pred = inverts[pred]
                L      = G.edges[pred, node]['length']
                # Recovery slope: exactly enough descent to arrive at target.
                # Clamped to min_slope from below.
                s_recovery = (I_pred - target) / L if L > 0 else min_slope
                s_pipe     = max(s_recovery, min_slope)
                candidates.append(I_pred - s_pipe * L)

            if not candidates:
                inverts[node] = target
            else:
                inverts[node] = min(candidates)

    if outfall_node not in inverts:
        return inverts, 'FAIL'

    # Feasibility: pipe must arrive within min_cover of outfall ground.
    # For a monotone path this is satisfied exactly; failures are from
    # non-monotone BFS paths or sources below the outfall elevation.
    status = 'PASS' if inverts[outfall_node] >= I_outfall else 'FAIL'
    return inverts, status


# ── Pruning helpers ────────────────────────────────────────────────────────────

def find_bottleneck(G, inverts, min_cover=0.0):
    """
    Return the node with the highest excess cover (deepest pipe relative to
    required minimum).  This node's upstream branch is the culprit forcing the
    outfall invert too deep.
    """
    best, best_ec = None, -float('inf')
    for node, inv in inverts.items():
        ec = (G.nodes[node]['ground_elev'] - inv) - min_cover
        if ec > best_ec:
            best_ec = ec
            best    = node
    return best


def find_guilty_branch(G, bottleneck, inverts):
    """
    Trace back upstream from bottleneck following the predecessor with the
    lowest (deepest) invert until a source is reached.

    Returns [source, ..., bottleneck] (high-to-low order).
    """
    if not list(G.predecessors(bottleneck)):
        return [bottleneck]

    branch, current = [bottleneck], bottleneck
    while True:
        preds = list(G.predecessors(current))
        if not preds:
            break
        nxt = min(preds, key=lambda p: inverts.get(p, float('inf')))
        branch.append(nxt)
        current = nxt

    branch.reverse()
    return branch


def collect_nodes_to_prune(G, guilty_src, outfall_node):
    """
    Remove guilty_src then collect any nodes that have lost their only path to
    the outfall.  Returns set including guilty_src itself.
    """
    to_remove = {guilty_src}
    G_temp    = G.copy()
    G_temp.remove_node(guilty_src)

    if outfall_node not in G_temp:
        return to_remove

    can_reach = nx.ancestors(G_temp, outfall_node) | {outfall_node}
    for node in G.nodes():
        if node != guilty_src and node not in can_reach:
            to_remove.add(node)
    return to_remove


def collect_branch_nodes(G, source_node):
    return set(nx.descendants(G, source_node)) | {source_node}


def prune_by_max_cover(G, outfall_node, inverts, max_cover):
    """
    Remove nodes where pipe cover exceeds max_cover, then orphan any node
    that loses its directed path to outfall_node.

    Returns
    -------
    working_G    : pruned copy of G
    newly_pruned : set of removed nodes
    """
    working_G    = G.copy()
    newly_pruned = set()

    deep_nodes = {
        node for node, inv in inverts.items()
        if (working_G.nodes[node]['ground_elev'] - inv) > max_cover
        and node != outfall_node
    }

    if not deep_nodes:
        return working_G, newly_pruned

    working_G.remove_nodes_from(deep_nodes)
    newly_pruned.update(deep_nodes)

    # Remove nodes that lost their only directed path to the outfall
    if outfall_node in working_G:
        can_reach = nx.ancestors(working_G, outfall_node) | {outfall_node}
        disconnected = {n for n in working_G.nodes() if n not in can_reach}
        working_G.remove_nodes_from(disconnected)
        newly_pruned.update(disconnected)

    return working_G, newly_pruned


def prune_to_feasibility(G, outfall_node, I_outfall,
                          min_slope=0.0005, min_cover=0.0, max_cover=None,
                          max_iterations=2000):
    """
    Iteratively prune until the routed outfall invert satisfies
    I_arrived >= I_outfall, or the graph is exhausted.

    Phase 1 (if max_cover set): remove nodes exceeding max_cover depth,
    rescue upstream nodes that retain alternative path to outfall.

    Phase 2 (fallback): bottleneck pruning for remaining outfall failures.

    Returns
    -------
    final_inverts : dict { node: float }
    pruned_nodes  : set
    status        : 'PASS' | 'REQUIRES_REVIEW'
    """
    working_G    = G.copy()
    pruned_nodes = set()

    # ── Phase 1: MAX_COVER pruning ────────────────────────────────────────────
    if max_cover is not None:
        for iteration in range(max_iterations):
            inverts, status = route_topdown(
                working_G, outfall_node, I_outfall, min_slope, min_cover)

            working_G, newly_pruned = prune_by_max_cover(
                working_G, outfall_node, inverts, max_cover)

            if not newly_pruned:
                break   # no more deep nodes — phase 1 complete

            pruned_nodes.update(newly_pruned)
            print(f"    max_cover prune: removed {len(newly_pruned)} nodes "
                  f"(>{max_cover}m cover)")

        # Re-route after phase 1
        inverts, status = route_topdown(
            working_G, outfall_node, I_outfall, min_slope, min_cover)

        if status == 'PASS':
            return inverts, pruned_nodes, 'PASS'

    # ── Phase 2: bottleneck pruning (fallback) ────────────────────────────────
    for iteration in range(max_iterations):
        inverts, status = route_topdown(
            working_G, outfall_node, I_outfall, min_slope, min_cover)

        if status == 'PASS':
            return inverts, pruned_nodes, 'PASS'

        if working_G.number_of_nodes() <= 1:
            return inverts, pruned_nodes, 'REQUIRES_REVIEW'

        bneck      = find_bottleneck(working_G, inverts, min_cover)
        branch     = find_guilty_branch(working_G, bneck, inverts)
        guilty_src = branch[0]
        to_remove  = collect_nodes_to_prune(working_G, guilty_src, outfall_node)
        to_remove.discard(outfall_node)

        if not to_remove:
            return inverts, pruned_nodes, 'REQUIRES_REVIEW'

        pruned_nodes.update(to_remove)
        working_G.remove_nodes_from(to_remove)
        print(f"    bottleneck prune iter {iteration+1}: removed {len(to_remove)} nodes "
              f"(guilty source: {guilty_src})")

    return inverts, pruned_nodes, 'REQUIRES_REVIEW'


# ── Gravity connectivity ───────────────────────────────────────────────────────

def check_gravity_connectivity(G, outfall_node):
    if outfall_node not in G:
        return set()
    return nx.ancestors(G, outfall_node) | {outfall_node}


# ── Pool re-assignment loop ────────────────────────────────────────────────────

def pool_reassignment_loop(assigned, graphs, of_inverts,
                            min_slope=0.0005, min_cover=0.0, max_cover=None,
                            max_rounds=10, of_xy=None,
                            outfall_snap_r=None):
    """
    Iteratively re-assign pool channels to territories.
    Acceptance: route_topdown passes AND no node exceeds max_cover.
    """
    def _endpoints(seg):
        pts = seg['pts']
        return (round_node(pts[0][0], pts[0][1]),
                round_node(pts[-1][0], pts[-1][1]))

    def _pool():
        return [s for s in assigned if s['territory'] is None]

    for _rnd in range(max_rounds):
        pool = _pool()
        if not pool:
            break
        accepted_any = False

        for seg in pool:
            if len(seg['pts']) < 2:
                continue
            ep_s, ep_e = _endpoints(seg)

            for of_id, G in graphs.items():
                if of_id in seg.get('blacklist', set()):
                    continue

                g_nodes = set(G.nodes())
                if ep_s not in g_nodes and ep_e not in g_nodes:
                    continue

                if of_xy and of_id in of_xy:
                    ox_s, oy_s = of_xy[of_id]
                    g_nl  = list(G.nodes())
                    g_nxy = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in g_nl])
                    _, _i = KDTree(g_nxy).query([ox_s, oy_s])
                    snap  = g_nl[_i]
                else:
                    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
                    if not sinks:
                        continue
                    snap = min(sinks, key=lambda n: G.nodes[n]['ground_elev'])

                seg['territory'] = of_id

                tmp_assigned = [s for s in assigned if s['territory'] == of_id]
                tmp_G = build_territory_graph(
                    tmp_assigned, of_id,
                    outfall_xy=of_xy.get(of_id) if of_xy else None,
                    outfall_snap_r=outfall_snap_r
                )

                if of_xy and of_id in of_xy:
                    ox_s, oy_s = of_xy[of_id]
                    tmp_nl = list(tmp_G.nodes())
                    if not tmp_nl:
                        seg['territory'] = None
                        continue
                    tmp_nxy = np.array([[tmp_G.nodes[n]['x'], tmp_G.nodes[n]['y']] for n in tmp_nl])
                    _, _ti  = KDTree(tmp_nxy).query([ox_s, oy_s])
                    tmp_snap = tmp_nl[_ti]
                else:
                    tmp_sinks = [n for n in tmp_G.nodes() if tmp_G.out_degree(n) == 0]
                    if not tmp_sinks:
                        seg['territory'] = None
                        continue
                    tmp_snap = min(tmp_sinks,
                                   key=lambda n: tmp_G.nodes[n]['ground_elev'])

                inv_tmp, status = route_topdown(
                    tmp_G, tmp_snap, of_inverts[of_id], min_slope, min_cover)

                if status == 'PASS' and max_cover is not None:
                    deep = any(
                        (tmp_G.nodes[n]['ground_elev'] - inv_tmp.get(n, float('-inf'))) > max_cover
                        for n in tmp_G.nodes()
                    )
                    if deep:
                        status = 'FAIL'

                if status == 'PASS':
                    graphs[of_id] = tmp_G
                    accepted_any  = True
                    break
                else:
                    seg['territory'] = None
                    seg.setdefault('blacklist', set()).add(of_id)

        if not accepted_any:
            break

    remaining = sum(1 for s in assigned if s['territory'] is None)
    print(f"  Pool remaining after re-assignment: {remaining}")
    return assigned


# ── Fan-out resolution ─────────────────────────────────────────────────────────

def _build_hydraulic_graph(G_bfs, inverts):
    """
    Rebuild a directed graph from the BFS graph's adjacency, orienting every
    edge purely by computed inverts (high → low).  The BFS edge directions are
    ignored completely; only the node set and connectivity are reused.

    Edges whose endpoints are missing from inverts (pruned nodes) are skipped.
    Ties in invert are broken by ground elevation; if still tied, the BFS
    direction is preserved.

    Returns a new DiGraph guaranteed to be a DAG (inverts strictly decrease
    along every route_topdown path, so no invert-based cycle can form).
    """
    G_hyd = nx.DiGraph()
    G_hyd.add_nodes_from(G_bfs.nodes(data=True))

    seen = set()
    for u, v in G_bfs.to_undirected().edges():
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)

        inv_u = inverts.get(u)
        inv_v = inverts.get(v)
        if inv_u is None or inv_v is None:
            continue

        # Retrieve whichever directed edge exists in G_bfs for its attributes
        if G_bfs.has_edge(u, v):
            attrs = dict(G_bfs.edges[u, v])
        else:
            attrs = dict(G_bfs.edges[v, u])

        if inv_u > inv_v:
            G_hyd.add_edge(u, v, **attrs)
        elif inv_v > inv_u:
            G_hyd.add_edge(v, u, **attrs)
        else:
            # Tie: break by ground elevation
            ge_u = G_bfs.nodes[u].get('ground_elev', 0)
            ge_v = G_bfs.nodes[v].get('ground_elev', 0)
            if ge_u >= ge_v:
                G_hyd.add_edge(u, v, **attrs)
            else:
                G_hyd.add_edge(v, u, **attrs)

    return G_hyd


def resolve_fanouts(G_bfs, outfall_node, I_outfall, min_slope=0.0005, min_cover=0.0):
    """
    Detect and resolve hydraulic fan-outs (junctions with more than one
    hydraulic outlet) by rebuilding the directed graph from computed inverts
    rather than from BFS connectivity.

    Algorithm
    ---------
    1. Run route_topdown on the BFS graph to obtain invert levels at every node.
    2. Rebuild a hydraulic directed graph: same nodes + undirected adjacency as
       the BFS graph, but every edge is re-oriented high-invert → low-invert.
       This graph is independent of BFS choices.
    3. Detect fan-out junctions: out_degree > 1 in the hydraulic graph.
    4. For each fan-out junction J (processed upstream-first):
       - Evaluate each outlet by temporarily treating J as a source and routing
         through only that outlet.  The outlet yielding the highest outfall
         invert wins.
       - Sever all losing outlets from J.
    5. Re-run route_topdown on the resolved hydraulic graph to refresh inverts.

    Returns
    -------
    G_resolved  : hydraulic graph with all fan-outs resolved (out_degree ≤ 1)
    loser_edges : set of (junction_node, severed_neighbor_node) pairs
                  — used by swnetwork.py to apply the 10 m visual gap
    """
    if outfall_node not in G_bfs:
        return G_bfs.copy(), set()

    # ── Step 1: initial inverts from BFS graph ─────────────────────────────────
    inverts, _ = route_topdown(G_bfs, outfall_node, I_outfall, min_slope, min_cover)
    if not inverts:
        return G_bfs.copy(), set()

    # ── Step 2: build hydraulic graph ─────────────────────────────────────────
    G_hyd       = _build_hydraulic_graph(G_bfs, inverts)
    loser_edges = set()

    if outfall_node not in G_hyd:
        return G_hyd, loser_edges

    # ── Step 3 & 4: detect and resolve fan-outs ───────────────────────────────
    try:
        topo_order = list(nx.topological_sort(G_hyd))
    except nx.NetworkXUnfeasible:
        # Cycle in hydraulic graph (should not happen; return as-is)
        return G_hyd, loser_edges

    fanout_nodes = [n for n in topo_order
                    if G_hyd.out_degree(n) > 1 and n != outfall_node]

    for junction in fanout_nodes:
        outlets = list(G_hyd.successors(junction))
        if len(outlets) <= 1:
            continue  # already resolved by a prior iteration

        best_outlet  = None
        best_invert  = float('-inf')

        for outlet in outlets:
            G_eval = G_hyd.copy()
            # Treat junction as a fresh source: sever its incoming edges
            for pred in list(G_eval.predecessors(junction)):
                G_eval.remove_edge(pred, junction)
            # Keep only junction → outlet
            for other in list(G_eval.successors(junction)):
                if other != outlet:
                    G_eval.remove_edge(junction, other)

            inv_eval, _ = route_topdown(
                G_eval, outfall_node, I_outfall, min_slope, min_cover)
            of_inv = inv_eval.get(outfall_node, float('-inf'))

            if of_inv > best_invert:
                best_invert  = of_inv
                best_outlet  = outlet

        # Sever losing outlets
        for outlet in outlets:
            if outlet != best_outlet and G_hyd.has_edge(junction, outlet):
                G_hyd.remove_edge(junction, outlet)
                loser_edges.add((junction, outlet))

        # Refresh inverts after modifying the graph so downstream junctions
        # are evaluated on up-to-date invert data
        inverts, _ = route_topdown(G_hyd, outfall_node, I_outfall, min_slope, min_cover)

    return G_hyd, loser_edges

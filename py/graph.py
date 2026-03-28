"""graph.py — Build directed road graph per territory."""
import numpy as np
import networkx as nx
from collections import deque
from scipy.spatial import KDTree

NK = 2   # decimal places for node key rounding (0.01 m precision)

def round_node(x, y):
    return (round(x, NK), round(y, NK))


def build_territory_graph(assigned_segs, territory_id, outfall_xy=None,
                          connect_tol=0.5, outfall_snap_r=None):
    """
    Build a directed graph for one territory.

    assigned_segs:   list of dicts from roads.assign_to_catchments
    territory_id:    the territory to build (filters by seg['territory'])
    outfall_xy:      (x, y) of outfall snap position; when provided, edges are
                     oriented by BFS from the snap node (child -> parent -> snap),
                     so flow always moves toward the outfall regardless of local
                     terrain dips. When None, falls back to high->low orientation.
    connect_tol:     gap-heal distance within territory (m)
    outfall_snap_r:  if set, the nearest road node must be within this distance
                     of outfall_xy or the graph is returned empty (outfall is not
                     physically connected — segments go to pool for reassignment).

    Returns networkx.DiGraph with node attributes:
        x, y, ground_elev
    and edge attributes:
        length, seg_pts (original sampled points)
    """
    G = nx.DiGraph()

    # Collect segments for this territory
    seg_list = []
    for seg in assigned_segs:
        if seg['territory'] != territory_id:
            continue
        pts = seg['pts']
        if len(pts) < 2:
            continue
        s, e = pts[0], pts[-1]
        sk = round_node(s[0], s[1])
        ek = round_node(e[0], e[1])
        if sk == ek:
            continue
        seg_len = float(np.sum(np.hypot(np.diff([p[0] for p in pts]),
                                         np.diff([p[1] for p in pts]))))
        start_type = seg.get('start_node_type', 'normal')
        end_type   = seg.get('end_node_type',   'normal')
        seg_list.append((sk, ek, s, e, seg_len, pts, start_type, end_type))

    if not seg_list:
        return G

    # Add all nodes, propagating node_type (sag > normal; ridge beats normal)
    _TYPE_PRIORITY = {'sag': 2, 'ridge': 1, 'normal': 0}

    def _set_node_type(key, pt, ntype):
        if key not in G:
            G.add_node(key, x=pt[0], y=pt[1], ground_elev=pt[2], node_type=ntype)
        else:
            existing = G.nodes[key].get('node_type', 'normal')
            if _TYPE_PRIORITY.get(ntype, 0) > _TYPE_PRIORITY.get(existing, 0):
                G.nodes[key]['node_type'] = ntype

    for sk, ek, s, e, seg_len, pts, start_type, end_type in seg_list:
        _set_node_type(sk, s, start_type)
        _set_node_type(ek, e, end_type)

    if outfall_xy is not None:
        # ── BFS-oriented mode ──────────────────────────────────────────────────
        # Build undirected adjacency and store edge data in both directions
        adj = {n: [] for n in G.nodes()}
        edge_data = {}
        for sk, ek, s, e, seg_len, pts, *_types in seg_list:
            adj[sk].append(ek)
            adj[ek].append(sk)
            edge_data[(sk, ek)] = (seg_len, pts)
            edge_data[(ek, sk)] = (seg_len, list(reversed(pts)))

        # Find snap node: nearest node in G to outfall_xy
        nl  = list(G.nodes())
        nxy = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nl])
        snap_dist, idx = KDTree(nxy).query(outfall_xy)
        snap = nl[idx]

        # Reject if outfall is too far from the road network
        if outfall_snap_r is not None and snap_dist > outfall_snap_r:
            return nx.DiGraph()   # empty — all segments released to pool

        # BFS from snap outward: for each child→parent pair, add edge child→parent
        # (flow direction: child drains to parent, parent drains to snap = outfall)
        visited = {snap}
        queue   = deque([snap])
        while queue:
            parent = queue.popleft()
            for child in adj.get(parent, []):
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)
                L, pts_cp = edge_data[(child, parent)]
                G.add_edge(child, parent, length=L, seg_pts=pts_cp)

        # Remove nodes NOT reached by BFS (no road connection to outfall)
        unreachable = set(G.nodes()) - visited
        if unreachable:
            G.remove_nodes_from(unreachable)

    else:
        # ── High→low fallback (no outfall_xy provided) ────────────────────────
        for sk, ek, s, e, seg_len, pts, *_types in seg_list:
            if s[2] >= e[2]:
                G.add_edge(sk, ek, length=seg_len, seg_pts=pts)
            else:
                G.add_edge(ek, sk, length=seg_len, seg_pts=list(reversed(pts)))

    if G.number_of_nodes() == 0:
        return G

    # Gap-heal within territory — connect near-miss endpoints
    nl  = list(G.nodes())
    nxy = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nl])
    for i, j in KDTree(nxy).query_pairs(connect_tol):
        ni, nj = nl[i], nl[j]
        if G.has_edge(ni, nj) or G.has_edge(nj, ni):
            continue
        d = float(np.hypot(nxy[i,0]-nxy[j,0], nxy[i,1]-nxy[j,1]))
        if G.nodes[ni]['ground_elev'] >= G.nodes[nj]['ground_elev']:
            candidate = (ni, nj)
        else:
            candidate = (nj, ni)
        G.add_edge(candidate[0], candidate[1], length=d, seg_pts=None)
        # Reject if cycle introduced — remove the edge we just added
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(candidate[0], candidate[1])

    return G


def build_territory_graphs(assigned_segs, of_ids, of_xy_dict=None,
                           connect_tol=0.5, outfall_snap_r=None):
    """Build one graph per territory. Returns dict {of_id: DiGraph}.

    of_xy_dict:     {of_id: (x, y)} or None. When provided, each graph is built
                    with BFS-oriented edges flowing toward the outfall snap position.
    outfall_snap_r: passed through to build_territory_graph — outfalls farther
                    than this from any road node produce an empty graph.
    """
    return {
        of_id: build_territory_graph(
            assigned_segs, of_id,
            outfall_xy=of_xy_dict.get(of_id) if of_xy_dict else None,
            connect_tol=connect_tol,
            outfall_snap_r=outfall_snap_r
        )
        for of_id in of_ids
    }

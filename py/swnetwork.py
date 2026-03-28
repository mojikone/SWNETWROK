"""
swnetwork.py  v5  -- SW Drainage Sub-network Assignment
=========================================================
Stage 1: DEM catchment delineation  (territory — no hydraulic assumptions)
Stage 2: Directed road graph per territory
Stage 3: Top-down invert routing + bottleneck pruning
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
import networkx as nx
import ezdxf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely.ops import substring as shapely_substring
from scipy.spatial import KDTree

from dem        import sample_elev, delineate_catchments
from roads      import node_roads, ridge_split, ridge_sag_split
from graph      import build_territory_graphs
from hydraulics import route_topdown, prune_to_feasibility
from outputs    import write_shp, write_dxf, write_img

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = "D:/Projects/Renardet/SW Net - 2"
W2          = f"{BASE}/W2"
ROADS_SHP   = f"{BASE}/SHP/Roads.shp"
OUTFALL_SHP = f"{BASE}/SHP/outfall.shp"
DEM_TIF     = f"{BASE}/Terrain/NSA 5m test.tif"

# ── Parameters ────────────────────────────────────────────────────────────────
SPACING      = 2.0      # m   elevation sample interval along roads
RIDGE_RISE   = 0.05     # m   min rise to detect ridge
SAG_DROP     = 0.05     # m   min drop to detect sag/inlet
MIN_SEG_LEN  = 0.5      # m   discard noded sub-segments shorter than this
CONNECT_TOL  = 0.5      # m   gap-heal snap tolerance within territory
SNAP_RADIUS  = 50.0     # m   outfall snap to high-accumulation DEM cell
OUTFALL_SNAP_R = 30.0   # m  outfall road-snap search radius
MIN_SLOPE    = 0.0005   # m/m minimum pipe gradient
MIN_COVER    = 1.0      # m   minimum cover below ground
MAX_COVER    = 3.0      # m   maximum allowable pipe depth (hard cap)
FANOUT_GAP_M = 10.0     # m   visual gap at loser-channel upstream end (fan-out resolution)


if __name__ == "__main__":
    from collections import Counter
    from scipy.spatial import KDTree as _KDT

    print("\n  SW Network  v5.1")
    print("  ================\n")

    # [1] Load data
    print("  [1/9]  Loading data...")
    roads_gdf = gpd.read_file(ROADS_SHP).to_crs("EPSG:32640")
    of_gdf    = gpd.read_file(OUTFALL_SHP).to_crs("EPSG:32640")
    of_pts    = [(int(r["id"]), r.geometry.x, r.geometry.y)
                 for _, r in of_gdf.iterrows()]
    of_depths = {int(r["id"]): float(r["depth"]) if "depth" in of_gdf.columns and r["depth"] is not None else 0.0
                 for _, r in of_gdf.iterrows()}
    print(f"         Roads: {len(roads_gdf)}   Outfalls: {len(of_pts)}")

    # [2] Load DEM
    print("\n  [2/9]  Loading DEM...")
    from dem import load_dem, make_sampler, delineate_catchments
    dem_data, dem_tfm, dem_nodata = load_dem(DEM_TIF)
    elev_at = make_sampler(dem_data, dem_tfm, dem_nodata)

    # [3] Node roads + ridge+sag split (outfall splits dropped — snapping handles it)
    print("\n  [3/9]  Noding roads + ridge+sag split...")
    from roads import (node_roads, ridge_sag_split, assign_majority,
                       snap_outfalls_to_road_graph, reassign_boundary_roads)
    noded      = node_roads(roads_gdf, min_seg_len=MIN_SEG_LEN)
    seg_tuples = ridge_sag_split(noded, elev_at, SPACING, RIDGE_RISE, SAG_DROP)
    print(f"         {len(seg_tuples)} sub-segments after ridge+sag split")

    # [4] Snap outfalls to road-graph local minima
    print("\n  [4/9]  Snapping outfalls to road network low points...")
    of_pts, seg_tuples = snap_outfalls_to_road_graph(
        of_pts, seg_tuples, OUTFALL_SNAP_R, elev_at=elev_at)

    # [5] Sample outfall invert levels (at snapped positions)
    print("\n  [5/9]  Outfall invert levels...")
    of_inverts = {}
    of_grounds = {}
    for of_id, ox, oy in of_pts:
        e = elev_at(ox, oy)
        if e is None:
            print(f"         WARNING OF{of_id}: outside DEM, using 0.0 m")
            e = 0.0
        depth = of_depths.get(of_id, 0.0)
        of_grounds[of_id]  = e
        of_inverts[of_id]  = e - depth  # outfall invert = ground - depth
        print(f"         OF{of_id}: ground={e:.2f}  depth={depth:.2f}  I_outfall={of_inverts[of_id]:.2f}")

    # Routing feasibility threshold: I_outfall is a physical constraint.
    # Monotone paths arrive exactly at I_outfall; non-monotone paths (uphill
    # sections, sources below outfall) arrive deeper and are pruned.
    of_inverts_routing = dict(of_inverts)

    # [6] Catchment delineation (using snapped positions)
    print("\n  [6/9]  Catchment delineation (D8)...")
    catchments = delineate_catchments(DEM_TIF, of_pts, SNAP_RADIUS)

    # Buffer catchments by one DEM cell to absorb raster edge uncertainty
    DEM_CELL_M = 5.0
    catchments_buffered = {
        tid: poly.buffer(DEM_CELL_M) if poly is not None else None
        for tid, poly in catchments.items()
    }

    # [7] Assign segments to catchments (majority overlap, no splitting)
    print("\n  [7/9]  Territory assignment (majority overlap)...")
    assigned = assign_majority(seg_tuples, catchments_buffered)
    cnts = Counter(s['territory'] for s in assigned)
    for tid, n in sorted(cnts.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        label = f"OF{tid}" if tid else "UNASSIGNED"
        print(f"         {label}: {n} segments")

    # [8] Build directed graphs + gravity check + boundary reassignment
    #     + hydraulic routing + pool re-assignment
    print("\n  [8/9]  Graphs, gravity check, hydraulics, pool re-assignment...")
    from graph      import build_territory_graphs, build_territory_graph, round_node
    from hydraulics import (prune_to_feasibility, pool_reassignment_loop)

    of_ids = [fid for fid, _, _ in of_pts]
    of_xy_dict = {fid: (ox, oy) for fid, ox, oy in of_pts}
    graphs = build_territory_graphs(assigned, of_ids, of_xy_dict, CONNECT_TOL,
                                    outfall_snap_r=OUTFALL_SNAP_R)

    inverts_by_territory = {}
    pruned_by_territory  = {}

    for of_id, ox, oy in of_pts:
        G = graphs[of_id]
        if G.number_of_nodes() == 0:
            print(f"\n         OF{of_id}: empty graph — skip")
            inverts_by_territory[of_id] = {}
            pruned_by_territory[of_id]  = set()
            continue

        # Find outfall snap node (nearest graph node to snapped outfall position)
        nl  = list(G.nodes())
        nxy = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in nl])
        d, i = _KDT(nxy).query([ox, oy])
        snap = nl[i]
        d    = float(d)

        print(f"\n         OF{of_id}: snap={d:.1f}m  "
              f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

        inverts, pruned, status = prune_to_feasibility(
            G, snap, of_inverts_routing[of_id], MIN_SLOPE, MIN_COVER,
            max_cover=MAX_COVER)

        inverts_by_territory[of_id] = inverts
        pruned_by_territory[of_id]  = pruned

        # Pruned nodes → pool (blacklist current territory)
        for seg in assigned:
            if seg['territory'] != of_id:
                continue
            pts  = seg['pts']
            nk_s = round_node(pts[0][0], pts[0][1])
            nk_e = round_node(pts[-1][0], pts[-1][1])
            if nk_s in pruned or nk_e in pruned:
                seg['blacklist'].add(of_id)
                seg['territory'] = None

        n_pruned   = len(pruned)
        n_assigned = G.number_of_nodes() - n_pruned
        print(f"         OF{of_id}: {status}  assigned={n_assigned}  pruned={n_pruned}")

    # Release road-disconnected segments to pool
    # Segments assigned by majority-catchment overlap but whose nodes are NOT
    # in their territory's BFS graph (no road path to the outfall snap node)
    # stay floating with a territory tag but get no invert data.
    # Free them here so pool_reassignment_loop can try other territories.
    print("\n         Releasing road-disconnected segments to pool...")
    n_released = 0
    for seg in assigned:
        tid = seg.get('territory')
        if tid is None:
            continue
        G_chk = graphs.get(tid)
        if G_chk is None or G_chk.number_of_nodes() == 0:
            seg['territory'] = None
            n_released += 1
            continue
        pts  = seg['pts']
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        if nk_s not in G_chk and nk_e not in G_chk:
            seg['territory'] = None   # don't blacklist — try other territories
            n_released += 1
    print(f"         Released {n_released} segments to pool")

    # Boundary road reassignment (convergence loop)
    print("\n         Boundary road reassignment...")
    assigned = reassign_boundary_roads(assigned, graphs)

    # Pool re-assignment loop
    print("\n         Pool re-assignment loop...")
    assigned = pool_reassignment_loop(
        assigned, graphs, of_inverts_routing, MIN_SLOPE, MIN_COVER,
        max_cover=MAX_COVER,
        of_xy=of_xy_dict, outfall_snap_r=OUTFALL_SNAP_R)

    # Recompute inverts on final territory assignments
    # (pool reassignment adds segments whose inverts were not yet stored)
    print("\n         Recomputing inverts on final territory assignments...")
    from hydraulics import route_topdown, prune_to_feasibility
    graphs_final = build_territory_graphs(assigned, of_ids, of_xy_dict, CONNECT_TOL,
                                          outfall_snap_r=OUTFALL_SNAP_R)
    for of_id, ox, oy in of_pts:
        G_f = graphs_final[of_id]
        if G_f.number_of_nodes() == 0:
            continue
        nl_f  = list(G_f.nodes())
        nxy_f = np.array([[G_f.nodes[n]['x'], G_f.nodes[n]['y']] for n in nl_f])
        _, i_f = _KDT(nxy_f).query([ox, oy])
        snap_f = nl_f[i_f]

        fresh_inv, final_pruned, _ = prune_to_feasibility(
            G_f, snap_f, of_inverts_routing[of_id], MIN_SLOPE, MIN_COVER,
            max_cover=MAX_COVER)

        # Segments pruned in final recompute → return to orphan
        for seg in assigned:
            if seg['territory'] != of_id:
                continue
            pts  = seg['pts']
            nk_s = round_node(pts[0][0], pts[0][1])
            nk_e = round_node(pts[-1][0], pts[-1][1])
            if nk_s in final_pruned or nk_e in final_pruned:
                seg['territory'] = None

        if snap_f in fresh_inv:
            fresh_inv[snap_f] = of_inverts[of_id]
        inverts_by_territory[of_id] = fresh_inv
        pruned_by_territory[of_id]  = pruned_by_territory.get(of_id, set()) | final_pruned
        graphs[of_id] = G_f   # use final graph for write_shp

    # [8b] Fan-out resolution — enforce tree structure (out_degree ≤ 1 at all non-outfall nodes)
    #
    # Single GLOBAL pass across ALL territories so cross-territory junctions are
    # also caught: two channels from different territories cannot both exit the same
    # upstream junction.
    #
    # Effective invert at each endpoint:
    #   • actual computed value from inverts_by_territory  (node is in the BFS graph)
    #   • ground_elevation − MIN_COVER                     (off-graph estimate)
    # The higher-invert endpoint is the "junction" (water exits from there).
    #
    # Fan-out rule: if >1 segment exits a junction →
    #   winner  = steepest hydraulic drop (lowest far-endpoint invert)
    #   losers  → 10 m gap from the junction end; head invert corrected to
    #             max(ground_at_new_head − MIN_COVER, far_eff + MIN_SLOPE × remaining)
    #             so the flow arrow always points AWAY from the new source.
    #   loser with remaining length < FANOUT_GAP_M → orphaned (territory = None)
    print("\n  [8b] Resolving fan-out violations...")

    from collections import defaultdict

    def _trim_pts_from_start(pts, gap_m):
        """Trim the first gap_m metres from a pts list.
        Returns new pts or None if the segment is shorter than gap_m."""
        cum = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            d  = float(np.hypot(dx, dy))
            if cum + d >= gap_m:
                rem  = gap_m - cum
                frac = rem / d if d > 0.0 else 0.0
                nx_  = pts[i-1][0] + frac * dx
                ny_  = pts[i-1][1] + frac * dy
                nz_  = pts[i-1][2] + frac * (pts[i][2] - pts[i-1][2])
                return [(nx_, ny_, nz_)] + list(pts[i:])
            cum += d
        return None  # segment shorter than gap_m

    def _trim_pts_from_end(pts, gap_m):
        """Trim the last gap_m metres from a pts list."""
        rev = _trim_pts_from_start(list(reversed(pts)), gap_m)
        return list(reversed(rev)) if rev is not None else None

    def _seg_length(pts):
        """2-D arc-length of a pts list."""
        total = 0.0
        for i in range(1, len(pts)):
            total += float(np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]))
        return total

    # Collect every outfall snap node (outlets converging here are never gapped)
    outfall_snap_nodes = set()
    for of_id, ox, oy in of_pts:
        G_f = graphs[of_id]
        if G_f.number_of_nodes() == 0:
            continue
        nl_f  = list(G_f.nodes())
        nxy_f = np.array([[G_f.nodes[n]['x'], G_f.nodes[n]['y']] for n in nl_f])
        _, i_f = _KDT(nxy_f).query([ox, oy])
        outfall_snap_nodes.add(nl_f[i_f])

    # Global outlet registry across ALL territories:
    #   outlet_segs[junction_key] = [(territory, gap_end, seg, far_eff_inv), ...]
    #   gap_end: 'start' → junction is pts[0];  'end' → junction is pts[-1]
    outlet_segs  = defaultdict(list)
    # inlet_counts[junction_key][territory_id] = number of segments in that territory
    # whose flow ARRIVES at junction_key (junction is their downstream/low-invert end).
    # Used for connectivity-first winner selection at cross-territory junctions.
    inlet_counts = defaultdict(lambda: defaultdict(int))

    for seg in assigned:
        tid = seg.get('territory')
        if tid is None:
            continue
        pts = seg['pts']
        if len(pts) < 2:
            continue
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])

        inv_dict = inverts_by_territory.get(tid, {})
        eff_s = inv_dict.get(nk_s, pts[0][2] - MIN_COVER)
        eff_e = inv_dict.get(nk_e, pts[-1][2] - MIN_COVER)

        if eff_s > eff_e:
            # Water exits from nk_s toward nk_e; nk_s is a potential junction for fan-outs,
            # nk_e is the downstream end (inlet to nk_e)
            outlet_segs[nk_s].append((tid, 'start', seg, eff_e))
            inlet_counts[nk_e][tid] += 1
        elif eff_e > eff_s:
            outlet_segs[nk_e].append((tid, 'end', seg, eff_s))
            inlet_counts[nk_s][tid] += 1
        # tied inverts: no clear hydraulic direction — skip

    # Resolve fan-outs
    head_inverts_by_territory = defaultdict(dict)
    n_gaps_total   = 0
    n_orphan_total = 0

    for junction, outlets in outlet_segs.items():
        if len(outlets) <= 1:
            continue
        if junction in outfall_snap_nodes:
            continue  # channels converging at an outfall are correct

        # ── Winner selection ───────────────────────────────────────────────────
        # Cross-territory fan-out: connectivity first.
        #   The outlet whose territory has the most segments ARRIVING at this junction
        #   (tributaries) wins — losing it would disconnect those upstream segments
        #   from their outfall because the winner belongs to a different territory.
        #   Tie-break: steepest hydraulic drop (lowest far_eff_inv).
        #
        # Same-territory fan-out: all tributaries still reach the outfall via the
        #   winner, so connectivity is always preserved → steepest drop only.
        territories_at_junction = {t for t, _, _, _ in outlets}
        if len(territories_at_junction) > 1:
            trib = [inlet_counts[junction].get(t, 0) for t, _, _, _ in outlets]
            if max(trib) > 0:
                winner_idx = max(range(len(outlets)),
                                 key=lambda i: (trib[i], -outlets[i][3]))
            else:
                winner_idx = min(range(len(outlets)), key=lambda i: outlets[i][3])
        else:
            winner_idx = min(range(len(outlets)), key=lambda i: outlets[i][3])

        for idx, (tid, gap_end, seg, _far_inv) in enumerate(outlets):
            if idx == winner_idx:
                continue  # winner is unchanged

            pts = seg['pts']
            if gap_end == 'start':
                new_pts = _trim_pts_from_start(pts, FANOUT_GAP_M)
                if new_pts and len(new_pts) >= 2:
                    head_nk = round_node(new_pts[0][0], new_pts[0][1])
                    far_nk  = round_node(new_pts[-1][0], new_pts[-1][1])
                    far_eff = inverts_by_territory.get(tid, {}).get(
                                  far_nk, new_pts[-1][2] - MIN_COVER)
                    rem_len = _seg_length(new_pts)
                    # Feasibility: a valid independent head must satisfy MIN_COVER
                    # and deliver water to the far end at >= MIN_SLOPE.
                    # If the slope constraint pushes head_inv above ground−MIN_COVER,
                    # the pipe would be shallower than MIN_COVER → orphan (route_topdown
                    # would prune this same violation).
                    if far_eff + MIN_SLOPE * rem_len > new_pts[0][2] - MIN_COVER:
                        seg['territory'] = None   # cover violation → orphan
                        n_orphan_total  += 1
                    else:
                        seg['pts']  = new_pts
                        seg['geom'] = LineString([(p[0], p[1]) for p in new_pts])
                        head_inverts_by_territory[tid][head_nk] = (
                            new_pts[0][2] - MIN_COVER)
                        n_gaps_total += 1
                else:
                    seg['territory'] = None   # too short for a gap → orphan
                    n_orphan_total  += 1
            else:  # 'end'
                new_pts = _trim_pts_from_end(pts, FANOUT_GAP_M)
                if new_pts and len(new_pts) >= 2:
                    head_nk = round_node(new_pts[-1][0], new_pts[-1][1])
                    far_nk  = round_node(new_pts[0][0], new_pts[0][1])
                    far_eff = inverts_by_territory.get(tid, {}).get(
                                  far_nk, new_pts[0][2] - MIN_COVER)
                    rem_len = _seg_length(new_pts)
                    if far_eff + MIN_SLOPE * rem_len > new_pts[-1][2] - MIN_COVER:
                        seg['territory'] = None   # cover violation → orphan
                        n_orphan_total  += 1
                    else:
                        seg['pts']  = new_pts
                        seg['geom'] = LineString([(p[0], p[1]) for p in new_pts])
                        head_inverts_by_territory[tid][head_nk] = (
                            new_pts[-1][2] - MIN_COVER)
                        n_gaps_total += 1
                else:
                    seg['territory'] = None   # too short for a gap → orphan
                    n_orphan_total  += 1

    # Inject corrected head inverts into each territory's invert dict
    for tid, hd in head_inverts_by_territory.items():
        inverts_by_territory[tid].update(hd)
        print(f"    OF{tid}: {len(hd)} fan-out gap(s) applied")

    if n_gaps_total == 0 and n_orphan_total == 0:
        print("    No fan-out violations found.")
    else:
        print(f"\n    Total: {n_gaps_total} gap(s) applied"
              f"   {n_orphan_total} loser(s) orphaned (cover violation or too short)")

    # Post-injection slope cleanup.
    # Fan-out resolutions are processed in junction order in a single pass.
    # If junction B's resolution injects a new head invert at the far-end node
    # of a segment that was already resolved at junction A (earlier in the same
    # pass), the feasibility check for A used the old far_eff, but the injected
    # dict now shows the new value — potentially yielding slope < MIN_SLOPE.
    # A second pass over all assigned segments catches and orphans these.
    n_post_orphan = 0
    for seg in assigned:
        tid = seg.get('territory')
        if tid is None:
            continue
        pts = seg['pts']
        if len(pts) < 2:
            continue
        inv_dict = inverts_by_territory.get(tid, {})
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        inv_s = inv_dict.get(nk_s)
        inv_e = inv_dict.get(nk_e)
        if inv_s is None or inv_e is None:
            continue
        seg_len = _seg_length(pts)
        if seg_len < 1e-3:
            continue
        if (max(inv_s, inv_e) - min(inv_s, inv_e)) / seg_len < MIN_SLOPE - 1e-9:
            seg['territory'] = None
            n_post_orphan   += 1
    if n_post_orphan:
        print(f"    Post-injection cleanup: {n_post_orphan} segment(s) orphaned"
              f" (slope < MIN_SLOPE after head-invert update)")

    # [8c] Hydraulic audit — verify MIN_COVER, MAX_COVER, MIN_SLOPE on every node
    print("\n  [8c] Hydraulic audit...")

    # Outfall discharge nodes are excluded from the MIN_COVER check: they are
    # open-end structures where the pipe exits at ground level (depth = 0 is correct).
    _outfall_nks = set(round_node(ox, oy) for _, ox, oy in of_pts)

    viol_min_cover  = []   # (tid, nk, ground, inv, depth)
    viol_max_cover  = []   # (tid, nk, ground, inv, depth)
    viol_min_slope  = []   # (tid, seg_id, inv_up, inv_dn, length, slope)

    for seg_idx, seg in enumerate(assigned):
        tid = seg.get('territory')
        if tid is None:
            continue
        pts = seg['pts']
        if len(pts) < 2:
            continue
        inv_dict = inverts_by_territory.get(tid, {})
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        inv_s = inv_dict.get(nk_s)
        inv_e = inv_dict.get(nk_e)

        # Node cover checks (skip outfall discharge nodes)
        for nk, pt, inv in [(nk_s, pts[0], inv_s), (nk_e, pts[-1], inv_e)]:
            if inv is None or nk in _outfall_nks:
                continue
            depth = pt[2] - inv
            if depth < MIN_COVER - 1e-6:
                viol_min_cover.append((tid, nk, pt[2], inv, depth))
            if depth > MAX_COVER + 1e-6:
                viol_max_cover.append((tid, nk, pt[2], inv, depth))

        # Slope check — only when both endpoint inverts are available
        if inv_s is not None and inv_e is not None:
            seg_len = _seg_length(pts)
            if seg_len > 1e-3:
                inv_up = max(inv_s, inv_e)
                inv_dn = min(inv_s, inv_e)
                slope  = (inv_up - inv_dn) / seg_len
                if slope < MIN_SLOPE - 1e-9:
                    viol_min_slope.append(
                        (tid, seg_idx, inv_up, inv_dn, seg_len, slope))

    # Deduplicate node violations (same node can appear in multiple segments)
    seen_min = set(); seen_max = set()
    viol_min_cover = [v for v in viol_min_cover
                      if v[1] not in seen_min and not seen_min.add(v[1])]
    viol_max_cover = [v for v in viol_max_cover
                      if v[1] not in seen_max and not seen_max.add(v[1])]

    # Print results
    ok = True
    if viol_min_cover:
        ok = False
        print(f"    MIN_COVER ({MIN_COVER} m) violated at {len(viol_min_cover)} node(s):")
        for tid, nk, g, inv, d in viol_min_cover[:20]:
            print(f"      OF{tid}  node={nk}  G={g:.2f}  I={inv:.2f}  D={d:.2f}")
        if len(viol_min_cover) > 20:
            print(f"      ... and {len(viol_min_cover)-20} more")
    if viol_max_cover:
        ok = False
        print(f"    MAX_COVER ({MAX_COVER} m) violated at {len(viol_max_cover)} node(s):")
        for tid, nk, g, inv, d in viol_max_cover[:20]:
            print(f"      OF{tid}  node={nk}  G={g:.2f}  I={inv:.2f}  D={d:.2f}")
        if len(viol_max_cover) > 20:
            print(f"      ... and {len(viol_max_cover)-20} more")
    if viol_min_slope:
        ok = False
        print(f"    MIN_SLOPE ({MIN_SLOPE*100:.4f}%) violated at {len(viol_min_slope)} segment(s):")
        for tid, sidx, iu, id_, ln, sl in viol_min_slope[:20]:
            print(f"      OF{tid}  seg={sidx}  I_up={iu:.4f}  I_dn={id_:.4f}"
                  f"  L={ln:.1f}m  slope={sl*100:.5f}%")
        if len(viol_min_slope) > 20:
            print(f"      ... and {len(viol_min_slope)-20} more")
    if ok:
        print("    All nodes and segments pass MIN_COVER, MAX_COVER, and MIN_SLOPE.")

    # [9] Outputs
    print("\n  [9/9]  Writing outputs...")
    from outputs import write_shp, write_dxf, write_img
    write_shp(assigned, graphs, inverts_by_territory, pruned_by_territory,
              of_pts, catchments, f"{W2}/shp",
              of_inverts=of_inverts, min_slope=MIN_SLOPE, min_cover=MIN_COVER)
    write_dxf(assigned, inverts_by_territory, pruned_by_territory,
              of_pts, of_grounds, of_inverts, f"{W2}/dxf/swnetwork.dxf",
              min_cover=MIN_COVER, min_slope=MIN_SLOPE)
    write_img(assigned, catchments, of_pts, f"{W2}/img")

    print("\n  Done.")

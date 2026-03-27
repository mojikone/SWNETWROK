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
from roads      import node_roads, ridge_split
from graph      import build_territory_graphs
from hydraulics import route_topdown, prune_to_feasibility
from outputs    import write_shp, write_dxf, write_img

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = "D:/Projects/Renardet/SW Net"
W2          = f"{BASE}/W2"
ROADS_SHP   = f"{BASE}/SHP/Roads.shp"
OUTFALL_SHP = f"{BASE}/SHP/outfall.shp"
DEM_TIF     = f"{BASE}/Terrain/NSA 5m test.tif"

# ── Parameters ────────────────────────────────────────────────────────────────
SPACING      = 2.0      # m   elevation sample interval along roads
RIDGE_RISE   = 0.05     # m   min rise to detect ridge
MIN_SEG_LEN  = 0.5      # m   discard noded sub-segments shorter than this
CONNECT_TOL  = 0.5      # m   gap-heal snap tolerance within territory
SNAP_RADIUS  = 50.0     # m   outfall snap to high-accumulation DEM cell
OUTFALL_SNAP_R = 30.0   # m  outfall road-snap search radius
MIN_SLOPE    = 0.0005   # m/m minimum pipe gradient
MIN_COVER    = 1.0      # m   minimum cover below ground
MAX_COVER    = 3.0      # m   maximum allowable pipe depth (hard cap)


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

    # [3] Node roads + ridge-split (outfall splits dropped — snapping handles it)
    print("\n  [3/9]  Noding roads + ridge-split...")
    from roads import (node_roads, ridge_split, assign_majority,
                       snap_outfalls_to_road_graph, reassign_boundary_roads)
    noded      = node_roads(roads_gdf, min_seg_len=MIN_SEG_LEN)
    seg_tuples = ridge_split(noded, elev_at, SPACING, RIDGE_RISE)
    print(f"         {len(seg_tuples)} sub-segments after ridge-split")

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

    # [9] Outputs
    print("\n  [9/9]  Writing outputs...")
    from outputs import write_shp, write_dxf, write_img
    write_shp(assigned, graphs, inverts_by_territory, pruned_by_territory,
              of_pts, catchments, f"{W2}/shp")
    write_dxf(assigned, inverts_by_territory, pruned_by_territory,
              of_pts, of_grounds, of_inverts, f"{W2}/dxf/swnetwork.dxf")
    write_img(assigned, catchments, of_pts, f"{W2}/img")

    print("\n  Done.")

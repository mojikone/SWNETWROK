"""outputs.py — SHP, DXF, PNG export for v5 results."""
import os, heapq, math
from collections import defaultdict
import numpy as np
import geopandas as gpd
import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point
from graph import round_node

ACI_COLORS = [1, 2, 3, 4, 5, 6]   # red, yellow, green, cyan, blue, magenta

TEXT_HT    = 1.5          # m — annotation text height
JUNC_R     = 1.5          # m — junction dot radius
OF_R_INNER = 3.0          # m — outfall inner circle radius
OF_R_OUTER = 6.0          # m — outfall outer circle radius
ARROW_LEN  = TEXT_HT * 2.5


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _add_circle(msp, cx, cy, radius, layer):
    msp.add_circle((cx, cy), radius, dxfattribs={'layer': layer})


def _add_flow_tick(msp, pts, layer, reverse=False,
                   name_label=None, slope_label=None, length_label=None):
    """
    Chevron tick at segment midpoint indicating flow direction.

    Label layout (perpendicular to the pipe, at midpoint):
        name_label  — above the line (left of downstream direction)
        [line itself]
        slope_label — below the line (right of downstream direction)
        length_label — further below slope
    """
    if len(pts) < 2:
        return
    if reverse:
        dx = pts[0][0] - pts[-1][0];  dy = pts[0][1] - pts[-1][1]
    else:
        dx = pts[-1][0] - pts[0][0];  dy = pts[-1][1] - pts[0][1]
    seg_len_2d = np.hypot(dx, dy)
    if seg_len_2d < 0.01:
        return
    dx /= seg_len_2d;  dy /= seg_len_2d   # unit downstream vector
    # perpendicular vectors: left = (-dy, dx), right = (dy, -dx)

    _line = LineString([(p[0], p[1]) for p in pts])
    _mid  = _line.interpolate(0.5, normalized=True)
    mx, my = _mid.x, _mid.y

    angle = math.atan2(dy, dx)
    for sign in (+1, -1):
        a  = angle + math.pi + sign * math.radians(30)
        ex = mx + ARROW_LEN * math.cos(a)
        ey = my + ARROW_LEN * math.sin(a)
        msp.add_line((mx, my), (ex, ey), dxfattribs={'layer': layer})

    rot_deg = math.degrees(angle)
    if rot_deg >  90: rot_deg -= 180
    elif rot_deg < -90: rot_deg += 180

    off = TEXT_HT * 1.5   # perpendicular offset from pipe centerline

    # name — above the line (left of downstream)
    if name_label is not None:
        ax = mx - dy * off
        ay = my + dx * off
        msp.add_text(name_label, dxfattribs={
            'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
            'color': 3, 'insert': (ax, ay), 'rotation': rot_deg})

    # slope — below the line (right of downstream)
    if slope_label is not None:
        bx = mx + dy * off
        by = my - dx * off
        msp.add_text(slope_label, dxfattribs={
            'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
            'color': 2, 'insert': (bx, by), 'rotation': rot_deg})

    # length — below slope (further right)
    if length_label is not None:
        cx_ = mx + dy * off * 2.5
        cy_ = my - dx * off * 2.5
        msp.add_text(length_label, dxfattribs={
            'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
            'color': 2, 'insert': (cx_, cy_), 'rotation': rot_deg})


def _safe_write_shp(rows, crs, path):
    if not rows:
        print(f"  (skipped empty layer: {os.path.basename(path)})")
        return
    gpd.GeoDataFrame(rows, crs=crs).to_file(path)


def _aci(of_id):
    return ACI_COLORS[(of_id - 1) % len(ACI_COLORS)]


def _seg_len(pts):
    return sum(
        float(np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]))
        for i in range(len(pts)-1))


# ── Network attribute builder ─────────────────────────────────────────────────

def _build_network_attrs(assigned_segs, inverts_by_territory, pruned_by_territory,
                          outfall_pts, of_inverts, min_cover, min_slope):
    """
    Compute hydraulic-order names and per-channel attributes for every
    non-orphan node and segment.

    Node naming convention (farthest from outfall = index 1):
        Outfall OF1 → "O1"
        Junction  → "O1-J1", "O1-J2", …
        Sag inlet → "O1-S1", "O1-S2", …
        Ridge     → "O1-R1", "O1-R2", …

    Channel naming:
        "O1-C1" = channel whose upstream end is farthest from outfall.

    Channel inverts are the pipe's OWN inverts at each endpoint:
        inv_up = upstream node invert (from route_topdown / estimate)
        inv_dn = inv_up − s_pipe × L
                 where s_pipe = max(min_slope, (inv_up − target_dn) / L)
                 and   target_dn = ground_dn − min_cover  (or outfall invert)
    This can differ from the junction node invert when another pipe
    dominates the junction.

    Returns a dict with all naming and attribute data.
    """
    TYPE_PRI = {'outfall': 0, 'sag': 1, 'ridge': 2, 'junction': 3}

    outfall_nk_to_id = {round_node(ox, oy): of_id for of_id, ox, oy in outfall_pts}

    # ── Phase 1: classify every endpoint ─────────────────────────────────────
    node_type      = {}   # nk → type str
    node_ground    = {}   # nk → float
    node_invert    = {}   # nk → float (None for orphan-only nodes)
    node_territory = {}   # nk → of_id (primary territory)
    node_xy        = {}   # nk → (x, y)  actual coordinates
    active_nks     = set()
    all_nk_xy      = {}   # nk → (x, y) including orphan nodes

    for seg in assigned_segs:
        pts = seg['pts']
        if len(pts) < 2:
            continue
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        all_nk_xy.setdefault(nk_s, (pts[0][0], pts[0][1]))
        all_nk_xy.setdefault(nk_e, (pts[-1][0], pts[-1][1]))

        tid    = seg.get('territory')
        pruned = pruned_by_territory.get(tid, set()) if tid else set()
        # Segments with pruned endpoints are treated as orphan for naming
        is_active = (tid is not None
                     and round_node(pts[0][0], pts[0][1]) not in pruned
                     and round_node(pts[-1][0], pts[-1][1]) not in pruned)

        inv_dict = inverts_by_territory.get(tid, {}) if tid else {}

        for nk, pt, pt_idx in [(nk_s, pts[0], 0), (nk_e, pts[-1], -1)]:
            node_xy.setdefault(nk, (pt[0], pt[1]))
            node_ground.setdefault(nk, pt[2])

            if not is_active:
                continue

            active_nks.add(nk)
            inv = inv_dict.get(nk, pt[2] - min_cover)
            node_invert.setdefault(nk, inv)
            node_territory.setdefault(nk, tid)

            # Determine type (highest priority wins)
            if nk in outfall_nk_to_id:
                raw_type = 'outfall'
                node_territory[nk] = outfall_nk_to_id[nk]
            elif seg.get('start_node_type' if pt_idx == 0 else 'end_node_type') == 'sag':
                raw_type = 'sag'
            elif seg.get('start_node_type' if pt_idx == 0 else 'end_node_type') == 'ridge':
                raw_type = 'ridge'
            else:
                raw_type = 'junction'

            current = node_type.get(nk, 'junction')
            if TYPE_PRI[raw_type] < TYPE_PRI[current]:
                node_type[nk] = raw_type
        # fill any node not yet typed
        for nk in (nk_s, nk_e):
            if nk in active_nks:
                node_type.setdefault(nk, 'junction')

    orphan_nks = {nk for nk in all_nk_xy if nk not in active_nks}

    # ── Phase 2: Dijkstra distance from outfall snap per territory ────────────
    adj_by_territory = defaultdict(lambda: defaultdict(list))
    for seg in assigned_segs:
        tid    = seg.get('territory')
        pruned = pruned_by_territory.get(tid, set()) if tid else set()
        if tid is None:
            continue
        pts = seg['pts']
        if len(pts) < 2:
            continue
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        if nk_s in pruned or nk_e in pruned:
            continue
        length = _seg_len(pts)
        adj_by_territory[tid][nk_s].append((nk_e, length))
        adj_by_territory[tid][nk_e].append((nk_s, length))

    def _dijkstra(adj, start):
        dist = {start: 0.0}
        pq   = [(0.0, start)]
        while pq:
            d, nk = heapq.heappop(pq)
            if d > dist.get(nk, float('inf')):
                continue
            for nb, w in adj.get(nk, []):
                nd = d + w
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        return dist

    dist_from_outfall = {}   # nk → float (distance from its territory's outfall)
    for of_id, ox, oy in outfall_pts:
        snap_nk = round_node(ox, oy)
        d = _dijkstra(adj_by_territory[of_id], snap_nk)
        for nk, dv in d.items():
            dist_from_outfall.setdefault(nk, dv)

    # ── Phase 3: assign names ─────────────────────────────────────────────────
    of_prefix = {of_id: f"O{of_id}" for of_id, _, _ in outfall_pts}
    node_names = {}

    # Outfall nodes first
    for of_id, ox, oy in outfall_pts:
        snap_nk = round_node(ox, oy)
        if snap_nk in active_nks:
            node_names[snap_nk] = of_prefix[of_id]

    nodes_by_territory = defaultdict(list)
    for nk in active_nks:
        if nk in node_names:      # already named (outfall)
            continue
        tid = node_territory.get(nk)
        if tid is not None:
            nodes_by_territory[tid].append(nk)

    def _dist(nk):
        return dist_from_outfall.get(nk, 0.0)

    for tid, nk_list in nodes_by_territory.items():
        prefix = of_prefix.get(tid, f"O{tid}")
        juncs  = sorted([n for n in nk_list if node_type.get(n) == 'junction'],
                         key=_dist, reverse=True)
        sags   = sorted([n for n in nk_list if node_type.get(n) == 'sag'],
                         key=_dist, reverse=True)
        ridges = sorted([n for n in nk_list if node_type.get(n) == 'ridge'],
                         key=_dist, reverse=True)
        for i, nk in enumerate(juncs,  1): node_names[nk] = f"{prefix}-J{i}"
        for i, nk in enumerate(sags,   1): node_names[nk] = f"{prefix}-S{i}"
        for i, nk in enumerate(ridges, 1): node_names[nk] = f"{prefix}-R{i}"

    # ── Phase 4: per-channel inverts and names ────────────────────────────────
    of_inv_map = of_inverts or {}
    temp_info  = []   # one entry per seg in assigned_segs

    for seg in assigned_segs:
        tid    = seg.get('territory')
        pruned = pruned_by_territory.get(tid, set()) if tid else set()
        pts    = seg['pts']
        if tid is None or len(pts) < 2:
            temp_info.append(None);  continue
        nk_s = round_node(pts[0][0], pts[0][1])
        nk_e = round_node(pts[-1][0], pts[-1][1])
        if nk_s in pruned or nk_e in pruned:
            temp_info.append(None);  continue

        inv_dict = inverts_by_territory.get(tid, {})
        inv_s    = inv_dict.get(nk_s, pts[0][2]  - min_cover)
        inv_e    = inv_dict.get(nk_e, pts[-1][2] - min_cover)

        if inv_s >= inv_e:
            up_nk, dn_nk = nk_s, nk_e
            up_pt, dn_pt = pts[0], pts[-1]
            up_inv = inv_s
        else:
            up_nk, dn_nk = nk_e, nk_s
            up_pt, dn_pt = pts[-1], pts[0]
            up_inv = inv_e

        length = _seg_len(pts)

        # Channel's OWN downstream invert
        if dn_nk in outfall_nk_to_id:
            target_dn = of_inv_map.get(outfall_nk_to_id[dn_nk], dn_pt[2] - min_cover)
        else:
            target_dn = dn_pt[2] - min_cover

        if length > 1e-3:
            s_rec  = (up_inv - target_dn) / length
            s_pipe = max(s_rec, min_slope)
            own_dn_inv = up_inv - s_pipe * length
        else:
            own_dn_inv = up_inv

        temp_info.append({
            'tid':     tid,
            'up_nk':  up_nk, 'dn_nk':  dn_nk,
            'up_inv': up_inv, 'dn_inv': own_dn_inv,
            'up_gnd': up_pt[2], 'dn_gnd': dn_pt[2],
            'length': length,
            'dist_up': dist_from_outfall.get(up_nk, 0.0),
        })

    # Channel naming: sort by dist_up DESC within territory → C1 = farthest
    by_territory_segs = defaultdict(list)
    for idx, info in enumerate(temp_info):
        if info is not None:
            by_territory_segs[info['tid']].append((info['dist_up'], idx))

    seg_name_map = {}
    for tid, items in by_territory_segs.items():
        prefix = of_prefix.get(tid, f"O{tid}")
        for i, (_, idx) in enumerate(sorted(items, key=lambda x: x[0], reverse=True), 1):
            seg_name_map[idx] = f"{prefix}-C{i}"

    # Assemble final parallel lists
    seg_names   = []
    seg_inv_up  = [];  seg_inv_dn  = []
    seg_gnd_up  = [];  seg_gnd_dn  = []
    seg_lengths = []
    seg_node_up = [];  seg_node_dn = []

    for idx, info in enumerate(temp_info):
        if info is None:
            seg_names.append(None)
            seg_inv_up.append(None);  seg_inv_dn.append(None)
            seg_gnd_up.append(None);  seg_gnd_dn.append(None)
            seg_lengths.append(None)
            seg_node_up.append(None); seg_node_dn.append(None)
        else:
            seg_names.append(seg_name_map.get(idx))
            seg_inv_up.append(info['up_inv']); seg_inv_dn.append(info['dn_inv'])
            seg_gnd_up.append(info['up_gnd']); seg_gnd_dn.append(info['dn_gnd'])
            seg_lengths.append(info['length'])
            seg_node_up.append(node_names.get(info['up_nk'], ''))
            seg_node_dn.append(node_names.get(info['dn_nk'], ''))

    return {
        'node_names':    node_names,
        'node_type':     node_type,
        'node_ground':   node_ground,
        'node_invert':   node_invert,
        'node_territory': node_territory,
        'node_xy':       node_xy,
        'active_nks':    active_nks,
        'orphan_nks':    orphan_nks,
        'all_nk_xy':     all_nk_xy,
        'seg_names':     seg_names,
        'seg_inv_up':    seg_inv_up,
        'seg_inv_dn':    seg_inv_dn,
        'seg_gnd_up':    seg_gnd_up,
        'seg_gnd_dn':    seg_gnd_dn,
        'seg_length':    seg_lengths,
        'seg_node_up':   seg_node_up,
        'seg_node_dn':   seg_node_dn,
    }


# ── SHP export ────────────────────────────────────────────────────────────────

def write_shp(assigned_segs, graphs, inverts_by_territory, pruned_by_territory,
              outfall_pts, catchments, out_dir,
              of_inverts=None, min_slope=0.0005, min_cover=1.0,
              crs="EPSG:32640"):
    """Write all shapefiles: swnetwork, nodes, orphan_channels, orphan_nodes,
    catchments, sw_inlets, sw_ridges."""
    os.makedirs(out_dir, exist_ok=True)

    na = _build_network_attrs(
        assigned_segs, inverts_by_territory, pruned_by_territory,
        outfall_pts, of_inverts, min_cover, min_slope)

    # ── swnetwork.shp (assigned channels with full attributes) ───────────────
    rows_assigned = []
    rows_orphan   = []

    for seg_idx, seg in enumerate(assigned_segs):
        tid  = seg['territory']
        geom = seg['geom']
        if geom is None or geom.is_empty:
            continue

        if tid is None:
            rows_orphan.append({
                'geometry': geom, 'territory': -1,
                'status': 'DISCONNECTED_ORPHAN',
                'name': None, 'node_up': None, 'node_dn': None,
                'inv_up': None, 'inv_dn': None,
                'gnd_up': None, 'gnd_dn': None, 'length_m': None,
            })
            continue

        inverts = inverts_by_territory.get(tid, {})
        pruned  = pruned_by_territory.get(tid, set())
        pts     = seg['pts']
        nk_s    = round_node(pts[0][0], pts[0][1])
        nk_e    = round_node(pts[-1][0], pts[-1][1])

        if nk_s in pruned or nk_e in pruned:
            rows_orphan.append({
                'geometry': geom, 'territory': tid,
                'status': 'DESIGN_ORPHAN',
                'name': None, 'node_up': None, 'node_dn': None,
                'inv_up': None, 'inv_dn': None,
                'gnd_up': None, 'gnd_dn': None, 'length_m': None,
            })
        else:
            rows_assigned.append({
                'geometry':  geom,
                'territory': tid,
                'status':    'ASSIGNED',
                'name':      na['seg_names'][seg_idx],
                'node_up':   na['seg_node_up'][seg_idx],
                'node_dn':   na['seg_node_dn'][seg_idx],
                'inv_up':    (round(na['seg_inv_up'][seg_idx], 3)
                              if na['seg_inv_up'][seg_idx] is not None else None),
                'inv_dn':    (round(na['seg_inv_dn'][seg_idx], 3)
                              if na['seg_inv_dn'][seg_idx] is not None else None),
                'gnd_up':    (round(na['seg_gnd_up'][seg_idx], 3)
                              if na['seg_gnd_up'][seg_idx] is not None else None),
                'gnd_dn':    (round(na['seg_gnd_dn'][seg_idx], 3)
                              if na['seg_gnd_dn'][seg_idx] is not None else None),
                'length_m':  (round(na['seg_length'][seg_idx], 2)
                              if na['seg_length'][seg_idx] is not None else None),
            })

    _safe_write_shp(rows_assigned, crs, f"{out_dir}/swnetwork.shp")
    _safe_write_shp(rows_orphan,   crs, f"{out_dir}/orphan_channels.shp")

    # ── nodes.shp (non-orphan nodes with names) ──────────────────────────────
    rows_nodes = []
    for nk in na['active_nks']:
        xy  = na['node_xy'].get(nk, nk)
        gnd = na['node_ground'].get(nk)
        inv = na['node_invert'].get(nk)
        dep = (round(gnd - inv, 3) if gnd is not None and inv is not None else None)
        rows_nodes.append({
            'geometry':  Point(xy[0], xy[1]),
            'name':      na['node_names'].get(nk, ''),
            'type':      na['node_type'].get(nk, 'junction'),
            'territory': na['node_territory'].get(nk, -1),
            'ground':    round(gnd, 3) if gnd is not None else None,
            'invert':    round(inv, 3) if inv is not None else None,
            'depth':     dep,
        })
    _safe_write_shp(rows_nodes, crs, f"{out_dir}/nodes.shp")

    # ── orphan_nodes.shp ─────────────────────────────────────────────────────
    rows_orp_nodes = []
    for nk in na['orphan_nks']:
        xy  = na['all_nk_xy'].get(nk, nk)
        gnd = na['node_ground'].get(nk)
        rows_orp_nodes.append({
            'geometry': Point(xy[0], xy[1]),
            'ground':   round(gnd, 3) if gnd is not None else None,
        })
    _safe_write_shp(rows_orp_nodes, crs, f"{out_dir}/orphan_nodes.shp")

    # ── catchments.shp ───────────────────────────────────────────────────────
    catch_rows = [{'geometry': poly, 'territory': tid}
                  for tid, poly in catchments.items() if poly is not None]
    _safe_write_shp(catch_rows, crs, f"{out_dir}/catchments.shp")

    # ── sw_inlets.shp (sag nodes) ─────────────────────────────────────────────
    seen_sag = set()
    rows_inlets = []
    for seg in assigned_segs:
        s_pts = seg.get('pts', [])
        if len(s_pts) < 2:
            continue
        tid = seg.get('territory')
        for key, pt in [('start_node_type', s_pts[0]), ('end_node_type', s_pts[-1])]:
            if seg.get(key) != 'sag':
                continue
            nk = round_node(pt[0], pt[1])
            if nk in seen_sag:
                continue
            seen_sag.add(nk)
            rows_inlets.append({
                'geometry':  Point(pt[0], pt[1]),
                'name':      na['node_names'].get(nk, ''),
                'ground':    round(float(pt[2]), 3),
                'territory': tid if tid is not None else -1,
            })
    _safe_write_shp(rows_inlets, crs, f"{out_dir}/sw_inlets.shp")

    # ── sw_ridges.shp ─────────────────────────────────────────────────────────
    seen_ridge = set()
    rows_ridges = []
    for seg in assigned_segs:
        s_pts = seg.get('pts', [])
        if len(s_pts) < 2:
            continue
        tid = seg.get('territory')
        for key, pt in [('start_node_type', s_pts[0]), ('end_node_type', s_pts[-1])]:
            if seg.get(key) != 'ridge':
                continue
            nk = round_node(pt[0], pt[1])
            if nk in seen_ridge:
                continue
            seen_ridge.add(nk)
            rows_ridges.append({
                'geometry':  Point(pt[0], pt[1]),
                'name':      na['node_names'].get(nk, ''),
                'ground':    round(float(pt[2]), 3),
                'territory': tid if tid is not None else -1,
            })
    _safe_write_shp(rows_ridges, crs, f"{out_dir}/sw_ridges.shp")

    print(f"  SHP: {len(rows_assigned)} channels, {len(rows_nodes)} nodes, "
          f"{len(rows_orp_nodes)} orphan nodes, {len(rows_orphan)} orphan channels, "
          f"{len(rows_inlets)} sag inlets, {len(rows_ridges)} ridge nodes written")


# ── DXF export ────────────────────────────────────────────────────────────────

def write_dxf(assigned_segs, inverts_by_territory, pruned_by_territory,
              outfall_pts, of_grounds, of_inverts, out_path,
              min_cover=1.0, min_slope=0.0005):
    """Write colored DXF: channels, flow ticks, outfall symbols,
    junction circles, and node label stacks (name / G / I / D)."""

    na = _build_network_attrs(
        assigned_segs, inverts_by_territory, pruned_by_territory,
        outfall_pts, of_inverts, min_cover, min_slope)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    all_tids = sorted(set(s['territory'] for s in assigned_segs if s['territory']))
    for tid in all_tids:
        doc.layers.new(f"SW-DRAIN-SUB-{tid:02d}", dxfattribs={'color': _aci(tid)})
    doc.layers.new("SW-DRAIN-ORPHAN",  dxfattribs={'color': 8})
    doc.layers.new("SW-OUTLETS",       dxfattribs={'color': 2})
    doc.layers.new("SW-JUNCTIONS",     dxfattribs={'color': 7})
    doc.layers.new("SW-DRAIN-LABELS",  dxfattribs={'color': 3})
    doc.layers.new("SW-INLETS",        dxfattribs={'color': 3})
    doc.layers.new("SW-RIDGES",        dxfattribs={'color': 6})

    outfall_node_keys = set(round_node(ox, oy) for _, ox, oy in outfall_pts)
    labelled_nodes    = set()

    # Pre-compute endpoints of fully-assigned (non-pruned) segments so that
    # orphan-only nodes don't claim the label slot before the assigned loop.
    assigned_endpoints = set()
    for _seg in assigned_segs:
        _tid = _seg.get('territory')
        if _tid is None:
            continue
        _spts = _seg.get('pts', [])
        if len(_spts) < 2:
            continue
        _pruned = pruned_by_territory.get(_tid, set())
        _nk_s = round_node(_spts[0][0], _spts[0][1])
        _nk_e = round_node(_spts[-1][0], _spts[-1][1])
        if _nk_s not in _pruned and _nk_e not in _pruned:
            assigned_endpoints.add(_nk_s)
            assigned_endpoints.add(_nk_e)

    # ── Helper: draw the name / G / I / D label stack ────────────────────────
    def _node_label(cx, cy, name, ground, inv_val, sp=TEXT_HT * 1.3):
        lx = cx + JUNC_R * 1.5
        # name (top) — white, slightly larger
        if name:
            msp.add_text(name, dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 7, 'insert': (lx, cy + sp * 2.0)})
        # G
        msp.add_text(f"G:{ground:.2f}", dxfattribs={
            'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
            'color': 7, 'insert': (lx, cy + sp)})
        if inv_val is not None:
            # I
            msp.add_text(f"I:{inv_val:.2f}", dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 4, 'insert': (lx, cy)})
            # D
            msp.add_text(f"D:{ground - inv_val:.2f}", dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 6, 'insert': (lx, cy - sp)})

    # ── Channels + flow ticks + junction labels ───────────────────────────────
    for seg_idx, seg in enumerate(assigned_segs):
        tid  = seg['territory']
        geom = seg['geom']
        if geom is None or geom.is_empty:
            continue
        pts_2d = [(c[0], c[1]) for c in geom.coords]
        if len(pts_2d) < 2:
            continue

        if tid is None:
            layer     = "SW-DRAIN-ORPHAN"
            is_orphan = True
        else:
            pruned = pruned_by_territory.get(tid, set())
            s_pts  = seg.get('pts', [])
            if len(s_pts) < 2:
                continue
            nk_s = round_node(s_pts[0][0], s_pts[0][1])
            nk_e = round_node(s_pts[-1][0], s_pts[-1][1])
            if nk_s in pruned or nk_e in pruned:
                layer     = "SW-DRAIN-ORPHAN"
                is_orphan = True
            else:
                layer     = f"SW-DRAIN-SUB-{tid:02d}"
                is_orphan = False

        msp.add_lwpolyline(pts_2d, dxfattribs={'layer': layer})

        # Flow tick on assigned channels
        if not is_orphan:
            s_pts = seg.get('pts', [])
            if s_pts and len(s_pts) >= 2:
                inverts_t = inverts_by_territory.get(tid, {})
                nk_s_t = round_node(s_pts[0][0],  s_pts[0][1])
                nk_e_t = round_node(s_pts[-1][0], s_pts[-1][1])
                inv_s_t = inverts_t.get(nk_s_t)
                inv_e_t = inverts_t.get(nk_e_t)
                seg_len2d = _seg_len(s_pts)
                if inv_s_t is not None and inv_e_t is not None:
                    tick_rev  = inv_e_t > inv_s_t
                    slope_lbl = (f"{abs(inv_s_t-inv_e_t)/seg_len2d*100:.2f}%"
                                 if seg_len2d > 0.01 else None)
                else:
                    tick_rev  = s_pts[-1][2] > s_pts[0][2]
                    slope_lbl = None
                len_lbl  = f"{seg_len2d:.1f}m" if seg_len2d > 0.01 else None
                name_lbl = na['seg_names'][seg_idx]
                _add_flow_tick(msp, s_pts, layer, reverse=tick_rev,
                               name_label=name_lbl,
                               slope_label=slope_lbl,
                               length_label=len_lbl)

        # Orphan-only node: simple ground label (skip if also an assigned endpoint)
        if is_orphan:
            s_pts = seg.get('pts', [])
            for pt in [s_pts[0], s_pts[-1]]:
                nk = round_node(pt[0], pt[1])
                if nk in labelled_nodes or nk in assigned_endpoints:
                    continue
                labelled_nodes.add(nk)
                cx, cy = pt[0], pt[1]
                _add_circle(msp, cx, cy, JUNC_R, "SW-JUNCTIONS")
                msp.add_text(f"G:{pt[2]:.2f}", dxfattribs={
                    'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                    'color': 7, 'insert': (cx + JUNC_R * 1.5, cy + TEXT_HT * 0.3)})

        # Full label on assigned nodes (non-outfall, non-sag, non-ridge — those
        # are handled in their own dedicated sections below)
        if not is_orphan and tid is not None:
            inverts = inverts_by_territory.get(tid, {})
            s_pts   = seg.get('pts', [])
            for pt_idx, pt in enumerate([s_pts[0], s_pts[-1]]):
                nk = round_node(pt[0], pt[1])
                if nk in labelled_nodes:
                    continue
                labelled_nodes.add(nk)
                node_type_key = 'start_node_type' if pt_idx == 0 else 'end_node_type'
                skip_label = (nk in outfall_node_keys or
                              seg.get(node_type_key) in ('sag', 'ridge'))
                cx, cy = pt[0], pt[1]
                _add_circle(msp, cx, cy, JUNC_R, "SW-JUNCTIONS")
                if not skip_label:
                    _node_label(cx, cy,
                                na['node_names'].get(nk, ''),
                                pt[2],
                                inverts.get(nk))

    # ── Sag inlet symbols: double circle + name/G/I/D ────────────────────────
    sag_drawn = set()
    for seg in assigned_segs:
        s_pts = seg.get('pts', [])
        if len(s_pts) < 2:
            continue
        tid       = seg.get('territory')
        inverts_t = inverts_by_territory.get(tid, {}) if tid else {}
        for key, pt in [('start_node_type', s_pts[0]), ('end_node_type', s_pts[-1])]:
            if seg.get(key) != 'sag':
                continue
            nk = round_node(pt[0], pt[1])
            if nk in sag_drawn:
                continue
            sag_drawn.add(nk)
            cx, cy  = pt[0], pt[1]
            inv_val = inverts_t.get(nk)
            _add_circle(msp, cx, cy, JUNC_R,       "SW-INLETS")
            _add_circle(msp, cx, cy, JUNC_R * 2.0, "SW-INLETS")
            _node_label(cx, cy, na['node_names'].get(nk, ''), pt[2], inv_val,
                        sp=TEXT_HT * 1.3)

    # ── Ridge symbols: X cross + name/G/I/D ──────────────────────────────────
    ridge_drawn = set()
    for seg in assigned_segs:
        s_pts = seg.get('pts', [])
        if len(s_pts) < 2:
            continue
        tid       = seg.get('territory')
        inverts_t = inverts_by_territory.get(tid, {}) if tid else {}
        for key, pt in [('start_node_type', s_pts[0]), ('end_node_type', s_pts[-1])]:
            if seg.get(key) != 'ridge':
                continue
            nk = round_node(pt[0], pt[1])
            if nk in ridge_drawn:
                continue
            ridge_drawn.add(nk)
            cx, cy  = pt[0], pt[1]
            inv_val = inverts_t.get(nk)
            r = JUNC_R * 1.5
            msp.add_line((cx-r, cy-r), (cx+r, cy+r),
                         dxfattribs={'layer': "SW-RIDGES", 'color': 6})
            msp.add_line((cx+r, cy-r), (cx-r, cy+r),
                         dxfattribs={'layer': "SW-RIDGES", 'color': 6})
            _node_label(cx, cy, na['node_names'].get(nk, ''), pt[2], inv_val,
                        sp=TEXT_HT * 1.3)

    # ── Outfall symbols: double circle + stacked label ────────────────────────
    for of_id, ox, oy in outfall_pts:
        nk         = round_node(ox, oy)
        color      = _aci(of_id)
        lx         = ox + OF_R_OUTER * 1.2
        sp         = TEXT_HT * 1.3
        ground_val = of_grounds.get(of_id)
        inv_val    = of_inverts.get(of_id)
        name       = na['node_names'].get(nk, f"O{of_id}")

        _add_circle(msp, ox, oy, OF_R_INNER, "SW-OUTLETS")
        _add_circle(msp, ox, oy, OF_R_OUTER, "SW-OUTLETS")

        # Name (top) — territory colour, slightly larger
        msp.add_text(name, dxfattribs={
            'layer': "SW-OUTLETS", 'height': TEXT_HT * 1.5,
            'color': color, 'insert': (lx, oy + sp * 1.5)})
        if ground_val is not None:
            msp.add_text(f"G:{ground_val:.2f}", dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 7, 'insert': (lx, oy + sp * 0.2)})
        if inv_val is not None:
            msp.add_text(f"I:{inv_val:.2f}", dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 4, 'insert': (lx, oy - sp * 0.9)})
        if ground_val is not None and inv_val is not None:
            msp.add_text(f"D:{ground_val - inv_val:.2f}", dxfattribs={
                'layer': "SW-DRAIN-LABELS", 'height': TEXT_HT,
                'color': 6, 'insert': (lx, oy - sp * 2.0)})

    out_dir_dxf = os.path.dirname(out_path)
    if out_dir_dxf:
        os.makedirs(out_dir_dxf, exist_ok=True)
    doc.saveas(out_path)
    print(f"  DXF saved: {out_path}")


# ── PNG export ────────────────────────────────────────────────────────────────

def write_img(assigned_segs, catchments, outfall_pts, img_dir):
    """Write territories.png — road segments colored by outfall territory."""
    COLORS = ['red', 'gold', 'limegreen', 'cyan', 'royalblue', 'magenta',
              'orange', 'white', 'pink', 'lime']

    os.makedirs(img_dir, exist_ok=True)

    tid_list  = sorted(t for t in set(s['territory'] for s in assigned_segs) if t)
    color_map = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(tid_list)}

    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')

    for seg in assigned_segs:
        tid  = seg['territory']
        geom = seg['geom']
        if geom is None or geom.is_empty:
            continue
        try:
            xs, ys = geom.xy
        except Exception:
            continue
        ax.plot(xs, ys, color=color_map.get(tid, 'gray'), lw=0.6, alpha=0.8)

    for of_id, ox, oy in outfall_pts:
        ax.plot(ox, oy, 'o', color='white', ms=8, zorder=5)
        ax.annotate(f"OF{of_id}", (ox, oy), color='white', fontsize=7,
                    xytext=(4, 4), textcoords='offset points')

    patches = [mpatches.Patch(color=color_map[t], label=f"OF{t}") for t in tid_list]
    ax.legend(handles=patches, loc='lower left', fontsize=7,
              facecolor='#1a1a2e', labelcolor='white')
    ax.set_title("SW Drainage — Territory Assignment (v5)", color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.savefig(f"{img_dir}/territories.png", dpi=150)
    plt.close()
    print(f"  PNG saved: {img_dir}/territories.png")

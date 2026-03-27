"""roads.py — Road loading, noding, outfall splits, ridge-splitting, catchment assignment."""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.ops import substring as shapely_substring
from scipy.spatial import KDTree
from collections import Counter


# ── Ridge detection ──────────────────────────────────────────────────────────

def detect_ridges(pts, ridge_rise=0.05):
    """Return indices (into pts) that are local elevation maxima above both neighbours."""
    return [i for i in range(1, len(pts) - 1)
            if pts[i][2] > pts[i-1][2] + ridge_rise
            and pts[i][2] > pts[i+1][2] + ridge_rise]


def ridge_split_pts(pts, ridge_rise=0.05):
    """
    Split a list of (x, y, elev) points at every ridge.
    Returns list of sub-lists, each with >= 2 points.
    """
    ri = detect_ridges(pts, ridge_rise)
    if not ri:
        return [pts]
    bounds = [0] + ri + [len(pts) - 1]
    return [pts[bounds[k]: bounds[k+1] + 1]
            for k in range(len(bounds) - 1)
            if bounds[k+1] - bounds[k] >= 1]


def sample_line(line, elev_at, spacing=2.0):
    """
    Sample (x, y, elev) points along a LineString at `spacing` intervals.
    elev_at: callable (x, y) -> float | None
    Returns list of (x, y, float) or [] if fewer than 2 valid points.
    """
    length = line.length
    sx, sy = line.coords[0][0], line.coords[0][1]
    ex, ey = line.coords[-1][0], line.coords[-1][1]
    pts = []
    e = elev_at(sx, sy)
    if e is not None:
        pts.append((sx, sy, e))
    d = spacing
    while d < length - spacing * 0.1:
        p = line.interpolate(d)
        e = elev_at(p.x, p.y)
        if e is not None:
            pts.append((p.x, p.y, e))
        d += spacing
    e = elev_at(ex, ey)
    if e is not None:
        pts.append((ex, ey, e))
    return pts if len(pts) >= 2 else []


# ── Noding ───────────────────────────────────────────────────────────────────

def node_roads(roads_gdf, min_seg_len=0.5):
    """
    Node all road geometries via unary_union to enforce shared endpoints.
    Returns list of LineString with length >= min_seg_len.
    """
    raw = []
    for g in roads_gdf.geometry:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            raw.append(g)
        elif g.geom_type == "MultiLineString":
            raw.extend(list(g.geoms))

    noded = []
    def _collect(geom):
        if geom.geom_type == "LineString":
            if geom.length >= min_seg_len:
                noded.append(geom)
        else:
            for g in geom.geoms:
                _collect(g)

    _collect(unary_union(raw))
    return noded


def split_at_outfalls(noded_segs, outfall_pts, min_part=0.5):
    """
    For each outfall, project it onto the nearest segment and split there.
    outfall_pts: list of (of_id, x, y)
    Returns updated list of LineString segments.
    """
    segs = list(noded_segs)
    for of_id, ox, oy in outfall_pts:
        of_pt    = Point(ox, oy)
        best_idx = min(range(len(segs)), key=lambda i: segs[i].distance(of_pt))
        best_seg = segs[best_idx]
        proj_d   = best_seg.project(of_pt)

        if proj_d < min_part or proj_d > best_seg.length - min_part:
            continue   # already at endpoint

        try:
            p1 = shapely_substring(best_seg, 0, proj_d)
            p2 = shapely_substring(best_seg, proj_d, best_seg.length)
            if p1.length >= min_part and p2.length >= min_part:
                segs[best_idx] = p1
                segs.insert(best_idx + 1, p2)
        except Exception:
            pass
    return segs


def ridge_split(noded_segs, elev_at, spacing=2.0, ridge_rise=0.05):
    """
    Sample elevation along each segment, split at ridges.
    Returns list of (pts_list, LineString) tuples where pts_list is [(x,y,elev)...].
    """
    result = []
    for seg in noded_segs:
        pts = sample_line(seg, elev_at, spacing)
        if len(pts) < 2:
            continue
        for sub_pts in ridge_split_pts(pts, ridge_rise):
            if len(sub_pts) < 2:
                continue
            coords = [(p[0], p[1]) for p in sub_pts]
            result.append((sub_pts, LineString(coords)))
    return result


# ── Outfall snapping ─────────────────────────────────────────────────────────

def _split_seg_at_proj(pts, geom, proj_d, proj_xy, proj_elev, min_len=0.5):
    """
    Split a (pts_list, geom) at distance proj_d along geom.
    Inserts a new point (proj_xy[0], proj_xy[1], proj_elev) at the split.

    Returns [(pts1, geom1), (pts2, geom2)] or [(pts, geom)] if either piece
    would be shorter than min_len or the split is at/beyond an endpoint.
    """
    total_L = geom.length
    if proj_d < min_len or proj_d > total_L - min_len:
        return [(pts, geom)]   # at or beyond endpoint — no split needed

    try:
        geom1 = shapely_substring(geom, 0,       proj_d)
        geom2 = shapely_substring(geom, proj_d,  total_L)
    except Exception:
        return [(pts, geom)]

    if geom1.length < min_len or geom2.length < min_len:
        return [(pts, geom)]

    px, py   = proj_xy
    new_pt   = (px, py, proj_elev)

    # Compute cumulative 2-D distances along pts to find insertion index
    cum = [0.0]
    for k in range(1, len(pts)):
        cum.append(cum[-1] + np.hypot(pts[k][0] - pts[k-1][0],
                                      pts[k][1] - pts[k-1][1]))

    # First index whose cumulative distance >= proj_d
    insert_at = len(pts)
    for k, cd in enumerate(cum):
        if cd >= proj_d - 1e-6:
            insert_at = k
            break

    pts1 = list(pts[:insert_at]) + [new_pt]
    pts2 = [new_pt] + list(pts[insert_at:])

    if len(pts1) < 2 or len(pts2) < 2:
        return [(pts, geom)]

    return [(pts1, geom1), (pts2, geom2)]


def snap_outfalls_to_road_graph(of_pts, seg_tuples, snap_r=30.0, elev_at=None):
    """
    Snap each outfall to a node on the road network.

    Priority order
    --------------
    1. On-road  (outfall within ROAD_ON_TOL of any road LINE):
       Project onto the nearest road, split that segment at the projection
       point to create a real junction node there.  Snap distance ~ 0 m.
    2. Off-road, local minimum within snap_r:
       Snap to the nearest road local-elevation-minimum node.
    3. Off-road fallback:
       Snap to the lowest-elevation node within snap_r.
    4. Nothing within snap_r: keep original position.

    of_pts     : list of (of_id, x, y)
    seg_tuples : list of (pts_list, geom)  — geom may be None
    snap_r     : float, search radius for off-road snapping (m)
    elev_at    : callable (x, y) -> float | None  (DEM sampler, optional)

    Returns (snapped_of_pts, updated_seg_tuples).
    seg_tuples is updated in-place with any split segments.
    """
    ROAD_ON_TOL = 2.0   # m — outfall is "on" a road if within this distance

    # Working copy of seg_tuples (we may insert split pieces)
    seg_list = list(seg_tuples)

    # ── Build node map for off-road snapping ──────────────────────────────────
    def _rebuild_nodes():
        ne, adj = {}, {}
        for p, _ in seg_list:
            if len(p) < 2:
                continue
            ks = (round(p[0][0], 2), round(p[0][1], 2))
            ke = (round(p[-1][0], 2), round(p[-1][1], 2))
            ne.setdefault(ks, p[0][2]);  ne.setdefault(ke, p[-1][2])
            adj.setdefault(ks, set()).add(ke)
            adj.setdefault(ke, set()).add(ks)
        return ne, adj

    node_elev, adjacency = _rebuild_nodes()
    if not node_elev:
        return list(of_pts), seg_list

    snapped = []

    for of_id, ox, oy in of_pts:
        of_pt = Point(ox, oy)

        # ── Mode 1: outfall is ON a road line ─────────────────────────────────
        min_line_dist = float('inf')
        best_idx      = None
        best_proj_d   = None

        for i, (pts, geom) in enumerate(seg_list):
            if geom is None or geom.is_empty or len(pts) < 2:
                continue
            d = geom.distance(of_pt)
            if d < min_line_dist:
                min_line_dist = d
                best_idx      = i
                best_proj_d   = geom.project(of_pt)

        if min_line_dist <= ROAD_ON_TOL and best_idx is not None:
            pts_orig, geom_orig = seg_list[best_idx]
            proj_pt = geom_orig.interpolate(best_proj_d)
            px, py  = proj_pt.x, proj_pt.y

            # Elevation at projected point
            pe = None
            if elev_at is not None:
                pe = elev_at(px, py)
            if pe is None:
                # Linear interpolation between segment endpoints
                L = geom_orig.length
                pe = float(np.interp(best_proj_d, [0.0, L],
                                     [pts_orig[0][2], pts_orig[-1][2]]))

            pieces = _split_seg_at_proj(pts_orig, geom_orig,
                                        best_proj_d, (px, py), pe)
            seg_list[best_idx:best_idx + 1] = pieces   # replace with 1 or 2 pieces

            dist = float(np.hypot(px - ox, py - oy))
            print(f"    OF{of_id}: on-road {dist:.1f}m  "
                  f"({ox:.1f},{oy:.1f}) -> ({px:.1f},{py:.1f})  "
                  f"elev={pe:.2f}")
            snapped.append((of_id, px, py))
            continue

        # ── Mode 2 / 3: off-road — search for nodes within snap_r ─────────────
        nodes   = list(node_elev.keys())
        node_xy = np.array([[n[0], n[1]] for n in nodes])
        kd      = KDTree(node_xy)

        local_mins = {n for n, e in node_elev.items()
                      if adjacency.get(n)
                      and all(node_elev.get(nb, e) > e
                              for nb in adjacency[n])}

        idxs = kd.query_ball_point([ox, oy], snap_r)
        if not idxs:
            snapped.append((of_id, ox, oy))
            continue

        candidates      = [nodes[i] for i in idxs]
        local_min_cands = [n for n in candidates if n in local_mins]
        best = (min(local_min_cands, key=lambda n: (n[0]-ox)**2 + (n[1]-oy)**2)
                if local_min_cands
                else min(candidates, key=lambda n: node_elev[n]))

        sx, sy = best
        dist = float(np.hypot(sx - ox, sy - oy))
        print(f"    OF{of_id}: snapped {dist:.1f}m  "
              f"({ox:.1f},{oy:.1f}) -> ({sx:.1f},{sy:.1f})  "
              f"elev={node_elev[best]:.2f}")
        snapped.append((of_id, sx, sy))

    return snapped, seg_list


# ── Catchment assignment ─────────────────────────────────────────────────────

def assign_to_catchments(seg_tuples, catchments):
    """
    Assign each (pts, LineString) to a territory by spatial intersection.
    Segments crossing a catchment boundary are split at the boundary.
    pts arrays are sliced to match each piece's actual geometry.

    catchments: dict { of_id: shapely.Polygon | None }

    Returns list of dicts:
        { 'pts': [...], 'geom': LineString, 'territory': of_id | None }
    """
    of_ids    = [k for k, v in catchments.items() if v is not None]
    cat_polys = [catchments[k] for k in of_ids]

    assigned = []
    for pts, geom in seg_tuples:
        matched = False
        for of_id, poly in zip(of_ids, cat_polys):
            inter = geom.intersection(poly)
            if inter.is_empty:
                continue

            # Fully inside this catchment
            if abs(inter.length - geom.length) < 0.01:
                assigned.append({'pts': pts, 'geom': geom, 'territory': of_id})
                matched = True
                break

            # Crosses boundary — split pieces
            if inter.geom_type in ("LineString", "MultiLineString"):
                pieces = (list(inter.geoms)
                          if inter.geom_type == "MultiLineString"
                          else [inter])
                for piece in pieces:
                    if piece.length < 0.5:
                        continue
                    # Slice pts to match the piece endpoints
                    piece_pts = _slice_pts_to_piece(pts, piece)
                    assigned.append({'pts': piece_pts, 'geom': piece, 'territory': of_id})
                matched = True

        if not matched:
            assigned.append({'pts': pts, 'geom': geom, 'territory': None})

    return assigned


def assign_majority(seg_tuples, catchments):
    """
    Assign each whole (pts, LineString) to its dominant catchment territory.
    Never splits roads at boundaries. Majority of length determines territory.

    catchments: dict { of_id: shapely.Polygon | None }

    Returns list of dicts:
        { 'pts': [...], 'geom': LineString, 'territory': of_id | None,
          'blacklist': set() }
    The 'blacklist' set records territories that have already rejected
    this channel — used by pool_reassignment_loop in later pipeline steps.
    """
    of_ids    = [k for k, v in catchments.items() if v is not None]
    cat_polys = [catchments[k] for k in of_ids]

    assigned = []
    for pts, geom in seg_tuples:
        best_tid    = None
        best_length = 0.0

        for of_id, poly in zip(of_ids, cat_polys):
            try:
                inter = geom.intersection(poly)
            except Exception:
                continue
            if inter.is_empty:
                continue
            L = inter.length if hasattr(inter, 'length') else 0.0
            if L > best_length:
                best_length = L
                best_tid    = of_id

        assigned.append({
            'pts':       pts,
            'geom':      geom,
            'territory': best_tid,
            'blacklist': set(),
        })

    return assigned


def reassign_boundary_roads(assigned, graphs=None):
    """
    Convergence loop: reassign roads whose both endpoints connect only to
    nodes of a different territory.

    For each road assigned to territory A: if both endpoints have NO other
    roads from territory A touching them (only roads from territory B or other),
    reassign to the most common other territory and blacklist A.

    assigned : list of seg dicts — modified in-place
    graphs   : reserved for future use, currently ignored
    Returns the modified assigned list.
    """

    def _ep(seg):
        pts = seg['pts']
        s = (round(pts[0][0], 2), round(pts[0][1], 2))
        e = (round(pts[-1][0], 2), round(pts[-1][1], 2))
        return s, e

    changed = True
    while changed:
        changed = False

        for seg in assigned:
            tid = seg['territory']
            if tid is None:
                continue
            ep_s, ep_e = _ep(seg)

            ep_s_counter = Counter()
            ep_e_counter = Counter()
            for other in assigned:  # O(n²) per pass — acceptable for ~1000 segments
                if other is seg:
                    continue
                o_tid = other['territory']
                if o_tid is None:
                    continue
                os, oe = _ep(other)
                if os == ep_s or oe == ep_s:
                    ep_s_counter[o_tid] += 1
                if os == ep_e or oe == ep_e:
                    ep_e_counter[o_tid] += 1

            # Both endpoints have NO other seg from current territory
            # (Counter returns 0 for absent keys, so missing tid is correctly treated as 0)
            if ep_s_counter[tid] == 0 and ep_e_counter[tid] == 0:
                # Collect territories from other segs at both endpoints
                candidates = (set(ep_s_counter.keys()) | set(ep_e_counter.keys())) \
                             - seg['blacklist'] - {tid}
                if not candidates:
                    continue
                new_tid = min(candidates)   # deterministic
                seg['blacklist'].add(tid)
                seg['territory'] = new_tid
                changed = True

    return assigned


def _slice_pts_to_piece(pts, piece):
    """
    Return the subset of `pts` that best covers `piece` geometry.
    Finds the pts indices closest to piece.coords[0] (start) and
    piece.coords[-1] (end), then returns pts[i_start : i_end+1].
    Falls back to the full pts list if fewer than 2 pts are found.
    """
    if len(pts) < 2:
        return pts

    start_xy = piece.coords[0]
    end_xy   = piece.coords[-1]

    def nearest_idx(xy):
        dists = [(p[0]-xy[0])**2 + (p[1]-xy[1])**2 for p in pts]
        return dists.index(min(dists))

    i_start = nearest_idx(start_xy)
    i_end   = nearest_idx(end_xy)

    if i_start > i_end:
        i_start, i_end = i_end, i_start

    sliced = pts[i_start: i_end + 1]
    return sliced if len(sliced) >= 2 else pts

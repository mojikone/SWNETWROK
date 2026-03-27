"""outputs.py — SHP, DXF, PNG export for v5 results."""
import os
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
ARROW_LEN  = TEXT_HT * 2.5   # m — flow tick length


def _add_circle(msp, cx, cy, radius, layer):
    """Add a circle entity to modelspace."""
    msp.add_circle((cx, cy), radius, dxfattribs={'layer': layer})


def _add_flow_tick(msp, pts, layer, reverse=False, slope_label=None):
    """
    Add a flow-direction chevron at the segment midpoint.
    pts: list of (x, y, elev).
    reverse=False  -> arrow points pts[0] -> pts[-1]  (pts[0] is upstream)
    reverse=True   -> arrow points pts[-1] -> pts[0]  (pts[-1] is upstream)
    Draws two short lines at ±30° from the downstream direction.
    If slope_label is provided, writes it above the arrow aligned to pipe direction.
    """
    import math
    if len(pts) < 2:
        return
    if reverse:
        dx = pts[0][0] - pts[-1][0]
        dy = pts[0][1] - pts[-1][1]
    else:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
    length = np.hypot(dx, dy)
    if length < 0.01:
        return
    dx /= length
    dy /= length   # unit vector downstream

    _line = LineString([(p[0], p[1]) for p in pts])
    _mid  = _line.interpolate(0.5, normalized=True)
    mx, my = _mid.x, _mid.y

    angle = math.atan2(dy, dx)
    for sign in (+1, -1):
        a = angle + math.pi + sign * math.radians(30)
        ex = mx + ARROW_LEN * math.cos(a)
        ey = my + ARROW_LEN * math.sin(a)
        msp.add_line((mx, my), (ex, ey), dxfattribs={'layer': layer})

    if slope_label is not None:
        # Offset text perpendicular (left of downstream direction) by 1.5×TEXT_HT
        perp_x = mx - dy * TEXT_HT * 1.5
        perp_y = my + dx * TEXT_HT * 1.5
        # Keep rotation in (-90, 90] so text is never upside-down
        rot_deg = math.degrees(angle)
        if rot_deg > 90:
            rot_deg -= 180
        elif rot_deg < -90:
            rot_deg += 180
        msp.add_text(
            slope_label,
            dxfattribs={
                'layer':    "SW-DRAIN-LABELS",
                'height':   TEXT_HT,
                'color':    2,          # yellow
                'insert':   (perp_x, perp_y),
                'rotation': rot_deg,
            }
        )


def _safe_write_shp(rows, crs, path):
    """Write rows to SHP, skipping if empty to avoid geopandas schema error."""
    if not rows:
        print(f"  (skipped empty layer: {os.path.basename(path)})")
        return
    gpd.GeoDataFrame(rows, crs=crs).to_file(path)

def _aci(of_id):
    return ACI_COLORS[(of_id - 1) % len(ACI_COLORS)]


def write_shp(assigned_segs, graphs, inverts_by_territory, pruned_by_territory,
              outfall_pts, catchments, out_dir, crs="EPSG:32640"):
    """Write swnetwork.shp, catchments.shp, orphans.shp."""
    rows_assigned = []
    rows_orphan   = []

    for seg in assigned_segs:
        tid  = seg['territory']
        geom = seg['geom']
        if geom is None or geom.is_empty:
            continue

        if tid is None:
            rows_orphan.append({
                'geometry': geom, 'territory': -1,
                'status': 'DISCONNECTED_ORPHAN',
                'inv_up': None, 'inv_dn': None
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
                'inv_up': None, 'inv_dn': None
            })
        else:
            # Assign inv_up/inv_dn by comparing actual invert levels.
            # Ground-elevation comparison is wrong for BFS-oriented graphs where
            # a road can climb toward the outfall (pipe at MIN_SLOPE through hump).
            inv_s = inverts.get(nk_s)
            inv_e = inverts.get(nk_e)
            if inv_s is not None and inv_e is not None:
                if inv_s >= inv_e:
                    inv_up_val, inv_dn_val = inv_s, inv_e
                else:
                    inv_up_val, inv_dn_val = inv_e, inv_s
            elif inv_s is not None:
                inv_up_val, inv_dn_val = inv_s, None
            elif inv_e is not None:
                inv_up_val, inv_dn_val = inv_e, None
            else:
                inv_up_val, inv_dn_val = None, None
            rows_assigned.append({
                'geometry': geom, 'territory': tid,
                'status': 'ASSIGNED',
                'inv_up': inv_up_val,
                'inv_dn': inv_dn_val
            })

    os.makedirs(out_dir, exist_ok=True)

    _safe_write_shp(rows_assigned, crs, f"{out_dir}/swnetwork.shp")
    _safe_write_shp(rows_orphan,   crs, f"{out_dir}/orphans.shp")

    catch_rows = [{'geometry': poly, 'territory': tid}
                  for tid, poly in catchments.items() if poly is not None]
    _safe_write_shp(catch_rows, crs, f"{out_dir}/catchments.shp")

    print(f"  SHP: {len(rows_assigned)} assigned, {len(rows_orphan)} orphan segments written")


def write_dxf(assigned_segs, inverts_by_territory, pruned_by_territory,
              outfall_pts, of_grounds, of_inverts, out_path):
    """Write colored DXF: channels, flow ticks, outfall symbols,
    junction dots, and two-line node labels (including at outfall positions)."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create layers
    all_tids = sorted(set(s['territory'] for s in assigned_segs if s['territory']))
    for tid in all_tids:
        doc.layers.new(f"SW-DRAIN-SUB-{tid:02d}", dxfattribs={'color': _aci(tid)})
    doc.layers.new("SW-DRAIN-ORPHAN",  dxfattribs={'color': 8})
    doc.layers.new("SW-OUTLETS",       dxfattribs={'color': 2})    # yellow
    doc.layers.new("SW-JUNCTIONS",     dxfattribs={'color': 7})    # white
    doc.layers.new("SW-DRAIN-LABELS",  dxfattribs={'color': 3})    # green

    # Track nodes already labelled to avoid duplicates
    labelled_nodes = set()

    # Draw channels + flow ticks + junction dots + labels
    for seg in assigned_segs:
        tid  = seg['territory']
        geom = seg['geom']
        if geom is None or geom.is_empty:
            continue
        pts_2d = [(c[0], c[1]) for c in geom.coords]
        if len(pts_2d) < 2:
            continue

        if tid is None:
            layer    = "SW-DRAIN-ORPHAN"
            is_orphan = True
        else:
            pruned = pruned_by_territory.get(tid, set())
            s_pts  = seg.get('pts', [])
            if len(s_pts) < 2:
                continue  # skip segments without valid pts
            nk_s   = round_node(s_pts[0][0], s_pts[0][1])
            nk_e   = round_node(s_pts[-1][0], s_pts[-1][1])
            if nk_s in pruned or nk_e in pruned:
                layer     = "SW-DRAIN-ORPHAN"
                is_orphan = True
            else:
                layer     = f"SW-DRAIN-SUB-{tid:02d}"
                is_orphan = False

        msp.add_lwpolyline(pts_2d, dxfattribs={'layer': layer})

        # Flow direction tick (assigned channels only)
        # Arrow points from higher invert (upstream) to lower invert (downstream).
        if not is_orphan:
            s_pts = seg.get('pts', [])
            if s_pts and len(s_pts) >= 2:
                inverts_t = inverts_by_territory.get(tid, {})
                nk_s_t = round_node(s_pts[0][0],  s_pts[0][1])
                nk_e_t = round_node(s_pts[-1][0], s_pts[-1][1])
                inv_s_t = inverts_t.get(nk_s_t)
                inv_e_t = inverts_t.get(nk_e_t)
                # reverse=True when pts[-1] has higher invert (pts[-1] is upstream)
                if inv_s_t is not None and inv_e_t is not None:
                    tick_rev  = inv_e_t > inv_s_t
                    seg_len   = sum(
                        np.hypot(s_pts[i+1][0] - s_pts[i][0],
                                 s_pts[i+1][1] - s_pts[i][1])
                        for i in range(len(s_pts) - 1)
                    )
                    if seg_len > 0.01:
                        slope_pct   = abs(inv_s_t - inv_e_t) / seg_len * 100
                        slope_label = f"{slope_pct:.2f}%"
                    else:
                        slope_label = None
                else:
                    # fallback: use ground elevation
                    tick_rev    = s_pts[-1][2] > s_pts[0][2]
                    slope_label = None
                _add_flow_tick(msp, s_pts, layer, reverse=tick_rev,
                               slope_label=slope_label)

        # Ground elevation label on orphan endpoints (no invert available)
        if is_orphan:
            s_pts = seg.get('pts', [])
            for pt in [s_pts[0], s_pts[-1]]:
                nk = round_node(pt[0], pt[1])
                if nk in labelled_nodes:
                    continue
                labelled_nodes.add(nk)
                cx, cy = pt[0], pt[1]
                _add_circle(msp, cx, cy, JUNC_R, "SW-JUNCTIONS")
                msp.add_text(
                    f"{pt[2]:.2f}",
                    dxfattribs={
                        'layer':  "SW-DRAIN-LABELS",
                        'height': TEXT_HT,
                        'color':  7,
                        'insert': (cx + JUNC_R * 1.5, cy + TEXT_HT * 0.3),
                    }
                )

        # Junction dots and labels on assigned nodes only (orphan segs not in any territory graph)
        if not is_orphan and tid is not None:
            inverts = inverts_by_territory.get(tid, {})
            s_pts   = seg.get('pts', [])
            for pt in [s_pts[0], s_pts[-1]]:
                nk = round_node(pt[0], pt[1])
                if nk in labelled_nodes:
                    continue
                labelled_nodes.add(nk)
                cx, cy = pt[0], pt[1]

                # Junction dot
                _add_circle(msp, cx, cy, JUNC_R, "SW-JUNCTIONS")

                ground_elev = pt[2]
                inv_val     = inverts.get(nk)
                lx          = cx + JUNC_R * 1.5
                sp          = TEXT_HT * 1.3   # line spacing

                # G: ground elevation (white)
                msp.add_text(
                    f"G:{ground_elev:.2f}",
                    dxfattribs={
                        'layer':  "SW-DRAIN-LABELS",
                        'height': TEXT_HT,
                        'color':  7,
                        'insert': (lx, cy + sp),
                    }
                )

                # I: invert elevation (cyan)
                if inv_val is not None:
                    msp.add_text(
                        f"I:{inv_val:.2f}",
                        dxfattribs={
                            'layer':  "SW-DRAIN-LABELS",
                            'height': TEXT_HT,
                            'color':  4,
                            'insert': (lx, cy),
                        }
                    )

                    # D: cover depth at junction (magenta)
                    cover = ground_elev - inv_val
                    msp.add_text(
                        f"D:{cover:.2f}",
                        dxfattribs={
                            'layer':  "SW-DRAIN-LABELS",
                            'height': TEXT_HT,
                            'color':  6,
                            'insert': (lx, cy - sp),
                        }
                    )

    # Outfall symbols: double circle + name label + ground/invert elevation labels
    for of_id, ox, oy in outfall_pts:
        color = _aci(of_id)
        _add_circle(msp, ox, oy, OF_R_INNER, "SW-OUTLETS")
        _add_circle(msp, ox, oy, OF_R_OUTER, "SW-OUTLETS")

        # "OFn" name label to the right of the outer circle
        msp.add_text(
            f"OF{of_id}",
            dxfattribs={
                'layer':  "SW-OUTLETS",
                'height': TEXT_HT * 1.5,
                'color':  color,
                'insert': (ox + OF_R_OUTER * 1.2, oy + TEXT_HT * 0.5),
            }
        )

        # Ground elevation label (white) — above the name
        ground_val = of_grounds.get(of_id)
        if ground_val is not None:
            msp.add_text(
                f"{ground_val:.2f}",
                dxfattribs={
                    'layer':  "SW-DRAIN-LABELS",
                    'height': TEXT_HT,
                    'color':  7,
                    'insert': (ox + OF_R_OUTER * 1.2, oy + TEXT_HT * 2.0),
                }
            )

        # Invert elevation label (cyan) — below the name
        inv_val = of_inverts.get(of_id)
        if inv_val is not None:
            msp.add_text(
                f"{inv_val:.2f}",
                dxfattribs={
                    'layer':  "SW-DRAIN-LABELS",
                    'height': TEXT_HT,
                    'color':  4,
                    'insert': (ox + OF_R_OUTER * 1.2, oy - TEXT_HT * 1.1),
                }
            )

    out_dir_dxf = os.path.dirname(out_path)
    if out_dir_dxf:
        os.makedirs(out_dir_dxf, exist_ok=True)
    doc.saveas(out_path)
    print(f"  DXF saved: {out_path}")


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
        c = color_map.get(tid, 'gray')
        ax.plot(xs, ys, color=c, lw=0.6, alpha=0.8)

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

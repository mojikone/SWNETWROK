"""dem.py — DEM bilinear sampling + D8 catchment delineation."""
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def make_sampler(dem_data, transform, nodata):
    """
    Returns a function  elev_at(x, y) -> float | None
    using bilinear interpolation on dem_data.
    dem_data: 2-D numpy array (rows × cols)
    transform: rasterio-style affine with .c, .a, .f, .e attributes
    nodata: value to treat as missing, or None
    """
    nr, nc = dem_data.shape
    data = dem_data.astype(np.float64).copy()
    if nodata is not None:
        data[np.abs(data - nodata) < 1e-6] = np.nan

    def elev_at(x, y):
        col_f = (x - transform.c) / transform.a
        row_f = (y - transform.f) / transform.e
        r0, c0 = int(np.floor(row_f)), int(np.floor(col_f))
        if r0 < 0 or c0 < 0 or r0 >= nr or c0 >= nc:
            return None
        r1 = min(r0 + 1, nr - 1)
        c1 = min(c0 + 1, nc - 1)
        dr = row_f - r0
        dc = col_f - c0
        v = (data[r0, c0] * (1 - dr) * (1 - dc) +
             data[r0, c1] * (1 - dr) * dc +
             data[r1, c0] * dr       * (1 - dc) +
             data[r1, c1] * dr       * dc)
        return float(v) if not np.isnan(v) else None

    return elev_at


def load_dem(tif_path, target_crs="EPSG:32640"):
    """
    Load DEM, reproject to target_crs in-memory.
    Returns (dem_data, transform, nodata).
    """
    with rasterio.open(tif_path) as src:
        if str(src.crs) == target_crs:
            dem_data   = src.read(1).astype(np.float64)
            transform  = src.transform
            nodata     = src.nodata
        else:
            dst_crs = target_crs
            t, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            dst = np.empty((h, w), dtype=np.float64)
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=t, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
            dem_data  = dst
            transform = t
            nodata    = src.nodata
    return dem_data, transform, nodata


def sample_elev(points_xy, tif_path, target_crs="EPSG:32640"):
    """
    Sample ground_elev for a list of (x, y) tuples.
    Returns dict { (x,y): float|None }.
    """
    dem_data, transform, nodata = load_dem(tif_path, target_crs)
    sampler = make_sampler(dem_data, transform, nodata)
    return {pt: sampler(pt[0], pt[1]) for pt in points_xy}


def delineate_catchments(tif_path, outfall_pts, snap_radius=50.0, target_crs="EPSG:32640"):
    """
    Delineate one watershed polygon per outfall using D8 flow direction.

    Parameters
    ----------
    tif_path     : str — path to DEM GeoTIFF
    outfall_pts  : list of (of_id, x, y) in target_crs
    snap_radius  : float — max snap distance to high-accumulation cell (m)
    target_crs   : str

    Returns
    -------
    catchments : dict { of_id: shapely.Polygon | None }
                 None if outfall could not be snapped or catchment empty
    """
    from pysheds.grid import Grid
    import rasterio
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import tempfile, os

    # ── Write reprojected DEM to temp file so pysheds can read it ────────────
    dem_data, transform, nodata = load_dem(tif_path, target_crs)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        nr, nc = dem_data.shape
        with rasterio.open(
            tmp_path, "w", driver="GTiff", height=nr, width=nc,
            count=1, dtype="float64", crs=target_crs, transform=transform,
            nodata=nodata if nodata is not None else -9999.0
        ) as dst:
            dst.write(dem_data, 1)

        # ── Condition DEM ─────────────────────────────────────────────────────────
        grid  = Grid.from_raster(tmp_path)
        dem_r = grid.read_raster(tmp_path)

        pit_filled  = grid.fill_pits(dem_r)
        flooded     = grid.fill_depressions(pit_filled)
        inflated    = grid.resolve_flats(flooded)

        # ── D8 flow direction + accumulation ─────────────────────────────────────
        fdir = grid.flowdir(inflated)
        acc  = grid.accumulation(fdir)

        catchments = {}
        for of_id, ox, oy in outfall_pts:
            # Snap outfall to nearest high-accumulation cell within snap_radius.
            # snap_to_mask expects xy as shape (N, 2).
            # Try 1 % threshold first (good for large catchments); if the nearest
            # cell is beyond snap_radius, retry with a 0.1 % threshold which picks
            # up smaller tributaries closer to the outfall location.
            try:
                acc_max = float(acc.max())
                xy_snap = grid.snap_to_mask(
                    acc > acc_max * 0.01,
                    np.array([[ox, oy]])
                )
                x_snap, y_snap = float(xy_snap[0, 0]), float(xy_snap[0, 1])
                snap_dist = float(np.hypot(x_snap - ox, y_snap - oy))
                if snap_dist > snap_radius:
                    # Retry with an absolute 50-cell minimum (catches small headwater
                    # catchments where the 1 % threshold is too restrictive)
                    xy_snap2 = grid.snap_to_mask(
                        acc >= 50,
                        np.array([[ox, oy]])
                    )
                    x2, y2 = float(xy_snap2[0, 0]), float(xy_snap2[0, 1])
                    dist2 = float(np.hypot(x2 - ox, y2 - oy))
                    if dist2 <= snap_radius:
                        x_snap, y_snap, snap_dist = x2, y2, dist2
                    else:
                        print(f"  WARNING OF{of_id}: snap dist {snap_dist:.1f}m > {snap_radius}m, using raw point")
                        x_snap, y_snap = ox, oy
            except Exception as e:
                print(f"  WARNING OF{of_id}: snap failed ({e}), using raw point")
                x_snap, y_snap = ox, oy

            # Delineate catchment
            try:
                catch_mask = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
                # polygonize requires an integer/float dtype; bool is not accepted by rasterio
                catch_mask = catch_mask.astype(np.uint8)
                polys = [shape(s) for s, v in grid.polygonize(catch_mask) if v]
                if not polys:
                    print(f"  WARNING OF{of_id}: empty catchment")
                    catchments[of_id] = None
                else:
                    catchments[of_id] = unary_union(polys)
                    print(f"  OF{of_id}: catchment area = {catchments[of_id].area/1e6:.2f} km²")
            except Exception as e:
                print(f"  ERROR OF{of_id}: catchment delineation failed ({e})")
                catchments[of_id] = None

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return catchments

# SW Network — Stormwater Drainage Network Pipeline

Automated Python pipeline that assigns road centrelines to stormwater drainage sub-networks and performs top-down hydraulic invert routing.

---

## What It Does

1. **Catchment delineation** — uses the DEM to delineate drainage catchments for each outfall (D8 flow accumulation)
2. **Territory assignment** — assigns road segments to the outfall whose catchment they fall within
3. **Directed graph** — builds a directed road network per territory (flow high → low)
4. **Hydraulic routing** — computes pipe invert levels top-down, respecting minimum slope and cover constraints
5. **Pruning** — removes segments that violate max cover depth or cannot hydraulically connect to the outfall
6. **Output** — writes results as SHP, DXF and PNG

---

## Input Files

| File | Description | Example |
|------|-------------|---------|
| `Roads.shp` | Road centrelines (polylines) | `data/SHP/Roads.shp` |
| `outfall.shp` | Outfall points with field `id` (integer) and optional field `depth` (m) | `data/SHP/outfall.shp` |
| `NSA 5m test.tif` | Digital Elevation Model (GeoTIFF, any CRS — auto-reprojected) | `data/Terrain/NSA 5m test.tif` |

> `depth` in the outfall shapefile sets the outfall invert: `I_outfall = ground - depth`. Default = 0 (invert at ground level).

---

## Parameters to Set

Open `py/swnetwork.py` and update the two blocks at the top:

### File paths
```python
BASE        = "D:/Projects/Renardet/SW Net"   # root folder
ROADS_SHP   = f"{BASE}/SHP/Roads.shp"
OUTFALL_SHP = f"{BASE}/SHP/outfall.shp"
DEM_TIF     = f"{BASE}/Terrain/NSA 5m test.tif"
```

### Hydraulic parameters
```python
MIN_SLOPE  = 0.0005   # m/m  minimum pipe gradient (e.g. 0.05%)
MIN_COVER  = 1.0      # m    minimum cover below ground surface
MAX_COVER  = 3.0      # m    maximum allowable pipe depth (segments exceeding this are pruned)
```

Other parameters (spacing, snap tolerances) generally do not need adjustment.

---

## How to Run

```cmd
cd W2/py
python swnetwork.py
```

> Requires: `geopandas`, `rasterio`, `networkx`, `ezdxf`, `scipy`, `matplotlib`, `shapely`

---

## Outputs

| File | Description |
|------|-------------|
| `shp/swnetwork.shp` | Assigned road segments with pipe invert levels |
| `shp/orphans.shp` | Unassigned segments (hydraulically infeasible or disconnected) |
| `dxf/swnetwork.dxf` | Full drainage network drawing with flow arrows, slope labels, invert annotations |
| `img/territories.png` | Catchment territory map |

### SHP attributes
| Field | Description |
|-------|-------------|
| `territory` | Outfall ID |
| `inv_up_m` | Upstream pipe invert (m) |
| `inv_dn_m` | Downstream pipe invert (m) |
| `gnd_up_m` | Upstream ground elevation (m) |
| `gnd_dn_m` | Downstream ground elevation (m) |

### DXF layers
| Layer | Content |
|-------|---------|
| `SW-DRAIN-SUB-01…N` | Pipe segments per sub-network (coloured by territory) |
| `SW-DRAIN-LABELS` | Junction labels: G (ground), I (invert), D (max cover depth) + slope % |
| `SW-INLETS` | Sag / inlet markers |
| `SW-OUTLETS` | Outfall markers with invert label |
| `SW-DRAIN-ORPHAN` | Unassigned segments |

---

## Coordinate System

Input files can be in any CRS. The pipeline reprojects everything to **EPSG:32640** (UTM Zone 40N) internally. Change the target CRS in `swnetwork.py` line `to_crs("EPSG:32640")` if your project is in a different UTM zone.

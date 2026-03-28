# SW Network — Stormwater Drainage Network Pipeline

Automated Python pipeline that builds a hydraulically-correct gravity stormwater drainage network from road centrelines and a DEM. Outputs are ready for GIS review, hydraulic simulation and Civil 3D import.

---

## What It Does — Overview

The pipeline takes road centrelines and outfall points as inputs and produces a fully attributed pipe network: every channel has a name, invert levels at both ends, connected node names, slope, and length. The resulting shapefiles are structured for direct import into hydraulic simulation software (e.g. SWMM, InfoDrainage, SewerGEMS).

---

## Method — Step-by-Step

### 1. Load & Reproject
Roads and outfall shapefiles are loaded and reprojected to the project CRS (UTM). Outfall invert levels are read from the `depth` attribute (`I_outfall = ground − depth`).

### 2. Road Noding & Outfall Snapping
Roads are split at every intersection using `shapely.unary_union` so that all shared endpoints are exact. Each outfall point is then projected onto its nearest road and an exact split node is inserted at that location, ensuring zero snap error.

### 3. DEM Sampling & Ridge–Sag Detection
Elevation is sampled at regular intervals (`SPACING`) along every road segment. Local maxima (ridges/divides) split segments so that each sub-segment has a single consistent flow direction. Local minima (sags) are flagged as potential inlet locations.

### 4. Directed Graph & Gap-Healing
A directed graph is built edge-by-edge (high → low elevation). Short gaps between road end-points are healed with a KDTree spatial search (`CONNECT_TOL`) so disconnected segments re-join the network.

### 5. Territory Assignment — Dijkstra from Outfalls
Multi-source Dijkstra propagates upstream from each outfall snap node using pipe length as cost. Every node is claimed by its nearest outfall, producing non-overlapping drainage territories. Nodes not reachable from any outfall undergo a geographic fallback (nearest outfall by straight-line distance) followed by BFS to capture isolated sub-graphs.

### 6. Catchment Delineation
D8 flow-accumulation on the DEM is used to delineate the contributing surface catchment polygon for each outfall territory.

### 7. Pooling Orphan Segments
Segments that remain unassigned after Dijkstra (disconnected components, hydraulically infeasible connections) enter a **pool**. Pool segments compete to join an existing territory through a connectivity+hydraulic feasibility loop; segments that still cannot connect are permanently orphaned.

### 8. Hydraulic Routing & Pruning
Top-down hydraulic routing (`route_topdown`) propagates invert levels from each source node towards the outfall, enforcing:

- **Minimum slope** (`MIN_SLOPE`) — every pipe must maintain at least this gradient
- **Minimum cover** (`MIN_COVER`) — invert must be at least this depth below ground
- **Maximum cover** (`MAX_COVER`) — pipes that exceed this depth are pruned from the network

Segments violating `MAX_COVER` are pruned iteratively. The pruning may disconnect upstream branches; those branches are re-pooled and may be assigned to a neighbouring territory or orphaned.

The outfall invert is a **physical fixed point**: the entire network routes hydraulically to meet it. The outfall depth is supplied via the `depth` field in the outfall shapefile.

### 9. Fan-out Resolution (One-Out Rule)
A valid gravity network is a tree: every junction may have multiple incoming channels but only **one** outgoing channel. The pipeline detects junctions with more than one outgoing channel and resolves them:

- The **winner** is the outgoing channel that, when routed top-down, yields the highest calculated outfall invert (least energy loss). At cross-territory junctions the winner is chosen by connectivity (most upstream tributaries), with steepest drop as a tie-breaker.
- All **losers** are detached from the junction and given a `FANOUT_GAP_M` gap from their upstream end, making them new source segments with `I_head = ground − MIN_COVER`.
- Losers shorter than `FANOUT_GAP_M`, or where maintaining `MIN_SLOPE` would violate `MIN_COVER`, are **orphaned**.

After fan-out resolution the full top-down hydraulic routing is re-run to update all invert levels in the corrected tree.

### 10. Dendritic Naming & Output
Dijkstra from each outfall on the resolved tree computes the longest path distance to every node. Nodes and channels are numbered in dendritic order — the **farthest element from the outfall receives index 1** (e.g. `O1-J1`, `O1-C1`). Output files are then written.

---

## Inputs

### Shapefiles

| File | Geometry | Required fields | Notes |
|------|----------|-----------------|-------|
| `data/SHP/Roads.shp` | Polyline | — | Road centrelines; any CRS (auto-reprojected) |
| `data/SHP/outfall.shp` | Point | `id` (integer), `depth` (float, m) | Outfall locations; `depth` = pipe depth below ground at outfall |

> If `depth` is missing or zero the outfall invert equals the ground elevation (pipe flush with surface). Set it to the physical invert depth of the outfall structure (e.g. culvert soffit, chamber invert).

### Terrain

| File | Format | Notes |
|------|--------|-------|
| `data/Terrain/NSA 5m test.tif` | GeoTIFF (any CRS) | 5 m DEM; auto-reprojected to project CRS |

---

## Parameters

Open `py/swnetwork.py` and update the blocks at the top.

### File paths
```python
BASE        = "D:/Projects/Renardet/SW Net - 2"
ROADS_SHP   = f"{BASE}/W2/data/SHP/Roads.shp"
OUTFALL_SHP = f"{BASE}/W2/data/SHP/outfall.shp"
DEM_TIF     = f"{BASE}/W2/data/Terrain/NSA 5m test.tif"
```

### Hydraulic parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `MIN_SLOPE` | `0.0005` | m/m | Minimum pipe gradient (0.05%). All pipes must meet or exceed this. |
| `MIN_COVER` | `1.0` | m | Minimum cover from ground to pipe crown. Pipes shallower than this are orphaned. |
| `MAX_COVER` | `3.0` | m | Maximum allowable pipe depth. Pipes deeper than this are pruned. |
| `FANOUT_GAP_M` | `10.0` | m | Gap applied to the upstream end of fan-out loser channels. |
| `SAG_DROP` | `0.05` | m | Minimum elevation drop to flag a local low as a sag/inlet point. |
| `RIDGE_RISE` | `0.10` | m | Minimum elevation rise to flag a local high as a ridge/divide. |
| `SPACING` | `2.0` | m | DEM sampling interval along roads. |
| `CONNECT_TOL` | `0.5` | m | Maximum gap between road endpoints to snap/heal. |

> Outfall invert depth is set per-outfall via the `depth` field in `outfall.shp`, not as a global parameter.

---

## How to Run

```cmd
cd W2
python -m venv .venv
.venv\Scripts\activate
pip install -r py/requirements.txt

python py/swnetwork.py
```

---

## Output Files

### Shapefiles (`shp/`)

| File | Geometry | Description |
|------|----------|-------------|
| `swnetwork.shp` | Polyline | All assigned pipe channels with full hydraulic attributes |
| `nodes.shp` | Point | All active network nodes (junctions, sags, ridges, outfalls) |
| `orphan_channels.shp` | Polyline | Segments that could not be assigned to any territory |
| `orphan_nodes.shp` | Point | Nodes where every connecting channel is orphaned |
| `catchments.shp` | Polygon | Surface drainage catchment per outfall territory |
| `sw_inlets.shp` | Point | Sag/inlet locations on the active network |
| `sw_ridges.shp` | Point | Ridge/divide points on the active network |

#### `swnetwork.shp` — channel attributes

| Field | Description |
|-------|-------------|
| `name` | Channel ID in dendritic order, e.g. `O1-C1` (farthest channel = C1) |
| `territory` | Outfall ID this channel drains to |
| `node_up` | Name of the upstream node |
| `node_dn` | Name of the downstream node |
| `inv_up` | Pipe invert elevation at upstream end (m) |
| `inv_dn` | Pipe invert elevation at downstream end (m) |
| `gnd_up` | Ground elevation at upstream end (m) |
| `gnd_dn` | Ground elevation at downstream end (m) |
| `length_m` | Channel length (m) |

#### `nodes.shp` — node attributes

| Field | Description |
|-------|-------------|
| `name` | Node ID in dendritic order, e.g. `O1-J3` (junction), `O1-S1` (sag), `O1-R2` (ridge), `O1` (outfall) |
| `type` | `outfall` / `junction` / `sag` / `ridge` |
| `territory` | Outfall ID |
| `ground` | Ground elevation (m) |
| `invert` | Pipe invert elevation (m) |
| `depth` | Cover depth: ground − invert (m) |

> **These two shapefiles are ready for hydraulic simulation.** Each channel carries its own invert levels and references its upstream and downstream node by name, providing all connectivity data needed by SWMM, InfoDrainage or equivalent tools.

### DXF (`dxf/swnetwork.dxf`)

The DXF is the most comprehensive representation of the network and is the recommended first check after every run.

| Layer | Colour | Content |
|-------|--------|---------|
| `SW-DRAIN-SUB-01` … `SW-DRAIN-SUB-N` | ACI 1–6 (cyclic) | Pipe segments coloured by territory |
| `SW-DRAIN-LABELS` | Mixed | All text annotations (see below) |
| `SW-DRAIN-ORPHAN` | Grey (8) | Unassigned orphan segments |
| `SW-INLETS` | Red (1) | Sag/inlet markers |
| `SW-OUTLETS` | Yellow (2) | Outfall markers (double circle + label) |

#### Label stack at each node
```
  Name   ← node name (e.g. O2-J5)         colour white
  G:nnn  ← ground elevation               colour white
  I:nnn  ← pipe invert elevation          colour cyan
  D:nnn  ← cover depth (G − I)            colour magenta
```

#### Label stack along each channel (at midpoint, perpendicular to pipe)
```
  O1-C7     ← channel name                above the line  (green)
  ─────── pipe ───────
  0.73%     ← pipe slope                  below the line  (yellow)
  52.3m     ← channel length              below slope     (yellow)
```

#### What to check in the DXF
- **Flow arrows** on every segment confirm the hydraulic flow direction.
- **D values** at nodes should be ≥ `MIN_COVER` (except at outfall nodes where D = outfall depth).
- **Fan-out gaps** — short breaks at the upstream end of loser channels are visible as small spaces at junctions; confirm only one line exits each junction.
- **Orphan segments** on layer `SW-DRAIN-ORPHAN` indicate areas with insufficient hydraulic gradient or disconnected roads.
- **Territory colours** give an immediate visual check of catchment extents and any anomalous territory boundaries.

### PNG (`img/territories.png`)

Plan-view map of territory assignment coloured by outfall, with orphan segments shown in grey. Useful for a quick overview before opening the DXF.

---

## Coordinate System

All inputs can be in any CRS. The pipeline reprojects everything to **EPSG:32640** (UTM Zone 40N) for all calculations. Change the `to_crs("EPSG:32640")` line in `swnetwork.py` if your project uses a different UTM zone.

---

## Project Structure

```
W2/
├── data/
│   ├── SHP/           ← input road and outfall shapefiles
│   └── Terrain/       ← input DEM (GeoTIFF)
├── py/
│   ├── swnetwork.py   ← main script (entry point)
│   ├── dem.py         ← DEM loading, sampling, catchment delineation
│   ├── roads.py       ← road loading, noding, ridge/sag detection
│   ├── graph.py       ← directed graph construction, territory assignment
│   ├── hydraulics.py  ← top-down routing, pruning, fan-out resolution
│   ├── outputs.py     ← SHP, DXF and PNG writers
│   ├── requirements.txt
│   └── tests/         ← unit tests
├── shp/               ← output shapefiles (generated)
├── dxf/               ← output DXF (generated)
└── img/               ← output PNG (generated)
```

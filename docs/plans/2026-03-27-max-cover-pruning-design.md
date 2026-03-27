# MAX_COVER Pruning with Alternative-Path Rescue
**Date:** 2026-03-27
**Feature:** Hard cap on pipe depth + rescue of viable upstream nodes

---

## Problem

Top-down hydraulic routing accumulates excess pipe cover on uphill road segments.
A road rising 3m over 200m toward the outfall forces the pipe ~3m deeper than
`min_cover`. The deepest-wins junction rule propagates this depth downstream,
causing outfall feasibility checks to fail and triggering aggressive pruning.

---

## Solution

Add `MAX_COVER = 2.0 m` as a hard per-node constraint applied **before** the
outfall feasibility check.

### Algorithm

```
repeat until stable:
  1. route_topdown → compute inverts for all nodes
  2. find all nodes where (ground − invert) > MAX_COVER  → "deep nodes"
  3. if none → check outfall feasibility (I_arrived ≥ I_outfall) → done
  4. remove deep nodes from working graph
  5. re-check connectivity: nodes that still have a directed path to the
     outfall are kept and re-routed; nodes that lost all paths → orphaned
  6. repeat
fallback: if outfall check still fails after max-cover pruning,
          existing bottleneck pruning handles the remainder
```

### Key Property: Alternative-Path Rescue

Removing a deep node may disconnect some upstream nodes — but not all.
At road intersections, an upstream node may have another directed edge
(high→low) that bypasses the deep segment and reconnects to the outfall.
Connectivity is re-checked after each removal so viable upstream nodes
are preserved rather than blindly orphaned.

---

## Code Changes

| File | Change |
|---|---|
| `swnetwork.py` | Add `MAX_COVER = 2.0` to parameters block; pass to `prune_to_feasibility` |
| `hydraulics.py` | Add `prune_by_max_cover(G, outfall_node, inverts, max_cover)` function |
| `hydraulics.py` | `prune_to_feasibility` calls `prune_by_max_cover` first, then bottleneck fallback |

---

## Parameters

```python
MAX_COVER = 2.0   # m — maximum allowable pipe depth below ground (adjustable)
MIN_COVER = 1.0   # m — minimum cover (unchanged)
```

---

## Expected Impact

- Uphill-hump segments (the main cause of deep cover) are pruned early
- They can no longer drag downstream junctions via deepest-wins
- Remaining network routes cleanly → outfall check passes more often
- Net result: fewer orphans, higher assignment percentage
- Segments needing depth > 2m flagged as orphans (require separate design)

# W2/py/tests/test_roads.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shapely.geometry import LineString, Polygon
from roads import ridge_split_pts, detect_ridges, snap_outfalls_to_road_graph, assign_majority, reassign_boundary_roads

def test_no_ridge_returns_single_segment():
    # Monotonically falling profile — no ridge
    pts = [(0,0,10.0), (1,0,9.0), (2,0,8.0), (3,0,7.0)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 1
    assert result[0] == pts

def test_ridge_in_middle_splits_into_two():
    # Goes up then down — ridge at index 2
    pts = [(0,0,5.0), (1,0,6.0), (2,0,7.5), (3,0,6.0), (4,0,5.0)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 2
    assert result[0][-1] == pts[2]   # first piece ends at ridge
    assert result[1][0]  == pts[2]   # second piece starts at ridge

def test_two_ridges_split_into_three():
    pts = [(0,0,5.0),(1,0,7.0),(2,0,5.0),(3,0,7.0),(4,0,5.0)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 3


def test_endpoint_high_point_is_not_a_ridge():
    """First and last points are excluded from ridge detection by design."""
    # High at start, descending — start is NOT a ridge
    pts = [(0,0,10.0), (1,0,8.0), (2,0,6.0)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 1  # no split

def test_ridge_rise_threshold_boundary():
    """Point must exceed BOTH neighbours by MORE than ridge_rise (strict >)."""
    # Middle point is exactly ridge_rise above neighbours — should NOT be a ridge
    pts = [(0,0,5.0), (1,0,5.05), (2,0,5.0)]
    result_exact = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result_exact) == 1   # 5.05 > 5.0 + 0.05 is False → no ridge

    # Middle point is ridge_rise + epsilon above neighbours — IS a ridge
    pts2 = [(0,0,5.0), (1,0,5.051), (2,0,5.0)]
    result_above = ridge_split_pts(pts2, ridge_rise=0.05)
    assert len(result_above) == 2   # split at ridge

def test_two_point_input_returns_unchanged():
    """Minimum valid input — no ridges possible, returned as-is."""
    pts = [(0,0,10.0), (1,0,5.0)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 1
    assert result[0] == pts

def test_flat_profile_no_ridge():
    """All equal elevations — no ridge detected."""
    pts = [(i, 0, 5.0) for i in range(5)]
    result = ridge_split_pts(pts, ridge_rise=0.05)
    assert len(result) == 1


def test_snap_moves_outfall_to_local_minimum():
    """
    Graph: A(elev=10) - B(elev=5) - C(elev=8)
    B is a local minimum. Outfall placed near A (1m away).
    Within snap_r=20m B is the nearest local min → outfall should snap to B.
    Outfall is NOT on any road line (y=5.0, roads are at y=0) → off-road mode.
    """
    seg_tuples = [
        ([(0,0,10.0),(5,0,5.0)], LineString([(0,0),(5,0)])),
        ([(5,0,5.0),(10,0,8.0)], LineString([(5,0),(10,0)])),
    ]
    of_pts = [(1, 0.5, 5.0)]   # OF1 near A but offset 5m off road
    snapped, _ = snap_outfalls_to_road_graph(of_pts, seg_tuples, snap_r=20.0)
    of_id, sx, sy = snapped[0]
    assert of_id == 1
    assert abs(sx - 5.0) < 0.1
    assert abs(sy - 0.0) < 0.1

def test_snap_falls_back_to_lowest_when_no_local_min():
    """
    Monotonically falling line: A(10) - B(7) - C(4).
    No internal local minimum. Outfall near A but offset 5m off road, snap_r=50m.
    Should fall back to C (lowest elevation candidate).
    """
    seg_tuples = [
        ([(0,0,10.0),(5,0,7.0)], LineString([(0,0),(5,0)])),
        ([(5,0,7.0),(10,0,4.0)], LineString([(5,0),(10,0)])),
    ]
    of_pts = [(1, 0.3, 5.0)]   # offset 5m off road → off-road mode
    snapped, _ = snap_outfalls_to_road_graph(of_pts, seg_tuples, snap_r=50.0)
    _, sx, sy = snapped[0]
    assert abs(sx - 10.0) < 0.1   # snapped to C (lowest)

def test_snap_outside_radius_returns_original():
    """If no road node within snap_r and outfall not on road, keep original position."""
    seg_tuples = [
        ([(100,100,5.0),(110,100,4.0)], LineString([(100,100),(110,100)])),
    ]
    of_pts = [(1, 0.0, 0.0)]   # far away
    snapped, _ = snap_outfalls_to_road_graph(of_pts, seg_tuples, snap_r=10.0)
    _, sx, sy = snapped[0]
    assert abs(sx - 0.0) < 0.01 and abs(sy - 0.0) < 0.01

def test_snap_on_road_inserts_node():
    """
    Outfall placed exactly on the middle of a road segment.
    Should split the segment and snap to 0m.
    """
    line = LineString([(0,0),(100,0)])
    pts  = [(0,0,10.0),(50,0,8.0),(100,0,6.0)]
    seg_tuples = [(pts, line)]
    of_pts = [(1, 50.0, 0.0)]   # exactly on road at midpoint
    snapped, updated = snap_outfalls_to_road_graph(of_pts, seg_tuples, snap_r=5.0)
    _, sx, sy = snapped[0]
    assert abs(sx - 50.0) < 0.1 and abs(sy - 0.0) < 0.1   # snapped to projection
    assert len(updated) == 2   # segment was split into two


def test_assign_majority_whole_segment_to_dominant_catchment():
    """
    Road: x=0..10m
    Catchment 2 (passed FIRST) covers x=0..3 → 3m overlap  (minority)
    Catchment 1 (passed SECOND) covers x=2..10 → 8m overlap (majority)
    Correct majority logic must return territory=1.
    A first-match implementation would wrongly return territory=2.
    Road geometry must be returned unchanged (no splitting).
    """
    road = LineString([(0,0),(10,0)])
    pts  = [(0,0,5.0),(10,0,4.0)]
    c2_small = Polygon([(0,-1),(3,-1),(3,1),(0,1)])    # 3 m overlap, visited first
    c1_large = Polygon([(2,-1),(10,-1),(10,1),(2,1)])  # 8 m overlap, visited second
    # Dict order: 2 first, 1 second
    result = assign_majority([(pts, road)], {2: c2_small, 1: c1_large})
    assert len(result) == 1              # never split
    assert result[0]['territory'] == 1   # majority (8m) wins over first-match (3m)
    assert result[0]['geom'] == road     # original geometry unchanged
    assert 'blacklist' in result[0]


def test_assign_majority_unassigned_when_no_overlap():
    """Road completely outside all catchments → territory=None."""
    road = LineString([(100,0),(110,0)])
    pts  = [(100,0,5.0),(110,0,4.0)]
    c1   = Polygon([(0,-1),(10,-1),(10,1),(0,1)])
    result = assign_majority([(pts, road)], {1: c1})
    assert result[0]['territory'] is None
    assert 'blacklist' in result[0]


def test_reassign_road_connected_only_to_other_territory():
    """
    Road R: endpoint P1 connects only to territory-2 nodes.
    Road R is initially assigned to territory 1 (wrong).
    Should be reassigned to territory 2.
    """
    # Territory 1: segment A→B (nodes at x=0 and x=5)
    # Territory 2: segment B→C (nodes at x=5 and x=10)
    # Road R: segment C→D (nodes at x=10 and x=15), initially assigned to territory 1 (wrong)
    segs = [
        {'pts':[(0,0,10.0),(5,0,8.0)], 'geom':None, 'territory':1, 'blacklist':set()},
        {'pts':[(5,0,8.0),(10,0,6.0)], 'geom':None, 'territory':2, 'blacklist':set()},
        {'pts':[(10,0,6.0),(15,0,4.0)],'geom':None, 'territory':1, 'blacklist':set()},  # R
    ]
    result = reassign_boundary_roads(segs)
    # Road R (index 2) should move to territory 2 because both its endpoints
    # are only connected to territory-2 nodes (not territory-1 nodes)
    assert result[2]['territory'] == 2
    assert 1 in result[2]['blacklist']

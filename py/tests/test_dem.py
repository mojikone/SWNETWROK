# W2/py/tests/test_dem.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dem import make_sampler

def test_sampler_bilinear_interpolation_correct():
    """
    Non-uniform 3×3 grid: verify bilinear interpolation gives analytically correct result.
    Grid (row, col) → elevation:
      row0: [0, 1, 2]
      row1: [3, 4, 5]
      row2: [6, 7, 8]
    Transform: c=0, a=1, f=2, e=-1  → x=col, y=2-row
    Query point (x=0.5, y=1.5) → col_f=0.5, row_f=0.5
    Bilinear: (0*(1-0.5)*(1-0.5) + 1*(1-0.5)*0.5 + 3*0.5*(1-0.5) + 4*0.5*0.5)
            = 0*0.25 + 1*0.25 + 3*0.25 + 4*0.25 = 0 + 0.25 + 0.75 + 1.0 = 2.0
    """
    data = np.arange(9, dtype=float).reshape(3, 3)
    class FakeTransform:
        c = 0.0; a = 1.0; f = 2.0; e = -1.0
    sampler = make_sampler(data, FakeTransform(), nodata=None)
    result = sampler(0.5, 1.5)
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-10

def test_sampler_returns_none_outside_bounds():
    data = np.full((3, 3), 50.0)
    class FakeTransform:
        c = 0.0; a = 1.0; f = 2.0; e = -1.0
    sampler = make_sampler(data, FakeTransform(), nodata=None)
    result = sampler(999.0, 999.0)   # way outside
    assert result is None

def test_sampler_returns_none_for_nodata():
    nodata = -9999.0
    data = np.full((3, 3), nodata)
    class FakeTransform:
        c = 0.0; a = 1.0; f = 2.0; e = -1.0
    sampler = make_sampler(data, FakeTransform(), nodata=nodata)
    result = sampler(0.5, 1.5)
    assert result is None

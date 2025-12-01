#!/usr/bin/env python
"""Test script for _assign_out functionality"""
import numpy as np
import sys

# Test imports
try:
    from pandas.core.arrays.sparse.array import SparseArray, _assign_out
    print("✓ Successfully imported SparseArray and _assign_out")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 1: Dense array output
print("\nTest 1: Dense array output")
try:
    arr = SparseArray([0, 1, 0, 2, 0])
    result = np.maximum(arr, 1)
    out = np.zeros(5, dtype=float)
    _assign_out(out, result, where=None)
    print(f"  Input: {arr.to_dense()}")
    print(f"  Result after np.maximum(arr, 1): {result}")
    print(f"  Output array: {out}")
    assert np.allclose(out, [1., 1., 1., 2., 1.]), f"Expected [1., 1., 1., 2., 1.], got {out}"
    print("✓ Test 1 passed")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: SparseArray output
print("\nTest 2: SparseArray output")
try:
    arr = SparseArray([0, 1, 0, 2, 0])
    result = SparseArray([1, 1, 1, 2, 1])
    out = SparseArray([0, 0, 0, 0, 0])
    _assign_out(out, result, where=None)
    print(f"  Input: {arr.to_dense()}")
    print(f"  Result: {result.to_dense()}")
    print(f"  Output array: {out.to_dense()}")
    assert np.allclose(out.to_dense(), [1., 1., 1., 2., 1.]), f"Expected [1., 1., 1., 2., 1.], got {out.to_dense()}"
    print("✓ Test 2 passed")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: With where mask
print("\nTest 3: With where mask")
try:
    result = SparseArray([1, 2, 3, 4, 5])
    out = np.array([0, 0, 0, 0, 0])
    where = np.array([True, False, True, False, True])
    _assign_out(out, result, where=where)
    print(f"  Result: {result.to_dense()}")
    print(f"  Where mask: {where}")
    print(f"  Output array: {out}")
    assert np.array_equal(out, [1, 0, 3, 0, 5]), f"Expected [1, 0, 3, 0, 5], got {out}"
    print("✓ Test 3 passed")
except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests passed!")

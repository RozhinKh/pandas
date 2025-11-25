#!/usr/bin/env python
"""
Simple validation script to test min/max ufunc operations with sparse arrays.
This script tests the basic functionality without requiring pytest.
"""

import numpy as np
import sys

# Add the pandas module to path (assuming we're in project root)
sys.path.insert(0, '.')

try:
    from pandas.core.arrays.sparse import SparseArray
    print("✓ Successfully imported SparseArray")
except ImportError as e:
    print(f"✗ Failed to import SparseArray: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic min/max ufunc functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test 1: sparse/sparse with same fill values
    print("\nTest 1: sparse/sparse with same fill values")
    a = SparseArray([0, 1, 0, 2, 0], fill_value=0)
    b = SparseArray([0, 3, 0, 1, 0], fill_value=0)
    
    result = np.maximum(a, b)
    print(f"  np.maximum([0, 1, 0, 2, 0], [0, 3, 0, 1, 0]) = {result.to_dense()}")
    assert isinstance(result, SparseArray), "Result should be SparseArray"
    expected = np.array([0, 3, 0, 2, 0])
    assert np.array_equal(result.to_dense(), expected), f"Expected {expected}, got {result.to_dense()}"
    print("  ✓ Test passed")
    
    # Test 2: sparse/scalar
    print("\nTest 2: sparse/scalar")
    arr = SparseArray([0, 1, 0, 2, 0], fill_value=0)
    result = np.maximum(arr, 1)
    print(f"  np.maximum([0, 1, 0, 2, 0], 1) = {result.to_dense()}")
    assert isinstance(result, SparseArray), "Result should be SparseArray"
    expected = np.array([1, 1, 1, 2, 1])
    assert np.array_equal(result.to_dense(), expected), f"Expected {expected}, got {result.to_dense()}"
    print("  ✓ Test passed")
    
    # Test 3: NaN propagation with maximum
    print("\nTest 3: NaN propagation with maximum")
    a = SparseArray([0, 1, np.nan, 2, 0], fill_value=0)
    b = SparseArray([0, 3, 1, 1, 0], fill_value=0)
    result = np.maximum(a, b)
    print(f"  np.maximum([0, 1, nan, 2, 0], [0, 3, 1, 1, 0]) = {result.to_dense()}")
    assert isinstance(result, SparseArray), "Result should be SparseArray"
    assert np.isnan(result.to_dense()[2]), "NaN should be propagated"
    print("  ✓ Test passed")
    
    # Test 4: NaN handling with fmax
    print("\nTest 4: NaN handling with fmax")
    a = SparseArray([0, 1, np.nan, 2, 0], fill_value=0)
    b = SparseArray([0, 3, 1, 1, 0], fill_value=0)
    result = np.fmax(a, b)
    print(f"  np.fmax([0, 1, nan, 2, 0], [0, 3, 1, 1, 0]) = {result.to_dense()}")
    assert isinstance(result, SparseArray), "Result should be SparseArray"
    assert result.to_dense()[2] == 1.0, "NaN should be ignored, value should be 1.0"
    print("  ✓ Test passed")
    
    # Test 5: All four ufuncs
    print("\nTest 5: All four ufuncs work")
    a = SparseArray([0, 1, 0, 2, 0], fill_value=0)
    b = SparseArray([0, 3, 0, 1, 0], fill_value=0)
    
    for ufunc in [np.maximum, np.minimum, np.fmax, np.fmin]:
        result = ufunc(a, b)
        assert isinstance(result, SparseArray), f"{ufunc.__name__} should return SparseArray"
        print(f"  ✓ {ufunc.__name__} works correctly")
    
    print("\n=== All Tests Passed! ===")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n✓ Validation successful!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

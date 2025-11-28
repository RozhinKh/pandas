#!/usr/bin/env python3
"""
Quick validation script for Phase 1 sparse ufunc edge cases.
Tests the new edge case functions to ensure they work correctly.
"""

import numpy as np
import pandas as pd
from pandas.arrays import SparseArray

def test_empty_array():
    """Test ufuncs on empty SparseArray"""
    print("Testing empty arrays...")
    arr = SparseArray([], dtype=np.float64)
    result = np.abs(arr)
    assert len(result) == 0
    print("  ✓ Empty array test passed")

def test_all_sparse():
    """Test ufuncs on arrays with no fill_value elements"""
    print("Testing all-sparse arrays...")
    arr = SparseArray([1, 2, 3], fill_value=0)
    result = np.abs(arr)
    assert np.array_equal(result.to_dense(), [1, 2, 3])
    print("  ✓ All-sparse array test passed")

def test_dtype_promotion():
    """Test dtype promotion in ufunc operations"""
    print("Testing dtype promotion...")
    int_arr = SparseArray([0, 1, 0, 2], dtype=np.int64, fill_value=0)
    result = np.add(int_arr, 1.5)
    assert result.dtype.subtype == np.float64
    print("  ✓ Dtype promotion test passed")

def test_nan_handling():
    """Test ufunc operations with NaN fill values"""
    print("Testing NaN handling...")
    arr = SparseArray([np.nan, 1.0, np.nan, 2.0], fill_value=np.nan)
    result = np.exp(arr)
    assert len(result) == 4
    print("  ✓ NaN handling test passed")

def test_inf_handling():
    """Test ufunc operations with infinity values"""
    print("Testing infinity handling...")
    arr = SparseArray([0.0, np.inf, 0.0, -np.inf, 1.0], fill_value=0.0)
    result = np.abs(arr)
    dense_result = result.to_dense()
    assert dense_result[1] == np.inf
    assert dense_result[3] == np.inf
    print("  ✓ Infinity handling test passed")

def test_extreme_sparsity():
    """Test ufuncs on arrays with extreme sparsity"""
    print("Testing extreme sparsity...")
    data = [0] * 1000 + [1]
    arr = SparseArray(data, fill_value=0)
    assert arr.density < 0.01
    result = np.abs(arr)
    assert result.density < 0.01
    print("  ✓ Extreme sparsity test passed")

def test_mixed_fill_values():
    """Test binary ufuncs with different fill_values"""
    print("Testing mixed fill_values...")
    arr1 = SparseArray([0, 1, 0, 2, 0], fill_value=0)
    arr2 = SparseArray([5, 5, 3, 5, 4], fill_value=5)
    result = np.add(arr1, arr2)
    expected = [5, 6, 3, 7, 4]
    assert np.array_equal(result.to_dense(), expected)
    print("  ✓ Mixed fill_values test passed")

def test_comparison_edge_cases():
    """Test comparison ufuncs with edge cases"""
    print("Testing comparison edge cases...")
    arr1 = SparseArray([1, 1, 1], fill_value=1)
    arr2 = SparseArray([1, 1, 1], fill_value=1)
    result = np.equal(arr1, arr2)
    assert np.all(result.to_dense())
    print("  ✓ Comparison edge cases test passed")

def main():
    print("\n" + "="*60)
    print("Phase 1 Sparse Ufunc Edge Case Validation")
    print("="*60 + "\n")
    
    try:
        test_empty_array()
        test_all_sparse()
        test_dtype_promotion()
        test_nan_handling()
        test_inf_handling()
        test_extreme_sparsity()
        test_mixed_fill_values()
        test_comparison_edge_cases()
        
        print("\n" + "="*60)
        print("✅ All edge case tests passed successfully!")
        print("="*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

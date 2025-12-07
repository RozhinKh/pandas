"""Test script for _sparse_ufunc_scalar implementation"""
import numpy as np
import pandas as pd
from pandas.arrays import SparseArray

# Test 1: Basic add operation
print("Test 1: Basic np.add operation")
arr = SparseArray([1, -1, 2, -2], fill_value=1)
result = np.add(arr, 1)
expected = SparseArray([2, 0, 3, -1], fill_value=2)
print(f"Input: {np.asarray(arr)}, scalar=1")
print(f"Result: {np.asarray(result)}")
print(f"Expected: {np.asarray(expected)}")
print(f"Match: {np.allclose(np.asarray(result), np.asarray(expected))}")
print(f"Result dtype: {result.dtype}")
print(f"Result fill_value: {result.fill_value}")
print()

# Test 2: Float operations
print("Test 2: Float operations")
arr = SparseArray([1.0, -1.0, 2.0, -2.0], fill_value=0.0)
result = np.add(arr, 1.5)
expected = np.add(np.asarray(arr), 1.5)
print(f"Input: {np.asarray(arr)}, scalar=1.5")
print(f"Result: {np.asarray(result)}")
print(f"Expected: {expected}")
print(f"Match: {np.allclose(np.asarray(result), expected)}")
print()

# Test 3: Multiply operation
print("Test 3: Multiply operation")
arr = SparseArray([1, 0, 2, 0], fill_value=0)
result = np.multiply(arr, 2)
expected = np.multiply(np.asarray(arr), 2)
print(f"Input: {np.asarray(arr)}, scalar=2")
print(f"Result: {np.asarray(result)}")
print(f"Expected: {expected}")
print(f"Match: {np.allclose(np.asarray(result), expected)}")
print()

# Test 4: Greater operation
print("Test 4: Greater operation")
arr = SparseArray([1, 0, 2, 0], fill_value=0)
result = np.greater(arr, 0)
expected = np.greater(np.asarray(arr), 0)
print(f"Input: {np.asarray(arr)}, scalar=0")
print(f"Result: {np.asarray(result)}")
print(f"Expected: {expected}")
print(f"Match: {np.array_equal(np.asarray(result), expected)}")
print()

# Test 5: Check that sparsity pattern is preserved
print("Test 5: Sparsity pattern preservation")
arr = SparseArray([0, 0, 1, 2, 0], fill_value=0)
print(f"Original sp_index.indices: {arr.sp_index.indices}")
result = np.add(arr, 1)
print(f"Result sp_index.indices: {result.sp_index.indices}")
print(f"Indices match: {np.array_equal(arr.sp_index.indices, result.sp_index.indices)}")
print(f"Original sparse values: {arr.sp_values}")
print(f"Result sparse values: {result.sp_values}")
print()

print("All tests completed!")

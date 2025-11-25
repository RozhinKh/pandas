# Sparse-Optimized Binary Ufunc Handling Implementation

## Summary

Successfully implemented sparse-optimized handling for critical binary ufuncs in the `__array_ufunc__` method of `SparseArray`. This optimization prevents unnecessary conversion to dense arrays for operations that have Cython implementations in splib.

## Changes Made

### File: `pandas/core/arrays/sparse/array.py`

**Location**: Lines 1763-1829 in the `__array_ufunc__` method

### Implementation Details

Added a new code block that intercepts binary operations before they fall back to dense conversion:

1. **Operation Name Mapping**
   - Uses `UFUNC_ALIASES` from `pandas._libs.ops_dispatch` to map numpy ufunc names to splib operation names
   - Added `_LOGICAL_ALIASES` to map logical ufuncs (logical_and, logical_or, logical_xor) to bitwise ops (and, or, xor)

2. **Supported Operations**
   - **Arithmetic**: add, sub, mul, truediv, floordiv, mod, pow
   - **Comparison**: eq, ne, lt, le, gt, ge
   - **Logical/Bitwise**: and, or, xor

3. **Three Input Combination Handlers**

   **a) Sparse/Sparse**: 
   - Directly uses `_sparse_array_op(left, right, ufunc, op_name)`
   - Leverages Cython-optimized sparse operations

   **b) Sparse/Scalar**:
   - Applies ufunc only to `sp_values`
   - Computes new `fill_value` using `ufunc(_get_fill(sparse_array), scalar)`
   - Wraps result using `_wrap_result(op_name, result, sp_index, fill)`

   **c) Sparse/Dense**:
   - Converts dense array to `SparseArray` using the sparse array's fill_value
   - Routes to sparse/sparse handler via `_sparse_array_op`

4. **Key Design Decisions**
   - Preserves original input order for proper non-commutative operation semantics
   - Uses `np.errstate(all="ignore")` to handle numerical errors gracefully
   - Validates array lengths before operations
   - Falls back to original dense conversion for unsupported operations

## Pattern Consistency

The implementation follows the established patterns from:
- `_arith_method()` (lines 1842-1873): Shows scalar and dense handling for arithmetic
- `_cmp_method()` (lines 1875-1902): Shows similar pattern for comparisons

Both methods use `_sparse_array_op()` for sparse/sparse operations, which is the same approach used in this implementation.

## Benefits

1. **Memory Efficiency**: Avoids converting sparse arrays to dense for supported operations
2. **Performance**: Leverages Cython-optimized sparse operations
3. **Correctness**: Proper fill_value computation maintains sparsity invariants
4. **Compatibility**: Seamlessly integrates with existing NumPy ufunc interface

## Test Coverage

The implementation handles the following success criteria:
- ✓ `np.add(sparse, sparse)`, `np.add(sparse, scalar)`, `np.add(sparse, dense)` return `SparseArray`
- ✓ Result values and fill_values are correct for all three input combinations
- ✓ Comparison ufuncs (e.g., `np.greater`) work with all combinations
- ✓ Logical ufuncs (e.g., `np.logical_and`) work with all combinations
- ✓ No conversion to dense arrays for critical ufuncs with Cython implementations
- ✓ Operations follow the same correctness patterns as `_arith_method()` and `_cmp_method()`

## Example Usage

```python
import numpy as np
from pandas.arrays import SparseArray

# Sparse + Sparse (optimized)
arr1 = SparseArray([0, 1, 0, 2, 0])
arr2 = SparseArray([0, 2, 0, 3, 0])
result = np.add(arr1, arr2)  # Returns SparseArray, no dense conversion

# Sparse + Scalar (optimized)
arr = SparseArray([0, 1, 0, 2, 0])
result = np.add(arr, 1)  # Returns SparseArray, operates on sp_values only

# Sparse + Dense (optimized)
arr = SparseArray([0, 1, 0, 2, 0])
dense = np.array([1, 2, 3, 4, 5])
result = np.add(arr, dense)  # Returns SparseArray, dense converted to sparse first

# Comparison operations (optimized)
arr = SparseArray([0, 1, 0, 2, 0])
result = np.greater(arr, 1)  # Returns SparseArray with boolean values

# Logical operations (optimized)
arr1 = SparseArray([True, False, True, False])
arr2 = SparseArray([False, False, True, True])
result = np.logical_and(arr1, arr2)  # Returns SparseArray, uses Cython-optimized 'and'
```

## Dependencies

- Depends on `_sparse_array_op()` (lines 161-266)
- Depends on `_get_fill()` (lines 137-158)
- Depends on `_wrap_result()` (lines 269-289)
- Depends on `UFUNC_ALIASES` from `pandas._libs.ops_dispatch`

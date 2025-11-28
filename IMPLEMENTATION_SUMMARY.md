# Implementation Summary: Min/Max Ufunc Special Handling

## Overview
Implemented dedicated sparse-aware handling for min/max ufuncs (np.minimum, np.maximum, np.fmin, np.fmax) which don't map to standard Python operators.

## Changes Made

### 1. Module-Level Constant (Line 137)
Added `MIN_MAX_UFUNCS` constant containing the four ufuncs that require special handling:
```python
MIN_MAX_UFUNCS = {np.minimum, np.maximum, np.fmin, np.fmax}
```

### 2. Helper Method: `_handle_minmax_ufunc` (Lines 1824-1916)
Created a comprehensive helper method that handles all cases for min/max ufuncs:

#### Sparse-Scalar Operations (Lines 1865-1876)
- Applies ufunc to `sp_values` with scalar
- Computes new `fill_value` by applying ufunc to old fill_value and scalar
- Returns new SparseArray with updated values
- **Maintains sparsity** - no dense conversion needed

#### Sparse-Sparse Aligned Indices (Lines 1878-1895)
- Checks if indices are aligned using `sp_index.equals()`
- When aligned: applies ufunc element-wise to `sp_values` where indices overlap
- Computes fill_value by applying ufunc to both fill_values
- Returns new SparseArray maintaining sparsity
- **Maintains sparsity** - no dense conversion needed

#### Sparse-Sparse Misaligned Indices (Lines 1896-1902)
- Uses dense fallback for Phase 1
- Converts both arrays to dense, applies ufunc, returns as SparseArray
- Includes TODO comment for Phase 2 optimization
- Documents this limitation

#### Sparse-Dense Operations (Lines 1904-1916)
- Converts dense array to SparseArray with same fill_value as self
- Applies sparse-sparse logic by recursing
- Efficient approach that can maintain sparsity when possible

### 3. Routing Logic in `__array_ufunc__` (Lines 1744-1748)
Added check before the unary operation handler:
```python
# Handle min/max ufuncs with dedicated sparse-aware logic
if ufunc in MIN_MAX_UFUNCS and len(inputs) == 2:
    result = self._handle_minmax_ufunc(ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result
```

## NaN Semantics
The implementation correctly handles NaN semantics automatically through NumPy's ufuncs:
- `np.minimum(x, np.nan)` → `np.nan` (propagates)
- `np.maximum(x, np.nan)` → `np.nan` (propagates)
- `np.fmin(x, np.nan)` → `x` (ignores NaN)
- `np.fmax(x, np.nan)` → `x` (ignores NaN)

## Performance Benefits
- **Sparse-scalar operations**: No dense conversion, O(nnz) complexity where nnz is number of non-zero elements
- **Aligned sparse-sparse**: No dense conversion, O(nnz) complexity
- **Misaligned sparse-sparse**: Dense fallback for Phase 1, will be optimized in Phase 2 (task #13)
- **Sparse-dense**: Converts to sparse first, then uses efficient sparse logic

## Success Criteria Met
✅ `np.minimum(sparse_arr, scalar)` maintains sparsity and correctness
✅ `np.fmax(sparse_arr, np.nan)` correctly ignores NaN per NumPy semantics
✅ `np.maximum(sparse_aligned, sparse_aligned)` works element-wise without dense conversion
✅ `np.fmin(sparse_misaligned, sparse_misaligned)` falls back to dense correctly
✅ All four ufuncs tested with: scalars, aligned sparse arrays, dense arrays, NaN values
✅ Performance benchmark shows no dense conversion for aligned sparse operations

## Code Quality
- Follows existing patterns from `_arith_method` and `_sparse_array_op`
- Uses `with np.errstate(all="ignore")` to handle potential warnings
- Uses `lib.item_from_zerodim` to convert fill_value properly
- Uses `self._simple_new` for efficient SparseArray creation
- Comprehensive docstring with proper parameter descriptions
- Clear comments explaining each code path

## Integration
- Integrates seamlessly with refactored binary ufunc handling (Task #5)
- Follows helper method pattern (Task #6)
- Ready for Phase 2 optimization of misaligned sparse-sparse operations (Task #13)

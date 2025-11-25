# Test Coverage Summary: Min/Max Ufunc Operations for Sparse Arrays

## Overview
This document summarizes the comprehensive test coverage added for min/max ufunc operations (`np.maximum`, `np.minimum`, `np.fmax`, `np.fmin`) with sparse arrays in pandas.

## File Modified
- **Location**: `pandas/tests/arrays/sparse/test_arithmetics.py`
- **Lines Added**: 143 lines of new test code (lines 447-590)

## Test Functions Added

### 1. `test_ufunc_minmax_sparse_sparse_same_fill(ufunc)`
**Purpose**: Test min/max ufuncs with sparse/sparse arrays having the same fill values.

**Parametrized over**: `[np.maximum, np.minimum, np.fmax, np.fmin]`

**Coverage**:
- ✅ Sparse array with fill_value=0 combined with another sparse array with fill_value=0
- ✅ Verifies result is a SparseArray instance
- ✅ Verifies sp_values correctness
- ✅ Verifies fill_value computation

**Test Case Example**:
```python
a = SparseArray([0, 1, 0, 2, 0], fill_value=0)
b = SparseArray([0, 3, 0, 1, 0], fill_value=0)
result = np.maximum(a, b)  # Should return SparseArray([0, 3, 0, 2, 0], fill_value=0)
```

### 2. `test_ufunc_minmax_sparse_sparse_different_fill(ufunc)`
**Purpose**: Test min/max ufuncs with sparse/sparse arrays having different fill values.

**Parametrized over**: `[np.maximum, np.minimum, np.fmax, np.fmin]`

**Coverage**:
- ✅ Sparse array with fill_value=0 combined with sparse array with fill_value=5
- ✅ Verifies result is a SparseArray instance
- ✅ Verifies sp_values correctness
- ✅ Verifies fill_value is computed correctly as ufunc(0, 5)

**Test Case Example**:
```python
a = SparseArray([0, 1, 0, 2, 0], fill_value=0)
b = SparseArray([5, 3, 5, 1, 5], fill_value=5)
result = np.maximum(a, b)  # Should return SparseArray with fill_value=max(0, 5)=5
```

### 3. `test_ufunc_minmax_sparse_scalar(ufunc)`
**Purpose**: Test min/max ufuncs with sparse array and scalar.

**Parametrized over**: `[np.maximum, np.minimum, np.fmax, np.fmin]`

**Coverage**:
- ✅ Scalar on right: `ufunc(sparse_array, scalar)`
- ✅ Scalar on left: `ufunc(scalar, sparse_array)`
- ✅ Verifies result is a SparseArray instance in both cases
- ✅ Verifies sp_values correctness
- ✅ Verifies fill_value updates correctly

**Test Case Examples**:
```python
arr = SparseArray([0, 1, 0, 2, 0], fill_value=0)
result = np.maximum(arr, 1)  # Scalar on right
result = np.maximum(1, arr)  # Scalar on left
```

### 4. `test_ufunc_minmax_sparse_dense(ufunc)`
**Purpose**: Test min/max ufuncs with sparse array and dense numpy array.

**Parametrized over**: `[np.maximum, np.minimum, np.fmax, np.fmin]`

**Coverage**:
- ✅ Dense on right: `ufunc(sparse_array, dense_array)`
- ✅ Dense on left: `ufunc(dense_array, sparse_array)`
- ✅ Verifies result is a SparseArray instance in both cases
- ✅ Verifies correctness of values

**Test Case Examples**:
```python
sparse_arr = SparseArray([0, 1, 0, 2, 0], fill_value=0)
dense_arr = np.array([1, 0, 2, 1, 3])
result = np.maximum(sparse_arr, dense_arr)  # Dense on right
result = np.maximum(dense_arr, sparse_arr)  # Dense on left
```

### 5. `test_ufunc_minmax_nan_propagation()`
**Purpose**: Test that maximum/minimum propagate NaN while fmax/fmin ignore NaN.

**Coverage**:
- ✅ Tests NaN in sp_values (non-fill positions)
- ✅ Verifies `maximum` and `minimum` propagate NaN (result contains NaN)
- ✅ Verifies `fmax` and `fmin` ignore NaN (result uses non-NaN value)
- ✅ Verifies all results are SparseArray instances

**Test Case Example**:
```python
a = SparseArray([0, 1, np.nan, 2, 0], fill_value=0)
b = SparseArray([0, 3, 1, 1, 0], fill_value=0)
np.maximum(a, b)[2]  # Should be NaN (propagated)
np.fmax(a, b)[2]     # Should be 1.0 (NaN ignored)
```

### 6. `test_ufunc_minmax_nan_in_fill_value()`
**Purpose**: Test min/max ufuncs with NaN as fill_value.

**Coverage**:
- ✅ Tests arrays with NaN as fill_value
- ✅ Tests both `maximum` and `fmax` operations
- ✅ Verifies results are SparseArray instances
- ✅ Verifies correctness compared to dense operations

**Test Case Example**:
```python
a = SparseArray([np.nan, 1, np.nan, 2, np.nan], fill_value=np.nan)
b = SparseArray([np.nan, 3, np.nan, 1, np.nan], fill_value=np.nan)
result = np.maximum(a, b)  # Works correctly with NaN fill values
```

### 7. `test_ufunc_minmax_both_nan()`
**Purpose**: Test min/max ufuncs when both inputs have NaN at the same position.

**Coverage**:
- ✅ Tests case where both arrays have NaN at identical positions
- ✅ Verifies `maximum` keeps NaN when both are NaN
- ✅ Verifies `fmax` keeps NaN when both are NaN (no non-NaN value to choose)
- ✅ Verifies results are SparseArray instances

**Test Case Example**:
```python
a = SparseArray([0, np.nan, 2, 0], fill_value=0)
b = SparseArray([1, np.nan, 1, 0], fill_value=0)
np.maximum(a, b)[1]  # Should be NaN
np.fmax(a, b)[1]     # Should be NaN (both are NaN)
```

## Success Criteria Verification

### ✅ All four ufuncs have test coverage
- `np.maximum` ✓
- `np.minimum` ✓
- `np.fmax` ✓
- `np.fmin` ✓

### ✅ Tests verify results are SparseArray instances
All test functions include explicit assertions:
```python
assert isinstance(result, SparseArray)
```

### ✅ Tests verify correctness of sp_values and fill_value
- Uses `tm.assert_sp_array_equal()` for complete SparseArray comparison
- Uses `tm.assert_numpy_array_equal()` for dense value comparison
- Explicit `fill_value` assertions where applicable

### ✅ Tests demonstrate correct NaN propagation behavior
- `test_ufunc_minmax_nan_propagation()` specifically tests this
- Verifies `maximum`/`minimum` propagate NaN
- Verifies `fmax`/`fmin` ignore NaN

### ✅ Tests cover all three input combination types
1. Sparse/sparse: ✓ (with same and different fill values)
2. Sparse/scalar: ✓ (scalar on both left and right)
3. Sparse/dense: ✓ (dense on both left and right)

### ✅ Tests include edge cases with different fill values
- `test_ufunc_minmax_sparse_sparse_different_fill()` explicitly tests this

## Testing Framework
- **Framework**: pytest with pandas test utilities
- **Assertions**: Uses `tm.assert_sp_array_equal()` and `tm.assert_numpy_array_equal()`
- **Parametrization**: Leverages `@pytest.mark.parametrize` for testing all four ufuncs

## Test Execution
To run these tests:
```bash
# Run all min/max tests
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_sparse_sparse_same_fill -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_sparse_sparse_different_fill -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_sparse_scalar -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_sparse_dense -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_nan_propagation -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_nan_in_fill_value -v
pytest pandas/tests/arrays/sparse/test_arithmetics.py::test_ufunc_minmax_both_nan -v

# Or run all tests in the file
pytest pandas/tests/arrays/sparse/test_arithmetics.py -v
```

## Dependencies
This test coverage depends on:
- The implementation of Python-based sparse handling for min/max ufuncs (completed in previous ticket)
- The `__array_ufunc__` method in `SparseArray` class

## Integration with Existing Tests
The new tests are integrated seamlessly with existing test patterns:
- Placed after existing ufunc tests (line 447)
- Follows the same naming conventions (`test_ufunc_*`)
- Uses the same testing utilities and assertion methods
- Maintains consistency with existing test structure

## Total Test Coverage
- **7 test functions** added
- **4 ufuncs** tested per parametrized function (where applicable)
- **Multiple scenarios** per test function
- **Total test cases**: 4×4 (parametrized) + 3 (specialized) = **19 test cases**

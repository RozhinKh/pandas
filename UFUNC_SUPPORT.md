# Extended NumPy Ufunc Support for Sparse Arrays

## Overview

The `SparseArray` class now supports a comprehensive set of NumPy universal functions (ufuncs) beyond the basic arithmetic operations. This document describes the supported operations and their behavior with sparse arrays.

## Implementation

The ufunc support is implemented through the `__array_ufunc__` method in `pandas/core/arrays/sparse/array.py` (lines 1712-1776). The implementation handles:

1. **Binary ufuncs**: Dispatched to dunder methods (e.g., `__add__`, `__pow__`) via `maybe_dispatch_ufunc_to_dunder_op`
2. **Unary ufuncs**: Handled by the generic unary path (lines 1741-1761) which:
   - Applies the ufunc to both sparse values and fill_value
   - Creates a new SparseArray with the transformed values
   - Automatically handles sparsity preservation

## Supported Ufuncs

### Power Operations
- `np.sqrt` - Square root (preserves sparsity with fill_value=0)
- `np.square` - Element-wise square (preserves sparsity with fill_value=0)
- `np.power` - Power function (binary operation)

**Example:**
```python
arr = SparseArray([0, 0, 1, 4, 0], fill_value=0)
result = np.sqrt(arr)  # [0, 0, 1, 2, 0], fill_value=0
```

### Exponential and Logarithmic Functions
- `np.exp` - Exponential (changes fill_value: exp(0)=1)
- `np.expm1` - exp(x) - 1
- `np.log` - Natural logarithm
- `np.log10` - Base-10 logarithm
- `np.log2` - Base-2 logarithm
- `np.log1p` - log(1 + x)

**Example:**
```python
arr = SparseArray([0, 0, 1, 2, 0], fill_value=0)
result = np.exp(arr)  # [1, 1, e, e², 1], fill_value=1
```

**Note:** Logarithmic functions should be used with appropriate input values to avoid undefined behavior (e.g., log of negative numbers).

### Trigonometric Functions
- `np.sin` - Sine (preserves sparsity: sin(0)=0)
- `np.cos` - Cosine (changes fill_value: cos(0)=1)
- `np.tan` - Tangent (preserves sparsity: tan(0)=0)
- `np.arcsin` - Inverse sine (preserves sparsity: arcsin(0)=0)
- `np.arccos` - Inverse cosine (changes fill_value: arccos(0)=π/2)
- `np.arctan` - Inverse tangent (preserves sparsity: arctan(0)=0)

**Example:**
```python
arr = SparseArray([0, 0, 0.5, -0.5, 0], fill_value=0)
result = np.sin(arr)  # [0, 0, sin(0.5), sin(-0.5), 0], fill_value=0
```

### Hyperbolic Functions
- `np.sinh` - Hyperbolic sine (preserves sparsity: sinh(0)=0)
- `np.cosh` - Hyperbolic cosine (changes fill_value: cosh(0)=1)
- `np.tanh` - Hyperbolic tangent (preserves sparsity: tanh(0)=0)
- `np.arcsinh` - Inverse hyperbolic sine (preserves sparsity)
- `np.arccosh` - Inverse hyperbolic cosine (requires input ≥ 1)
- `np.arctanh` - Inverse hyperbolic tangent (preserves sparsity)

**Example:**
```python
arr = SparseArray([0, 0, 1, 2, 0], fill_value=0)
result = np.sinh(arr)  # [0, 0, sinh(1), sinh(2), 0], fill_value=0
```

### Rounding Functions
- `np.floor` - Floor function
- `np.ceil` - Ceiling function
- `np.trunc` - Truncate to integer
- `np.rint` - Round to nearest integer
- `np.round` - Round with specified decimals

**Example:**
```python
arr = SparseArray([0, 0, 1.7, 2.3, 0], fill_value=0)
result = np.floor(arr)  # [0, 0, 1, 2, 0], fill_value=0
```

### Sign and Absolute Value
- `np.sign` - Sign function (preserves sparsity: sign(0)=0)
- `np.absolute` (or `np.abs`) - Absolute value

**Example:**
```python
arr = SparseArray([0, 0, 1, -2, 0], fill_value=0)
result = np.sign(arr)  # [0, 0, 1, -1, 0], fill_value=0
```

## Sparsity Preservation

### Operations that Preserve Sparsity (with fill_value=0)
When the fill_value is 0 and remains 0 after the operation, the sparse structure is maintained efficiently:

- `np.sqrt`, `np.square`
- `np.sin`, `np.tan`, `np.sinh`, `np.tanh`
- `np.arcsin`, `np.arctan`, `np.arcsinh`, `np.arctanh`
- `np.floor`, `np.ceil`, `np.trunc`, `np.rint` (when applied to 0)
- `np.sign`

### Operations that Change Fill Value
When a ufunc changes the fill_value, the sparse array still works correctly but represents a different sparse pattern:

- `np.exp`: 0 → 1
- `np.cos`: 0 → 1
- `np.cosh`: 0 → 1
- `np.arccos`: 0 → π/2
- `np.log` (with fill_value=1): 1 → 0

**Example:**
```python
# Original: [0, 0, 1, 2, 0] with fill_value=0 (3 non-zero values stored)
arr = SparseArray([0, 0, 1, 2, 0], fill_value=0)

# After exp: [1, 1, e, e², 1] with fill_value=1 (still 2 non-one values stored)
result = np.exp(arr)
assert result.fill_value == 1.0
assert len(result.sp_values) == 2  # Only stores [e, e²]
```

## Test Coverage

Comprehensive tests have been added to `pandas/tests/arrays/sparse/test_arithmetics.py`:

1. `test_unary_ufuncs` - Tests all unary ufuncs with fill_value=0
2. `test_unary_ufuncs_with_nan_fill` - Tests with fill_value=nan
3. `test_inverse_trig_ufuncs` - Tests inverse trig functions with appropriate ranges
4. `test_arccosh_ufunc` - Tests arccosh with input ≥ 1
5. `test_log_ufuncs_positive_values` - Tests logarithmic functions
6. `test_power_ufunc_binary` - Tests binary power operation
7. `test_ufunc_preserves_sparsity` - Verifies sparsity preservation
8. `test_ufunc_changes_fill_value` - Verifies correct behavior when fill_value changes
9. `test_round_function` - Tests rounding with decimals argument

## Performance Considerations

- **Sparsity-preserving operations** maintain optimal memory usage and performance
- **Fill-value-changing operations** still maintain sparsity but with a different fill value
- No unnecessary conversion to dense arrays - all operations work directly on sparse values
- The sparse structure efficiently handles both types of operations

## Edge Cases

1. **Invalid inputs**: Operations like `np.log(0)` will produce `-inf`, `np.sqrt(-1)` will produce `nan`
2. **Domain restrictions**: `np.arccosh` requires input ≥ 1, `np.arctanh` requires |input| < 1
3. **NaN propagation**: NaN values are handled correctly throughout all operations
4. **Multiple outputs**: Ufuncs like `np.modf` that return multiple outputs are supported

## Usage Examples

```python
import numpy as np
from pandas.core.arrays.sparse import SparseArray

# Create a sparse array
arr = SparseArray([0, 0, 1, 2, 0, 3, 0], fill_value=0)

# Power operations
sqrt_result = np.sqrt(arr)
square_result = np.square(arr)
power_result = np.power(arr, 2)

# Trigonometric
sin_result = np.sin(arr)
cos_result = np.cos(arr)  # Note: fill_value becomes 1

# Exponential
exp_result = np.exp(arr)  # Note: fill_value becomes 1

# Logarithmic (use appropriate fill_value)
arr_log = SparseArray([1, 1, 2, 3, 1], fill_value=1)
log_result = np.log(arr_log)  # fill_value becomes 0

# Rounding
arr_float = SparseArray([0, 0, 1.7, 2.3, 0], fill_value=0)
floor_result = np.floor(arr_float)
ceil_result = np.ceil(arr_float)

# Sign
arr_signed = SparseArray([0, 0, 1, -2, 0, 3], fill_value=0)
sign_result = np.sign(arr_signed)
```

## Backward Compatibility

All existing ufunc behavior is preserved. The extended support builds on the existing infrastructure without breaking any existing functionality. Previously working code will continue to work exactly as before.

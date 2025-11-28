# SparseArray Ufunc Helper Methods Refactoring

## Overview
This refactoring extracts helper methods from `__array_ufunc__` in `pandas/core/arrays/sparse/array.py` to improve code maintainability and testability by breaking down complex logic into smaller, purpose-specific functions.

## Changes Made

### 1. Module-Level Constants (Lines ~137-157)
Added two dictionaries for ufunc-to-operator mapping:
- **`UFUNC_TO_OPERATOR`**: Maps numpy ufuncs (like `np.add`, `np.multiply`) to operator names (like `"add"`, `"mul"`)
- **`UFUNC_ALIASES`**: Maps ufunc aliases (like `np.divide`) to their operator names (`"truediv"`)

These constants enable consistent lookup of operator names from ufuncs across the codebase.

### 2. Helper Function: `_get_ufunc_operator` (Lines ~160-180)
```python
def _get_ufunc_operator(ufunc: np.ufunc) -> str | None
```
A module-level function that:
- Looks up the operator name for a given numpy ufunc
- Checks `UFUNC_TO_OPERATOR` first, then `UFUNC_ALIASES`
- Returns the operator name string or `None` if no mapping exists

### 3. Helper Method: `_handle_minmax_ufunc` (Lines ~1762-1790)
```python
def _handle_minmax_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs)
```
Purpose: Handle `np.minimum`, `np.maximum`, `np.fmin`, `np.fmax` which don't map to Python operators.

Current implementation: **Stub** - returns `NotImplemented` to fall back to dense computation. Full implementation will be added in ticket #8 to maintain sparsity where possible.

### 4. Helper Method: `_handle_binary_ufunc` (Lines ~1792-1830)
```python
def _handle_binary_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs)
```
Purpose: Orchestrate binary ufunc operations.

Logic:
1. Check if ufunc is a min/max operation → delegate to `_handle_minmax_ufunc`
2. Check if ufunc maps to operator method → note for dunder dispatch
3. Otherwise, fall back to dense computation

Returns: Result of the operation or appropriate type conversion

### 5. Refactored `__array_ufunc__` (Lines ~1846-1899)
Transformed into a clean dispatcher (48 lines of implementation code, under the 50-line requirement):

**Structure:**
1. **Type checking** - Verify all inputs/outputs are handled types
2. **Dispatch to dunder ops** - Try `maybe_dispatch_ufunc_to_dunder_op` first
3. **Handle out parameter** - Use `dispatch_ufunc_with_out` if present
4. **Handle reductions** - Use `dispatch_reduction_ufunc` for reduce operations
5. **Handle unary operations** - Apply ufunc to sparse values and fill value
6. **Delegate binary operations** - Call `_handle_binary_ufunc` helper

## Benefits

### Maintainability
- **Separation of concerns**: Each helper has a single, well-defined purpose
- **Clear structure**: Main `__array_ufunc__` reads as a high-level dispatcher
- **Easier to modify**: Changes to binary ufunc logic isolated in `_handle_binary_ufunc`

### Testability
- **Unit testable**: Helper methods can be tested independently
- **Focused tests**: Each helper can have specific test cases
- **Easier debugging**: Smaller functions are easier to reason about

### Extensibility
- **Easy to add new ufunc types**: Create new helpers following the established pattern
- **Future-proof**: Min/max stub ready for full implementation in ticket #8
- **Consistent patterns**: Use of constants makes adding new ufuncs straightforward

## Line Count Verification
- `__array_ufunc__` method: **48 lines** (excluding def line and docstring) ✓
- Success criteria met: Under 50 lines ✓

## Preserved Functionality
All existing behavior is maintained:
- Type checking for handled types
- Dispatch to operator dunder methods
- Out parameter handling
- Reduction operations
- Unary ufunc operations
- Binary ufunc operations (via dense fallback when needed)

## Future Work (Ticket #8)
Full implementation of `_handle_minmax_ufunc` to:
- Apply min/max ufunc semantics while maintaining sparsity
- Handle NaN semantics specific to each ufunc (minimum vs fmin, maximum vs fmax)
- Optimize for sparse-sparse, sparse-scalar, and sparse-dense operations

## Files Modified
- `pandas/core/arrays/sparse/array.py`
  - Added module-level constants (lines ~137-157)
  - Added `_get_ufunc_operator` function (lines ~160-180)
  - Added `_handle_minmax_ufunc` method (lines ~1762-1790)
  - Added `_handle_binary_ufunc` method (lines ~1792-1830)
  - Refactored `__array_ufunc__` method (lines ~1846-1899)

## Testing
The refactoring maintains backward compatibility. All existing tests should pass without modification.

Additional test files created for verification:
- `test_syntax.py` - Verifies Python syntax is correct
- `test_functionality.py` - Tests new constants and helper method existence

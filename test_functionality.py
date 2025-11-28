#!/usr/bin/env python
"""Simple functional test for sparse array ufunc changes."""
import sys
import numpy as np

# Add the repo to path
sys.path.insert(0, '.')

try:
    # Import the module
    from pandas.core.arrays.sparse.array import (
        UFUNC_TO_OPERATOR,
        UFUNC_ALIASES,
        _get_ufunc_operator,
        SparseArray
    )
    
    # Test module-level constants
    assert isinstance(UFUNC_TO_OPERATOR, dict), "UFUNC_TO_OPERATOR should be a dict"
    assert isinstance(UFUNC_ALIASES, dict), "UFUNC_ALIASES should be a dict"
    assert np.add in UFUNC_TO_OPERATOR, "np.add should be in UFUNC_TO_OPERATOR"
    print("✓ Module-level constants defined correctly")
    
    # Test _get_ufunc_operator function
    assert _get_ufunc_operator(np.add) == "add", "_get_ufunc_operator(np.add) should return 'add'"
    assert _get_ufunc_operator(np.divide) == "truediv", "_get_ufunc_operator(np.divide) should return 'truediv'"
    assert _get_ufunc_operator(np.sin) is None, "_get_ufunc_operator(np.sin) should return None"
    print("✓ _get_ufunc_operator works correctly")
    
    # Test SparseArray has the new methods
    arr = SparseArray([0, 1, 0, 2, 0])
    assert hasattr(arr, '_handle_minmax_ufunc'), "SparseArray should have _handle_minmax_ufunc"
    assert hasattr(arr, '_handle_binary_ufunc'), "SparseArray should have _handle_binary_ufunc"
    print("✓ Helper methods exist on SparseArray")
    
    # Test basic ufunc operations still work
    arr1 = SparseArray([1, 0, 3, 0])
    arr2 = SparseArray([2, 0, 1, 0])
    result = np.add(arr1, arr2)
    assert isinstance(result, SparseArray), "np.add should return SparseArray"
    expected = [3, 0, 4, 0]
    assert list(result) == expected, f"Result {list(result)} != expected {expected}"
    print("✓ Basic ufunc operations work")
    
    print("\n✅ All tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

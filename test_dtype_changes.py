#!/usr/bin/env python3
"""
Simple test to verify dtype promotion changes work correctly
"""
import numpy as np
import sys

# Test if the imports work
try:
    import pandas as pd
    print("✓ pandas imported successfully")
except Exception as e:
    print(f"✗ Failed to import pandas: {e}")
    sys.exit(1)

# Test basic dtype promotion
try:
    arr_int = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int8)
    arr_float = pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5], dtype=np.float32)
    
    result = np.add(arr_int, arr_float)
    
    print(f"✓ Int8 + Float32 = {result.dtype.subtype}")
    print(f"✓ Result fill_value type: {type(result.fill_value)}")
    
    # Check dtype promotion
    assert result.dtype.subtype == np.float32, f"Expected float32, got {result.dtype.subtype}"
    print("✓ Dtype promotion works correctly")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All basic tests passed!")

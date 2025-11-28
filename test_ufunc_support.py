#!/usr/bin/env python3
"""Quick test to verify sparse ufunc support works correctly."""

import numpy as np
import sys
sys.path.insert(0, '.')

from pandas.core.arrays.sparse import SparseArray

def test_basic_ufuncs():
    """Test that basic unary ufuncs work with sparse arrays."""
    arr = SparseArray([0, 0, 1, 2, 0, 3], fill_value=0)
    
    # Test power operations
    print("Testing sqrt...")
    result = np.sqrt(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ sqrt works: {result.to_dense()}")
    
    print("Testing square...")
    result = np.square(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ square works: {result.to_dense()}")
    
    # Test exponential functions
    print("Testing exp...")
    result = np.exp(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == np.exp(0.0)
    print(f"  ✓ exp works, fill_value changed to {result.fill_value}")
    
    print("Testing expm1...")
    result = np.expm1(arr)
    assert isinstance(result, SparseArray)
    print(f"  ✓ expm1 works: {result.to_dense()}")
    
    # Test trigonometric functions
    print("Testing sin...")
    result = np.sin(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ sin works: {result.to_dense()}")
    
    print("Testing cos...")
    result = np.cos(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == np.cos(0.0)
    print(f"  ✓ cos works, fill_value changed to {result.fill_value}")
    
    print("Testing tan...")
    result = np.tan(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ tan works: {result.to_dense()}")
    
    # Test hyperbolic functions
    print("Testing sinh...")
    result = np.sinh(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ sinh works: {result.to_dense()}")
    
    print("Testing cosh...")
    result = np.cosh(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == np.cosh(0.0)
    print(f"  ✓ cosh works, fill_value changed to {result.fill_value}")
    
    print("Testing tanh...")
    result = np.tanh(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ tanh works: {result.to_dense()}")
    
    # Test rounding functions
    arr_float = SparseArray([0, 0, 1.7, 2.3, 0], fill_value=0)
    
    print("Testing floor...")
    result = np.floor(arr_float)
    assert isinstance(result, SparseArray)
    print(f"  ✓ floor works: {result.to_dense()}")
    
    print("Testing ceil...")
    result = np.ceil(arr_float)
    assert isinstance(result, SparseArray)
    print(f"  ✓ ceil works: {result.to_dense()}")
    
    print("Testing trunc...")
    result = np.trunc(arr_float)
    assert isinstance(result, SparseArray)
    print(f"  ✓ trunc works: {result.to_dense()}")
    
    print("Testing rint...")
    result = np.rint(arr_float)
    assert isinstance(result, SparseArray)
    print(f"  ✓ rint works: {result.to_dense()}")
    
    # Test sign function
    arr_signed = SparseArray([0, 0, 1, -2, 0, 3], fill_value=0)
    print("Testing sign...")
    result = np.sign(arr_signed)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ sign works: {result.to_dense()}")
    
    # Test inverse trig functions
    arr_small = SparseArray([0, 0, 0.5, -0.5, 0], fill_value=0)
    
    print("Testing arcsin...")
    result = np.arcsin(arr_small)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ arcsin works: {result.to_dense()}")
    
    print("Testing arccos...")
    result = np.arccos(arr_small)
    assert isinstance(result, SparseArray)
    print(f"  ✓ arccos works: {result.to_dense()}")
    
    print("Testing arctan...")
    result = np.arctan(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ arctan works: {result.to_dense()}")
    
    print("Testing arcsinh...")
    result = np.arcsinh(arr)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ arcsinh works: {result.to_dense()}")
    
    print("Testing arctanh...")
    result = np.arctanh(arr_small)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0
    print(f"  ✓ arctanh works: {result.to_dense()}")
    
    # Test log functions with appropriate values
    arr_pos = SparseArray([1, 1, 2, 3, 1], fill_value=1)
    
    print("Testing log...")
    with np.errstate(all='ignore'):
        result = np.log(arr_pos)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0  # log(1) = 0
    print(f"  ✓ log works: {result.to_dense()}")
    
    print("Testing log10...")
    with np.errstate(all='ignore'):
        result = np.log10(arr_pos)
    assert isinstance(result, SparseArray)
    print(f"  ✓ log10 works: {result.to_dense()}")
    
    print("Testing log2...")
    with np.errstate(all='ignore'):
        result = np.log2(arr_pos)
    assert isinstance(result, SparseArray)
    print(f"  ✓ log2 works: {result.to_dense()}")
    
    print("Testing log1p...")
    result = np.log1p(arr)
    assert isinstance(result, SparseArray)
    print(f"  ✓ log1p works: {result.to_dense()}")
    
    # Test arccosh (requires input >= 1)
    arr_ge1 = SparseArray([1, 1, 2, 3, 1], fill_value=1)
    print("Testing arccosh...")
    result = np.arccosh(arr_ge1)
    assert isinstance(result, SparseArray)
    assert result.fill_value == 0  # arccosh(1) = 0
    print(f"  ✓ arccosh works: {result.to_dense()}")
    
    # Test binary power
    print("Testing np.power (binary)...")
    arr = SparseArray([0, 0, 1, 2, 0, 3], fill_value=0)
    result = np.power(arr, 2)
    assert isinstance(result, SparseArray)
    print(f"  ✓ power works: {result.to_dense()}")
    
    print("\n✅ All ufunc tests passed!")


if __name__ == "__main__":
    test_basic_ufuncs()

import re
import warnings

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", 0),
        ("float", np.nan),
        ("bool", False),
        ("object", np.nan),
        ("datetime64[ns]", np.datetime64("NaT", "ns")),
        ("timedelta64[ns]", np.timedelta64("NaT", "ns")),
    ],
)
def test_inferred_dtype(dtype, fill_value):
    sparse_dtype = SparseDtype(dtype)
    result = sparse_dtype.fill_value
    if pd.isna(fill_value):
        assert pd.isna(result) and type(result) == type(fill_value)
    else:
        assert result == fill_value


def test_from_sparse_dtype():
    dtype = SparseDtype("float", 0)
    result = SparseDtype(dtype)
    assert result.fill_value == 0


def test_from_sparse_dtype_fill_value():
    dtype = SparseDtype("int", 1)
    result = SparseDtype(dtype, fill_value=2)
    expected = SparseDtype("int", 2)
    assert result == expected


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", None),
        ("float", None),
        ("bool", None),
        ("object", None),
        ("datetime64[ns]", None),
        ("timedelta64[ns]", None),
        ("int", np.nan),
        ("float", 0),
    ],
)
def test_equal(dtype, fill_value):
    a = SparseDtype(dtype, fill_value)
    b = SparseDtype(dtype, fill_value)
    assert a == b
    assert b == a


def test_nans_equal():
    a = SparseDtype(float, float("nan"))
    b = SparseDtype(float, np.nan)
    assert a == b
    assert b == a


def test_nans_not_equal():
    # GH 54770
    a = SparseDtype(float, 0)
    b = SparseDtype(float, pd.NA)
    assert a != b
    assert b != a


with warnings.catch_warnings():
    msg = "Allowing arbitrary scalar fill_value in SparseDtype is deprecated"
    warnings.filterwarnings("ignore", msg, category=FutureWarning)

    tups = [
        (SparseDtype("float64"), SparseDtype("float32")),
        (SparseDtype("float64"), SparseDtype("float64", 0)),
        (SparseDtype("float64"), SparseDtype("datetime64[ns]", np.nan)),
        (SparseDtype("float64"), np.dtype("float64")),
    ]


@pytest.mark.parametrize(
    "a, b",
    tups,
)
def test_not_equal(a, b):
    assert a != b


def test_construct_from_string_raises():
    with pytest.raises(
        TypeError, match="Cannot construct a 'SparseDtype' from 'not a dtype'"
    ):
        SparseDtype.construct_from_string("not a dtype")


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (int, True),
        (float, True),
        (bool, True),
        (object, False),
        (str, False),
    ],
)
def test_is_numeric(dtype, expected):
    assert SparseDtype(dtype)._is_numeric is expected


def test_str_uses_object():
    result = SparseDtype(str).subtype
    assert result == np.dtype("object")


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[float64]", SparseDtype(np.dtype("float64"))),
        ("Sparse[float32]", SparseDtype(np.dtype("float32"))),
        ("Sparse[int]", SparseDtype(np.dtype("int"))),
        ("Sparse[str]", SparseDtype(np.dtype("str"))),
        ("Sparse[datetime64[ns]]", SparseDtype(np.dtype("datetime64[ns]"))),
        ("Sparse", SparseDtype(np.dtype("float"), np.nan)),
    ],
)
def test_construct_from_string(string, expected):
    result = SparseDtype.construct_from_string(string)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (SparseDtype(float, 0.0), SparseDtype(np.dtype("float"), 0.0), True),
        (SparseDtype(int, 0), SparseDtype(int, 0), True),
        (SparseDtype(float, float("nan")), SparseDtype(float, np.nan), True),
        (SparseDtype(float, 0), SparseDtype(float, np.nan), False),
        (SparseDtype(int, 0.0), SparseDtype(float, 0.0), False),
    ],
)
def test_hash_equal(a, b, expected):
    result = a == b
    assert result is expected

    result = hash(a) == hash(b)
    assert result is expected


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[int]", "int"),
        ("Sparse[int, 0]", "int"),
        ("Sparse[int64]", "int64"),
        ("Sparse[int64, 0]", "int64"),
        ("Sparse[datetime64[ns], 0]", "datetime64[ns]"),
    ],
)
def test_parse_subtype(string, expected):
    subtype, _ = SparseDtype._parse_subtype(string)
    assert subtype == expected


@pytest.mark.parametrize(
    "string", ["Sparse[int, 1]", "Sparse[float, 0.0]", "Sparse[bool, True]"]
)
def test_construct_from_string_fill_value_raises(string):
    with pytest.raises(TypeError, match="fill_value in the string is not"):
        SparseDtype.construct_from_string(string)


@pytest.mark.parametrize(
    "original, dtype, expected",
    [
        (SparseDtype(int, 0), float, SparseDtype(float, 0.0)),
        (SparseDtype(int, 1), float, SparseDtype(float, 1.0)),
        (SparseDtype(int, 1), np.str_, SparseDtype(object, "1")),
        (SparseDtype(float, 1.5), int, SparseDtype(int, 1)),
    ],
)
def test_update_dtype(original, dtype, expected):
    result = original.update_dtype(dtype)
    assert result == expected


@pytest.mark.parametrize(
    "original, dtype, expected_error_msg",
    [
        (
            SparseDtype(float, np.nan),
            int,
            re.escape("Cannot convert non-finite values (NA or inf) to integer"),
        ),
        (
            SparseDtype(str, "abc"),
            int,
            r"invalid literal for int\(\) with base 10: ('abc'|np\.str_\('abc'\))",
        ),
    ],
)
def test_update_dtype_raises(original, dtype, expected_error_msg):
    with pytest.raises(ValueError, match=expected_error_msg):
        original.update_dtype(dtype)


def test_repr():
    # GH-34352
    result = str(SparseDtype("int64", fill_value=0))
    expected = "Sparse[int64, 0]"
    assert result == expected

    result = str(SparseDtype(object, fill_value="0"))
    expected = "Sparse[object, '0']"
    assert result == expected


def test_sparse_dtype_subtype_must_be_numpy_dtype():
    # GH#53160
    msg = "SparseDtype subtype must be a numpy dtype"
    with pytest.raises(TypeError, match=msg):
        SparseDtype("category", fill_value="c")


class TestDtypePromotion:
    """Test dtype preservation and promotion in sparse ufunc operations."""

    def test_int_plus_float_promotion(self):
        # Integer + Float → Float
        arr_int = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int8)
        arr_float = pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5], dtype=np.float32)
        
        result = np.add(arr_int, arr_float)
        assert result.dtype.subtype == np.float32
        assert result.fill_value == 0.0
        assert isinstance(result.fill_value, (float, np.floating))

    def test_int8_plus_int64_promotion(self):
        # Mixed integer sizes → larger integer
        arr_int8 = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int8)
        arr_int64 = pd.arrays.SparseArray([0, 10, 0, 20], dtype=np.int64)
        
        result = np.add(arr_int8, arr_int64)
        assert result.dtype.subtype == np.int64

    def test_real_plus_complex_promotion(self):
        # Real + Complex → Complex
        arr_float = pd.arrays.SparseArray([0.0, 1.0, 0.0, 2.0], dtype=np.float64)
        arr_complex = pd.arrays.SparseArray([0j, 1+2j, 0j, 3+4j], dtype=np.complex128)
        
        result = np.add(arr_float, arr_complex)
        assert result.dtype.subtype == np.complex128

    def test_unsigned_signed_promotion(self):
        # Unsigned + signed integers
        arr_uint = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.uint8)
        arr_int = pd.arrays.SparseArray([0, -1, 0, -2], dtype=np.int8)
        
        result = np.add(arr_uint, arr_int)
        # NumPy promotes uint8 + int8 to int16
        assert result.dtype.subtype in (np.int16, np.int32, np.int64)

    def test_scalar_int_plus_float_array(self):
        # Scalar int + float array → float
        arr = pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5], dtype=np.float64)
        result = arr + 1  # int scalar
        
        assert result.dtype.subtype == np.float64
        assert result.fill_value == 1.0
        assert isinstance(result.fill_value, (float, np.floating))

    def test_fill_value_dtype_matches_sp_values(self):
        # Ensure fill_value.dtype == sp_values.dtype after operations
        arr1 = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int32, fill_value=0)
        arr2 = pd.arrays.SparseArray([0.0, 1.0, 0.0, 2.0], dtype=np.float64, fill_value=0.0)
        
        result = np.add(arr1, arr2)
        
        # Result should be float64
        assert result.dtype.subtype == np.float64
        # fill_value should also be float64
        assert isinstance(result.fill_value, (float, np.floating))
        # Check that fill_value can be represented in result dtype
        assert np.asarray(result.fill_value, dtype=result.dtype.subtype).dtype == result.dtype.subtype

    def test_explicit_dtype_kwarg(self):
        # Test explicit dtype= parameter in ufuncs
        arr1 = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int32)
        arr2 = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int32)
        
        result = np.add(arr1, arr2, dtype=np.float32)
        
        assert result.dtype.subtype == np.float32
        assert isinstance(result.fill_value, (float, np.floating))

    def test_unary_ufunc_with_dtype(self):
        # Test unary ufunc with explicit dtype
        arr = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int32)
        
        result = np.negative(arr, dtype=np.float64)
        
        assert result.dtype.subtype == np.float64
        assert isinstance(result.fill_value, (float, np.floating))

    def test_mixed_precision_floats(self):
        # Different float precisions
        arr_f16 = pd.arrays.SparseArray([0.0, 1.0, 0.0, 2.0], dtype=np.float16)
        arr_f64 = pd.arrays.SparseArray([0.0, 1.0, 0.0, 2.0], dtype=np.float64)
        
        result = np.add(arr_f16, arr_f64)
        assert result.dtype.subtype == np.float64

    def test_operation_with_empty_array(self):
        # Edge case: operations with empty arrays
        arr1 = pd.arrays.SparseArray([], dtype=np.int32)
        arr2 = pd.arrays.SparseArray([], dtype=np.float64)
        
        result = np.add(arr1, arr2)
        assert result.dtype.subtype == np.float64

    def test_all_fill_value_array_operation(self):
        # Edge case: all elements are fill_value
        arr1 = pd.arrays.SparseArray([0, 0, 0, 0], dtype=np.int8, fill_value=0)
        arr2 = pd.arrays.SparseArray([0.0, 0.0, 0.0, 0.0], dtype=np.float32, fill_value=0.0)
        
        result = np.add(arr1, arr2)
        assert result.dtype.subtype == np.float32
        assert result.fill_value == 0.0

    def test_dtype_promotion_minimum_maximum(self):
        # Test dtype promotion for minimum/maximum operations
        arr_int = pd.arrays.SparseArray([0, 1, 0, 2], dtype=np.int32)
        arr_float = pd.arrays.SparseArray([0.0, 1.5, 0.0, 2.5], dtype=np.float64)
        
        result = np.minimum(arr_int, arr_float)
        assert result.dtype.subtype == np.float64
        
        result = np.maximum(arr_int, arr_float)
        assert result.dtype.subtype == np.float64

import operator

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.fixture(params=["integer", "block"])
def kind(request):
    """kind kwarg to pass to SparseArray"""
    return request.param


@pytest.fixture(params=[True, False])
def mix(request):
    """
    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    """
    return request.param


class TestSparseArrayArithmetics:
    def _assert(self, a, b):
        # We have to use tm.assert_sp_array_equal. See GH #45126
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
        # Check that arithmetic behavior matches non-Sparse Series arithmetic

        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values
        else:
            raise NotImplementedError

        with np.errstate(invalid="ignore", divide="ignore"):
            if mix:
                result = op(a, b_dense).to_dense()
            else:
                result = op(a, b).to_dense()

        self._assert(result, expected)

    def _check_bool_result(self, res):
        assert isinstance(res, SparseArray)
        assert isinstance(res.dtype, SparseDtype)
        assert res.dtype.subtype == np.bool_
        assert isinstance(res.fill_value, bool)

    def _check_comparison_ops(self, a, b, a_dense, b_dense):
        with np.errstate(invalid="ignore"):
            # Unfortunately, trying to wrap the computation of each expected
            # value is with np.errstate() is too tedious.
            #
            # sparse & sparse
            self._check_bool_result(a == b)
            self._assert((a == b).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b)
            self._assert((a != b).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b)
            self._assert((a >= b).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b)
            self._assert((a <= b).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b)
            self._assert((a > b).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b)
            self._assert((a < b).to_dense(), a_dense < b_dense)

            # sparse & dense
            self._check_bool_result(a == b_dense)
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b_dense)
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b_dense)
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b_dense)
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b_dense)
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b_dense)
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(self, a, b, a_dense, b_dense):
        # sparse & sparse
        self._check_bool_result(a & b)
        self._assert((a & b).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b)
        self._assert((a | b).to_dense(), a_dense | b_dense)
        # sparse & dense
        self._check_bool_result(a & b_dense)
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b_dense)
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize("scalar", [0, 1, 3])
    @pytest.mark.parametrize("fill_value", [None, 0, 2])
    def test_float_scalar(
        self, kind, mix, all_arithmetic_functions, fill_value, scalar, request
    ):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        self._check_numeric_ops(a, scalar, values, scalar, mix, op)

    def test_float_scalar_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])

        a = SparseArray(values, kind=kind)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=0)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=2)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

    def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
        # when sp_index are the same
        op = all_arithmetic_functions

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_with_nans(
        self, kind, mix, all_arithmetic_functions, request
    ):
        # when sp_index are the same
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_comparison(self, kind):
        # when sp_index are the same
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_float_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_different_kind(self, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind="integer")
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block", fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=1)
        b = SparseArray(rvalues, kind="block", fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_comparison(self, kind):
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_int_array(self, kind, mix, all_arithmetic_functions):
        op = all_arithmetic_functions

        # have to specify dtype explicitly until fixing GH 667
        dtype = np.int64

        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind):
        dtype = "int64"
        # int32 NI ATM

        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        a = SparseArray(values, dtype=dtype, kind=kind)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_same_index(self, kind, fill_value):
        # GH 14000
        # when sp_index are the same
        values = np.array([True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, True, True], dtype=np.bool_)

        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_array_logical(self, kind, fill_value):
        # GH 14000
        # when sp_index are the same
        values = np.array([True, False, True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)

        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)
        self._check_logical_ops(a, b, values, rvalues)

    def test_mixed_array_float_int(self, kind, mix, all_arithmetic_functions, request):
        op = all_arithmetic_functions
        rdtype = "int64"
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_mixed_array_comparison(self, kind):
        rdtype = "int64"
        # int32 NI ATM

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)

        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        assert b.dtype == SparseDtype(rdtype)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_xor(self):
        s = SparseArray([True, True, False, False])
        t = SparseArray([True, False, True, False])
        result = s ^ t
        sp_index = pd.core.arrays.sparse.IntIndex(4, np.array([0, 1, 2], dtype="int32"))
        expected = SparseArray([False, True, True], sparse_index=sp_index)
        tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("op", [operator.eq, operator.add])
def test_with_list(op):
    arr = SparseArray([0, 1], fill_value=0)
    result = op(arr, [0, 1])
    expected = op(arr, SparseArray([0, 1]))
    tm.assert_sp_array_equal(result, expected)


def test_with_dataframe():
    # GH#27910
    arr = SparseArray([0, 1], fill_value=0)
    df = pd.DataFrame([[1, 2], [3, 4]])
    result = arr.__add__(df)
    assert result is NotImplemented


def test_with_zerodim_ndarray():
    # GH#27910
    arr = SparseArray([0, 1], fill_value=0)

    result = arr * np.array(2)
    expected = arr * 2
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.abs, np.exp])
@pytest.mark.parametrize(
    "arr", [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])]
)
def test_ufuncs(ufunc, arr):
    result = ufunc(arr)
    fill_value = ufunc(arr.fill_value)
    expected = SparseArray(ufunc(np.asarray(arr)), fill_value=fill_value)
    tm.assert_sp_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        (SparseArray([0, 0, 0]), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
    ],
)
@pytest.mark.parametrize("ufunc", [np.add, np.greater])
def test_binary_ufuncs(ufunc, a, b):
    # can't say anything about fill value here.
    result = ufunc(a, b)
    expected = ufunc(np.asarray(a), np.asarray(b))
    assert isinstance(result, SparseArray)
    tm.assert_numpy_array_equal(np.asarray(result), expected)


def test_ndarray_inplace():
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    ndarray += sparray
    expected = np.array([0, 3, 2, 3])
    tm.assert_numpy_array_equal(ndarray, expected)


def test_sparray_inplace():
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    sparray += ndarray
    expected = SparseArray([0, 3, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(sparray, expected)


@pytest.mark.parametrize("cons", [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons):
    left = SparseArray([True, True])
    right = cons([True, True, True])
    with pytest.raises(ValueError, match="operands have mismatched length"):
        left & right


@pytest.mark.parametrize(
    "a, b",
    [
        ([0, 1, 2], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 1, 2]),
    ],
)
def test_mismatched_length_arith_op(a, b, all_arithmetic_functions):
    op = all_arithmetic_functions
    with pytest.raises(AssertionError, match=f"length mismatch: {len(a)} vs. {len(b)}"):
        op(SparseArray(a, fill_value=0), np.array(b))


@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "floordiv", "pow"])
@pytest.mark.parametrize("fill_value", [np.nan, 3])
def test_binary_operators(op, fill_value):
    op = getattr(operator, op)
    data1 = np.random.default_rng(2).standard_normal(20)
    data2 = np.random.default_rng(2).standard_normal(20)

    data1[::2] = fill_value
    data2[::3] = fill_value

    first = SparseArray(data1, fill_value=fill_value)
    second = SparseArray(data2, fill_value=fill_value)

    with np.errstate(all="ignore"):
        res = op(first, second)
        exp = SparseArray(
            op(first.to_dense(), second.to_dense()), fill_value=first.fill_value
        )
        assert isinstance(res, SparseArray)
        tm.assert_almost_equal(res.to_dense(), exp.to_dense())

        res2 = op(first, second.to_dense())
        assert isinstance(res2, SparseArray)
        tm.assert_sp_array_equal(res, res2)

        res3 = op(first.to_dense(), second)
        assert isinstance(res3, SparseArray)
        tm.assert_sp_array_equal(res, res3)

        res4 = op(first, 4)
        assert isinstance(res4, SparseArray)

        # Ignore this if the actual op raises (e.g. pow).
        try:
            exp = op(first.to_dense(), 4)
            exp_fv = op(first.fill_value, 4)
        except ValueError:
            pass
        else:
            tm.assert_almost_equal(res4.fill_value, exp_fv)
            tm.assert_almost_equal(res4.to_dense(), exp)


# Edge case tests for Phase 1 validation
def test_ufunc_empty_array():
    """Test ufuncs on empty SparseArray"""
    # Empty float array
    arr = SparseArray([], dtype=np.float64)
    result = np.abs(arr)
    expected = SparseArray([], dtype=np.float64)
    tm.assert_sp_array_equal(result, expected)
    
    # Empty int array
    arr = SparseArray([], dtype=np.int64, fill_value=0)
    result = np.abs(arr)
    expected = SparseArray([], dtype=np.int64, fill_value=0)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_all_sparse():
    """Test ufuncs on arrays with no fill_value elements (all sparse)"""
    # All elements are non-zero (sparse)
    arr = SparseArray([1, 2, 3], fill_value=0)
    result = np.abs(arr)
    expected = SparseArray([1, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(result, expected)
    
    # Negative transformation
    result = np.negative(arr)
    expected = SparseArray([-1, -2, -3], fill_value=0)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_dtype_promotion():
    """Test dtype promotion in ufunc operations"""
    # int + float should promote to float
    int_arr = SparseArray([0, 1, 0, 2], dtype=np.int64, fill_value=0)
    float_scalar = 1.5
    
    result = np.add(int_arr, float_scalar)
    expected = SparseArray([1.5, 2.5, 1.5, 3.5], fill_value=1.5)
    tm.assert_sp_array_equal(result, expected)
    assert result.dtype.subtype == np.float64


def test_ufunc_nan_handling():
    """Test ufunc operations with NaN fill values"""
    # NaN fill_value with ufunc
    arr = SparseArray([np.nan, 1.0, np.nan, 2.0], fill_value=np.nan)
    
    # Test unary ufunc
    result = np.exp(arr)
    expected_values = np.exp(np.array([np.nan, 1.0, np.nan, 2.0]))
    expected = SparseArray(expected_values, fill_value=np.exp(np.nan))
    tm.assert_sp_array_equal(result, expected)
    
    # Test binary ufunc with NaN
    arr2 = SparseArray([1.0, np.nan, 2.0, np.nan], fill_value=np.nan)
    result = np.add(arr, arr2)
    expected_values = np.array([np.nan, np.nan, np.nan, np.nan])
    expected = SparseArray(expected_values, fill_value=np.nan)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_inf_handling():
    """Test ufunc operations with infinity values"""
    # Array with inf values
    arr = SparseArray([0.0, np.inf, 0.0, -np.inf, 1.0], fill_value=0.0)
    
    # Test abs with inf
    result = np.abs(arr)
    expected = SparseArray([0.0, np.inf, 0.0, np.inf, 1.0], fill_value=0.0)
    tm.assert_sp_array_equal(result, expected)
    
    # Test add with inf
    result = np.add(arr, 1.0)
    expected = SparseArray([1.0, np.inf, 1.0, -np.inf, 2.0], fill_value=1.0)
    tm.assert_sp_array_equal(result, expected)
    
    # Inf as fill_value
    arr_inf_fill = SparseArray([np.inf, 1.0, np.inf, 2.0], fill_value=np.inf)
    result = np.negative(arr_inf_fill)
    expected = SparseArray([-np.inf, -1.0, -np.inf, -2.0], fill_value=-np.inf)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_extreme_sparsity():
    """Test ufuncs on arrays with extreme sparsity (>99% fill_value)"""
    # Create array with 99.9% sparsity
    data = [0] * 1000 + [1]
    arr = SparseArray(data, fill_value=0)
    assert arr.density < 0.01  # Verify >99% sparse
    
    # Test ufunc preserves sparsity
    result = np.abs(arr)
    expected = SparseArray(data, fill_value=0)
    tm.assert_sp_array_equal(result, expected)
    assert result.density < 0.01
    
    # Test with transformation
    result = np.add(arr, 10)
    expected_data = [10] * 1000 + [11]
    expected = SparseArray(expected_data, fill_value=10)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_almost_dense():
    """Test ufuncs on arrays with low sparsity (<1% fill_value)"""
    # Create array with <1% sparsity (mostly non-fill values)
    data = [1, 2, 3, 4, 5] * 20 + [0]  # 100 non-zero, 1 zero
    arr = SparseArray(data, fill_value=0)
    assert arr.density > 0.99  # Verify <1% sparse
    
    # Test ufunc still works correctly
    result = np.abs(arr)
    expected = SparseArray(data, fill_value=0)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_mixed_fill_values():
    """Test binary ufuncs with arrays having different fill_values"""
    # Arrays with different fill values
    arr1 = SparseArray([0, 1, 0, 2, 0], fill_value=0)
    arr2 = SparseArray([5, 5, 3, 5, 4], fill_value=5)
    
    # Test addition
    result = np.add(arr1, arr2)
    expected_values = [5, 6, 3, 7, 4]
    expected = SparseArray(expected_values, fill_value=5)  # 0 + 5
    tm.assert_sp_array_equal(result, expected)
    
    # Test with np.maximum
    result = np.maximum(arr1, arr2)
    expected_values = [5, 5, 3, 5, 4]
    expected = SparseArray(expected_values, fill_value=5)  # max(0, 5)
    tm.assert_sp_array_equal(result, expected)


def test_ufunc_comparison_edge_cases():
    """Test comparison ufuncs with edge cases"""
    # Empty array comparison
    arr1 = SparseArray([], dtype=np.float64)
    arr2 = SparseArray([], dtype=np.float64)
    result = np.greater(arr1, arr2)
    expected = SparseArray([], dtype=bool)
    tm.assert_sp_array_equal(result, expected)
    
    # All equal elements
    arr1 = SparseArray([1, 1, 1], fill_value=1)
    arr2 = SparseArray([1, 1, 1], fill_value=1)
    result = np.equal(arr1, arr2)
    expected = SparseArray([True, True, True], fill_value=True)
    tm.assert_sp_array_equal(result, expected)
    
    # With NaN comparisons
    arr1 = SparseArray([np.nan, 1.0, 2.0], fill_value=np.nan)
    arr2 = SparseArray([np.nan, 1.0, 3.0], fill_value=np.nan)
    result = np.greater(arr1, arr2)
    expected_values = np.greater(np.array([np.nan, 1.0, 2.0]), np.array([np.nan, 1.0, 3.0]))
    expected = SparseArray(expected_values, fill_value=False)
    tm.assert_sp_array_equal(result, expected)

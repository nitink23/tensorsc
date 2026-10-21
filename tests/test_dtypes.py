import itertools
import math

import pytest

import tensorsc as ts


# ---- dtype objects and inference ----


def test_dtype_objects():
    assert ts.float32.name == "float32"
    assert ts.int64.itemsize == 8
    assert ts.float64.is_floating_point and not ts.int32.is_floating_point
    assert repr(ts.uint8) == "tensorsc.uint8"
    assert str(ts.bool) == "bool"
    assert ts.float32 == "float32" and ts.float32 != ts.float64
    assert len({ts.int32, ts.int32, ts.int64}) == 2


def test_inference_from_python_values():
    assert ts.tensor([True, False]).dtype == ts.bool
    assert ts.tensor([1, 2]).dtype == ts.int64
    assert ts.tensor([1, 2.5]).dtype == ts.float32
    assert ts.tensor([True, 2]).dtype == ts.int64
    assert ts.tensor([[True], [1.5]]).dtype == ts.float32
    assert ts.tensor([]).dtype == ts.float32
    assert ts.tensor(7).dtype == ts.int64
    assert ts.tensor([1, 2]).tolist() == [1, 2]


@pytest.mark.parametrize(
    "spelling, expected",
    [(ts.int32, ts.int32), ("uint8", ts.uint8), (bool, ts.bool), (int, ts.int64), (float, ts.float64)],
)
def test_dtype_spellings(spelling, expected):
    assert ts.tensor([1, 0], dtype=spelling).dtype == expected


def test_unknown_dtype():
    with pytest.raises(TypeError, match="unsupported dtype"):
        ts.zeros(2, dtype="float16")


def test_explicit_dtype_converts_values():
    assert ts.tensor([1.7, -1.7], dtype=ts.int32).tolist() == [1, -1]
    assert ts.tensor([0, 2, 0.5], dtype=ts.bool).tolist() == [False, True, True]
    assert ts.tensor([300], dtype=ts.uint8).tolist() == [44]


def test_large_ints_are_exact():
    big = 2**62 + 1
    t = ts.tensor([big])
    assert t.item() == big and t[0] == big
    assert (t + 1).item() == big + 1
    with pytest.raises(OverflowError):
        ts.tensor([2**64])


def test_python_types_out():
    assert type(ts.tensor([True]).item()) is bool
    assert type(ts.tensor([1])[0]) is int
    assert type(ts.tensor([1.5]).tolist()[0]) is float
    assert type(ts.tensor(2, dtype=ts.uint8).item()) is int


def test_creation_dtypes():
    assert ts.zeros(2).dtype == ts.float32
    assert ts.ones(2, dtype=ts.int32).dtype == ts.int32
    assert ts.full((2,), 7).dtype == ts.int64
    assert ts.full((2,), 7.0).dtype == ts.float32
    assert ts.full(2, True).dtype == ts.bool
    assert ts.full(2, 7, dtype=ts.float64).dtype == ts.float64
    assert ts.arange(5).dtype == ts.int64
    assert ts.arange(5).tolist() == [0, 1, 2, 3, 4]
    assert ts.arange(0, 1, 0.5).dtype == ts.float32
    assert ts.arange(3, dtype=ts.float64).dtype == ts.float64


# ---- promotion ----


def test_tensor_tensor_promotion():
    i32 = ts.tensor([1], dtype=ts.int32)
    i64 = ts.tensor([1])
    f32 = ts.tensor([1.0])
    f64 = ts.tensor([1.0], dtype=ts.float64)
    u8 = ts.tensor([1], dtype=ts.uint8)
    b = ts.tensor([True])
    assert (i32 + i64).dtype == ts.int64
    assert (i64 + f32).dtype == ts.float32
    assert (f32 + f64).dtype == ts.float64
    assert (u8 + i32).dtype == ts.int32
    assert (b + u8).dtype == ts.uint8


def test_python_numbers_adopt_the_tensor_dtype():
    u8 = ts.tensor([250], dtype=ts.uint8)
    assert (u8 + 10).dtype == ts.uint8
    assert (u8 + 10).tolist() == [4], "wraps like NumPy/PyTorch"
    assert (ts.tensor([1], dtype=ts.int32) * 3).dtype == ts.int32
    assert (ts.tensor([1]) * 2.5).dtype == ts.float32
    assert (ts.tensor([True]) + 1).dtype == ts.int64
    f64 = ts.tensor([0.0], dtype=ts.float64) + 0.1
    assert f64.dtype == ts.float64 and f64.item() == 0.1, "no detour through float32"
    assert (ts.tensor([1.0]) + True).dtype == ts.float32


def test_division_and_float_functions_give_floats():
    assert (ts.tensor([1, 2]) / 2).tolist() == [0.5, 1.0]
    assert (ts.tensor([1, 2]) / 2).dtype == ts.float32
    assert ts.tensor([0]).exp().dtype == ts.float32
    assert ts.tensor([0], dtype=ts.float64).exp().dtype == ts.float64


# ---- integer arithmetic ----

VALUES = [-7, -3, -1, 0, 1, 2, 5, 9]


def test_floor_division_and_modulo_match_python_ints():
    for a, b in itertools.product(VALUES, VALUES):
        if b == 0:
            continue
        assert (ts.tensor([a]) // b).item() == a // b, f"{a} // {b}"
        assert (ts.tensor([a]) % b).item() == a % b, f"{a} % {b}"


def test_floor_division_and_modulo_match_python_floats():
    floats = [-7.5, -2.0, -0.5, 0.0, 0.5, 3.25, 8.0]
    for a, b in itertools.product(floats, floats):
        if b == 0:
            continue
        x = ts.tensor([a], dtype=ts.float64)
        assert (x // b).item() == a // b, f"{a} // {b}"
        assert (x % b).item() == a % b, f"{a} % {b}"
        assert math.copysign(1, (x % b).item()) == math.copysign(1, a % b)


def test_integer_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        ts.tensor([1, 2]) // ts.tensor([1, 0])
    with pytest.raises(ZeroDivisionError):
        ts.tensor([1]) % 0
    # Floats follow IEEE instead.
    assert (ts.tensor([1.0]) // 0).item() == math.inf


def test_integer_powers():
    assert (ts.tensor([2, 3]) ** 3).tolist() == [8, 27]
    assert (2 ** ts.tensor([0, 10])).tolist() == [1, 1024]
    with pytest.raises(ValueError, match="negative integer powers"):
        ts.tensor([2]) ** -1
    assert (ts.tensor([2]) ** -1.0).tolist() == [0.5]


def test_bool_arithmetic():
    a, b = ts.tensor([True, False]), ts.tensor([True, True])
    assert (a + b).tolist() == [True, True]
    assert (a * b).tolist() == [True, False]
    with pytest.raises(TypeError, match="subtraction"):
        a - b
    with pytest.raises(TypeError, match="negation"):
        -a


def test_in_place_keeps_dtype():
    t = ts.tensor([1, 2])
    t += 1
    t *= ts.tensor([2], dtype=ts.int32)
    assert t.tolist() == [4, 6] and t.dtype == ts.int64
    with pytest.raises(ValueError, match="can't be cast"):
        t += 0.5
    with pytest.raises(ValueError, match="can't be cast"):
        t /= 2
    f = ts.tensor([1.0])
    f += ts.tensor([1])
    assert f.tolist() == [2.0]


# ---- comparisons, bitwise and logical ----


def test_comparisons_return_bool_tensors():
    t = ts.tensor([1, 2, 3])
    assert (t > 1).dtype == ts.bool
    assert (t > 1).tolist() == [False, True, True]
    assert (t == 2).tolist() == [False, True, False]
    assert (t != ts.tensor([1, 0, 3])).tolist() == [False, True, False]
    assert (2 <= t).tolist() == [False, True, True]
    nan = ts.tensor([math.nan])
    assert (nan == nan).tolist() == [False]


def test_tensors_are_not_hashable():
    with pytest.raises(TypeError):
        hash(ts.tensor([1]))


def test_truthiness():
    assert bool(ts.tensor([3]))
    assert not ts.tensor(0.0)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(ts.tensor([1, 2]))
    with pytest.raises(ValueError, match="empty"):
        bool(ts.tensor([]))
    if ts.tensor([1]) == 1:
        pass


def test_int_and_index_conversion():
    assert int(ts.tensor([2.9])) == 2
    assert ["a", "b", "c"][ts.tensor(2)] == "c"
    with pytest.raises(TypeError):
        ["a"][ts.tensor(0.0)]


def test_bitwise_ops():
    a, b = ts.tensor([0b1100, 5]), ts.tensor([0b1010, 3])
    assert (a & b).tolist() == [0b1000, 1]
    assert (a | b).tolist() == [0b1110, 7]
    assert (a ^ b).tolist() == [0b0110, 6]
    assert (~a).tolist() == [~0b1100, ~5]
    m = ts.tensor([True, False])
    assert (~m).tolist() == [False, True]
    assert (m | True).tolist() == [True, True]
    with pytest.raises(TypeError, match="bitwise"):
        ts.tensor([1.0]) & ts.tensor([1.0])
    m &= ts.tensor([False, False])
    assert m.tolist() == [False, False]


def test_logical_functions():
    x = ts.tensor([0.0, 2.5, -1.0])
    assert ts.logical_not(x).tolist() == [True, False, False]
    assert ts.logical_and(x, [1, 1, 0]).tolist() == [False, True, False]
    assert x.logical_or(ts.tensor([True, False, False])).tolist() == [True, True, True]
    assert ts.logical_xor([True, False], [True, True]).tolist() == [False, True]


def test_where():
    x = ts.tensor([-2, 0, 3])
    assert ts.where(x > 0, x, 0).tolist() == [0, 0, 3]
    assert ts.where(x > 0, x, 0).dtype == ts.int64
    assert ts.where(x > 0, 1.5, x).dtype == ts.float32
    assert ts.where([True, False], 1, 2).tolist() == [1, 2]


def test_clamp_and_float_tests():
    x = ts.tensor([-2, 0, 3])
    assert x.clamp(min=0).tolist() == [0, 0, 3]
    assert x.clamp(min=0).dtype == ts.int64
    assert x.clamp(max=0.5).dtype == ts.float32
    f = ts.tensor([1.0, math.nan, math.inf])
    assert f.isnan().tolist() == [False, True, False]
    assert ts.isinf(f).tolist() == [False, False, True]
    assert ts.isfinite(x).tolist() == [True, True, True]


def test_rounding():
    x = ts.tensor([-1.5, -0.5, 0.5, 1.5, 2.5])
    assert x.round().tolist() == [-2.0, -0.0, 0.0, 2.0, 2.0]
    assert ts.floor(x).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert ts.ceil(x).tolist() == [-1.0, -0.0, 1.0, 2.0, 3.0]
    assert ts.trunc(x).tolist() == [-1.0, -0.0, 0.0, 1.0, 2.0]


# ---- conversion ----


def test_astype_and_to():
    t = ts.tensor([1.5, -2.5])
    assert t.astype(ts.int32).tolist() == [1, -2]
    assert not t.astype("float32").shares_memory(t), "astype always copies"
    assert t.to(ts.float32).shares_memory(t), "to() shares when nothing changes"
    assert t.to("float64").dtype == ts.float64
    assert t.is_floating_point() and not t.astype(int).is_floating_point()


def test_assignment_converts_to_the_tensor_dtype():
    t = ts.zeros(3, dtype=ts.int64)
    t[0] = 2.9
    t[1:] = ts.tensor([1.5, -1.5])
    assert t.tolist() == [2, 1, -1]


# ---- advanced indexing ----


def test_mask_indexing():
    t = ts.arange(6).reshape(2, 3)
    assert t[t > 2].tolist() == [3, 4, 5]
    assert not t[t > 2].shares_memory(t), "mask indexing copies"
    assert t[ts.tensor([True, False])].tolist() == [[0, 1, 2]]
    assert t[[False, True]].tolist() == [[3, 4, 5]]
    assert t[True].shape == (1, 2, 3)
    with pytest.raises(IndexError, match="doesn't match"):
        t[ts.tensor([True, False, True])]


def test_mask_assignment():
    t = ts.tensor([-2.0, 1.0, -3.0, 4.0])
    t[t < 0] = 0
    assert t.tolist() == [0.0, 1.0, 0.0, 4.0]
    t[t > 0] = ts.tensor([10, 20])
    assert t.tolist() == [0.0, 10.0, 0.0, 20.0]
    with pytest.raises(ValueError, match="shape mismatch"):
        t[t > 0] = [1, 2, 3]


def test_position_indexing():
    t = ts.arange(12).reshape(3, 4)
    assert t[[2, 0]].tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]
    assert t[ts.tensor([-1])].tolist() == [[8, 9, 10, 11]]
    assert t[[[0], [1]]].shape == (2, 1, 4)
    assert t.index_select(1, [3, 0]).tolist() == [[3, 0], [7, 4], [11, 8]]
    with pytest.raises(IndexError, match="out of bounds"):
        t[[3]]
    with pytest.raises(IndexError, match="integer and bool"):
        t[ts.tensor([0.0])]
    with pytest.raises(IndexError, match="can't be combined"):
        t[[0, 1], 0]


def test_position_assignment():
    t = ts.zeros(3, 2, dtype=ts.int64)
    t[[0, 2]] = ts.tensor([[1, 2], [3, 4]])
    t[[1]] = 9
    assert t.tolist() == [[1, 2], [9, 9], [3, 4]]

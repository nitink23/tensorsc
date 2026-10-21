import itertools
import math
import random

import pytest

import tensorsc as ts

np = pytest.importorskip("numpy")

SHAPE = (3, 4, 5)
AXES = [None, 0, 1, 2, -1, (0, 2), (1, 2), (0, 1, 2)]


def random_data(shape, seed=0, ints=False):
    rng = random.Random(seed)
    n = math.prod(shape)
    values = [rng.randint(-5, 5) if ints else rng.uniform(-1, 1) for _ in range(n)]
    return ts.tensor(values).reshape(*shape)


def as_numpy(t):
    return np.array(t.tolist(), dtype=t.dtype.name)


def check(got, expected):
    expected = np.asarray(expected)
    assert got.shape == expected.shape
    np.testing.assert_allclose(as_numpy(got), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["sum", "prod", "mean", "max", "min"])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
def test_matches_numpy(name, axis, keepdims):
    t = random_data(SHAPE)
    expected = getattr(np, name)(as_numpy(t).astype(np.float64), axis=axis, keepdims=keepdims)
    check(getattr(t, name)(axis=axis, keepdims=keepdims), expected)
    # PyTorch spelling and the module function give the same thing.
    check(getattr(t, name)(dim=axis, keepdim=keepdims), expected)
    check(getattr(ts, name)(t, axis, keepdims), expected)


@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1])
def test_arg_extremes_match_numpy(axis):
    t = random_data(SHAPE, ints=True)  # ints have ties: check first occurrence
    a = as_numpy(t)
    check(t.argmax(axis), np.argmax(a, axis=axis))
    check(t.argmin(dim=axis), np.argmin(a, axis=axis))


@pytest.mark.parametrize("axis", AXES)
def test_strided_views_match_numpy(axis):
    t = random_data((4, 6, 5)).permute(2, 0, 1)[::-1, 1:, ::2]
    a = as_numpy(t).astype(np.float64)
    check(t.sum(axis), a.sum(axis=axis))
    check(t.max(axis=axis, keepdims=True), a.max(axis=axis, keepdims=True))


def test_all_any():
    t = ts.tensor([[1, 0, 2], [3, 4, 5]])
    assert t.all().item() is False
    assert t.any().item() is True
    assert t.all(dim=1).tolist() == [False, True]
    assert ts.any(t == 0, axis=0).tolist() == [False, True, False]


def test_result_dtypes():
    assert ts.tensor([1, 2]).sum().dtype == ts.int64
    assert ts.tensor([True, True]).sum().item() == 2
    assert ts.tensor([1, 2]).mean().dtype == ts.float32
    assert ts.tensor([1.0], dtype=ts.float64).sum().dtype == ts.float64
    assert ts.tensor([1, 2], dtype=ts.uint8).max().dtype == ts.uint8
    assert ts.tensor([1.0, 2.0]).argmax().dtype == ts.int64
    assert ts.tensor([1.0]).all().dtype == ts.bool


def test_full_reductions_are_0d_tensors():
    s = ts.tensor([1.5, 2.5]).sum()
    assert s.shape == () and s.item() == 4.0 and float(s) == 4.0


def test_float32_sum_is_accurate():
    t = ts.full((1000, 1000), 0.1)
    assert abs(t.sum().item() - 100_000) < 0.01


def test_nan_handling():
    t = ts.tensor([1.0, math.nan, 3.0])
    assert math.isnan(t.max().item())
    assert t.argmax().item() == 1


def test_empty():
    e = ts.zeros(0, 3)
    assert e.sum().item() == 0.0
    assert math.isnan(e.mean().item())
    with pytest.raises(ValueError, match="empty"):
        e.max()


def test_argument_errors():
    t = ts.zeros(2, 3)
    with pytest.raises(TypeError, match="not both"):
        t.sum(dim=0, axis=0)
    with pytest.raises(IndexError):
        t.sum(dim=2)
    with pytest.raises(ValueError, match="more than once"):
        t.sum(dim=(0, -2))
    with pytest.raises(TypeError, match="single axis"):
        t.argmax(dim=(0, 1))


# ---- matmul ----

MATMUL_SHAPES = [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((3,), (3, 4)),
    ((2, 3), (3, 4)),
    ((5, 2, 3), (3, 4)),
    ((2, 1, 2, 3), (4, 3, 2)),
    ((3,), (2, 3, 4)),
]


@pytest.mark.parametrize("a_shape, b_shape", MATMUL_SHAPES)
def test_matmul_matches_numpy(a_shape, b_shape):
    a, b = random_data(a_shape, seed=1), random_data(b_shape, seed=2)
    expected = as_numpy(a) @ as_numpy(b)
    check(a @ b, expected)
    check(ts.matmul(a, b), expected)
    check(a.matmul(b), expected)


def test_matmul_ints_lists_and_views():
    a = ts.tensor([[1, 2], [3, 4]])
    assert (a @ a).tolist() == [[7, 10], [15, 22]]
    assert (a @ a).dtype == ts.int64
    assert (a.T @ [1, 1]).tolist() == [4, 6]
    assert ([1, 0] @ a).tolist() == [1, 2]
    check(a.T @ a.T, as_numpy(a).T @ as_numpy(a).T)


def test_matmul_errors():
    with pytest.raises(ValueError, match="don't line up"):
        ts.zeros(2, 3) @ ts.zeros(2, 3)
    with pytest.raises(ValueError, match="1-D"):
        ts.zeros(2) @ 3
    with pytest.raises(TypeError, match="matmul"):
        ts.tensor([[True]]) @ ts.tensor([[True]])

import math

import pytest

import tensorsc as ts


def approx(t, expected):
    """Compare a 1-D tensor's values to Python numbers, allowing float rounding."""
    assert t.ndim == 1
    assert t.tolist() == pytest.approx(expected)


def test_arithmetic_operators():
    a = ts.tensor([1, 2, 3])
    b = ts.tensor([4, 5, 6])
    assert (a + b).tolist() == [5.0, 7.0, 9.0]
    assert (a - b).tolist() == [-3.0, -3.0, -3.0]
    assert (a * b).tolist() == [4.0, 10.0, 18.0]
    approx(a / b, [0.25, 0.4, 0.5])
    assert (a**2).tolist() == [1.0, 4.0, 9.0]
    assert (-a).tolist() == [-1.0, -2.0, -3.0]
    assert abs(ts.tensor([-1, 2])).tolist() == [1.0, 2.0]


def test_numbers_and_lists_on_either_side():
    a = ts.tensor([1, 2, 4])
    assert (a + 1).tolist() == [2.0, 3.0, 5.0]
    assert (10 - a).tolist() == [9.0, 8.0, 6.0]
    assert (2 * a).tolist() == [2.0, 4.0, 8.0]
    approx(1 / a, [1.0, 0.5, 0.25])
    assert (2**a).tolist() == [2.0, 4.0, 16.0]
    assert (a + [10, 20, 30]).tolist() == [11.0, 22.0, 34.0]
    assert ([1, 1, 1] - a).tolist() == [0.0, -1.0, -3.0]


def test_unsupported_operand_raises_type_error():
    with pytest.raises(TypeError):
        ts.tensor([1]) + "a"
    with pytest.raises(TypeError):
        "a" + ts.tensor([1])


def test_broadcasting():
    col = ts.tensor([[1], [2]])
    row = ts.tensor([10, 20, 30])
    assert (col + row).tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    assert (ts.zeros(2, 1, 3) + ts.ones(4, 1)).shape == (2, 4, 3)
    with pytest.raises(ValueError, match="cannot be broadcast"):
        ts.zeros(2, 3) + ts.zeros(4)
    assert ts.broadcast_shapes((3, 1), (4,), (2, 1, 1)) == (2, 3, 4)


def test_expand_is_a_view():
    x = ts.tensor([[1], [2], [3]])
    e = x.expand(2, -1, 4)
    assert e.shape == (2, 3, 4)
    assert e.strides == (0, 1, 0)
    assert e.shares_memory(x)
    assert e[1, 2, 3] == 3.0
    assert ts.broadcast_to([1, 2], (3, 2)).tolist() == [[1.0, 2.0]] * 3
    with pytest.raises(ValueError, match="cannot expand"):
        x.expand(2, 2)


def test_in_place_ops_write_through_views():
    a = ts.zeros(2, 3)
    view = a[1]
    view += [1, 2, 3]
    a[:, 0] *= 0  # in-place through a temporary view
    a -= 1
    assert a.tolist() == [[-1.0, -1.0, -1.0], [-1.0, 1.0, 2.0]]
    alias = a
    a /= 2
    assert alias is a
    with pytest.raises(ValueError, match="shape mismatch"):
        a[0] += ts.zeros(2, 3)


def test_assignment_broadcasts():
    t = ts.zeros(2, 3)
    t[:] = [1, 2, 3]
    assert t.tolist() == [[1.0, 2.0, 3.0]] * 2
    t[:, 1:] = [[7], [8]]
    assert t.tolist() == [[1.0, 7.0, 7.0], [1.0, 8.0, 8.0]]


def test_unary_functions():
    x = ts.tensor([-1.0, 0.0, 2.0])
    approx(x.exp(), [math.exp(v) for v in (-1, 0, 2)])
    approx(ts.exp(x), [math.exp(v) for v in (-1, 0, 2)])
    approx(ts.sin([0, math.pi / 2]), [0.0, 1.0])
    approx(ts.tanh(x), [math.tanh(v) for v in (-1, 0, 2)])
    approx(x.sigmoid(), [1 / (1 + math.exp(-v)) for v in (-1, 0, 2)])
    assert x.relu().tolist() == [0.0, 0.0, 2.0]
    assert ts.relu([-3, 3]).tolist() == [0.0, 3.0]
    approx(ts.sqrt([4, 9]), [2.0, 3.0])
    approx(ts.log(ts.exp(x)), [-1.0, 0.0, 2.0])
    assert math.isnan(ts.sqrt(-1).item())
    assert x.clamp(min=-0.5).tolist() == [-0.5, 0.0, 2.0]
    assert ts.clamp(x, max=1).tolist() == [-1.0, 0.0, 1.0]


def test_maximum_minimum():
    a = ts.tensor([1, 5, 3])
    assert ts.maximum(a, 2).tolist() == [2.0, 5.0, 3.0]
    assert a.minimum([2, 2, 2]).tolist() == [1.0, 2.0, 2.0]
    assert math.isnan(ts.maximum([float("nan")], 1).item())


def test_allclose():
    a = ts.tensor([1.0, 2.0])
    assert a.allclose([1.0, 2.0000001])
    assert not ts.allclose(a, [1.0, 2.1])
    assert ts.allclose(a, [1.05, 2.0], atol=0.1)
    assert ts.allclose((ts.arange(4) + 0.5) * 2, [1, 3, 5, 7])


def test_division_by_zero_follows_ieee():
    r = ts.tensor([1.0, -1.0, 0.0]) / 0
    assert r[0] == math.inf and r[1] == -math.inf and math.isnan(r[2])


def test_results_are_new_tensors():
    a = ts.tensor([1, 2])
    b = a + 0
    assert not b.shares_memory(a)
    c = +a
    assert not c.shares_memory(a)

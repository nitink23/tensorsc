import pytest

import tensorsc as ts


def test_from_nested_list():
    t = ts.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.numel() == 6
    assert t.strides == (3, 1)
    assert t.dtype == "int64"
    assert t.device == "cpu"
    assert t.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert all(type(x) is int for x in t.tolist()[0])
    assert ts.Tensor((1.5, 2.5)).tolist() == [1.5, 2.5]


def test_scalar_tensor():
    t = ts.tensor(3)
    assert t.shape == ()
    assert t.item() == 3.0
    assert float(t) == 3.0
    assert t.tolist() == 3.0
    with pytest.raises(TypeError):
        len(t)


@pytest.mark.parametrize("bad", [[[1, 2], [3]], [1, [2]], [[1], 2]])
def test_ragged_lists_rejected(bad):
    with pytest.raises(ValueError, match="rectangular"):
        ts.tensor(bad)


def test_non_numbers_rejected():
    with pytest.raises(TypeError, match="must be numbers, got str"):
        ts.tensor([1, "a"])


def test_creation_functions():
    assert ts.zeros(2, 3).tolist() == [[0.0] * 3] * 2
    assert ts.zeros((2, 3)).shape == (2, 3)
    assert ts.ones(4).tolist() == [1.0] * 4
    assert ts.full((2, 2), 7).tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert ts.arange(4).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert ts.arange(1, 2, 0.5).tolist() == [1.0, 1.5]
    with pytest.raises(ValueError):
        ts.arange(0, 1, 0)


def test_indexing():
    t = ts.tensor([[1, 2, 3], [4, 5, 6]])
    assert t[1, 2] == 6.0
    assert t[-1, -1] == 6.0
    t[0, 0] = 10
    assert t[0, 0] == 10.0
    with pytest.raises(IndexError):
        t[2, 0]
    with pytest.raises(IndexError, match="too many indices"):
        t[0, 0, 0]
    with pytest.raises(IndexError):
        t["a"]
    assert ts.tensor([5, 6])[1] == 6.0


def test_copies_are_independent():
    a = ts.tensor([1, 2])
    for b in (a.clone(), a.copy(), ts.tensor(a)):
        assert not b.shares_memory(a)
        b[0] = 99
        assert a[0] == 1.0
    assert a.shares_memory(a)


def test_repr():
    assert repr(ts.tensor([[1, 2], [3, 4]])) == "tensor([[1, 2], [3, 4]])"
    assert repr(ts.tensor([[1.0, 2], [3, 4]])) == "tensor([[1.0, 2.0], [3.0, 4.0]])"

import itertools

import pytest

import tensorsc as ts


def grid(rows=3, cols=4):
    """A rows x cols tensor holding 0, 1, 2, ... plus the same data as lists."""
    data = [[float(r * cols + c) for c in range(cols)] for r in range(rows)]
    return ts.tensor(data), data


BOUNDS = [None, -5, -1, 0, 1, 3, 10]
STEPS = [None, 1, 2, -1, -2]


@pytest.mark.parametrize("start,stop,step", list(itertools.product(BOUNDS, BOUNDS, STEPS)))
def test_slicing_matches_python_lists(start, stop, step):
    t, data = grid()
    sl = slice(start, stop, step)
    assert t[sl].tolist() == data[sl]
    assert t[:, sl].tolist() == [row[sl] for row in data]


def test_huge_slice_bounds_behave_like_lists():
    t, data = grid()
    for sl in (slice(None, 10**30), slice(-(10**30), None), slice(None, None, 10**30),
               slice(None, None, -(10**30)), slice(10**30, None, -1)):
        assert t[sl].tolist() == data[sl], sl


def test_indexing_returns_views():
    t, data = grid()
    assert t[1].tolist() == data[1]
    assert t[-1].shares_memory(t)
    assert t[:, 2].tolist() == [row[2] for row in data]
    assert t[1:, ::2].tolist() == [row[::2] for row in data[1:]]
    assert t[..., 0].tolist() == [row[0] for row in data]
    assert t[None].shape == (1, 3, 4)
    assert t[:, None, :].shape == (3, 1, 4)
    assert t[0][1] == t[0, 1]


def test_assignment_writes_through_views():
    t, _ = grid()
    t[0] = -1
    t[:, -1] = [7, 8, 9]
    t[1:, 0] = ts.tensor([5, 6])
    assert t.tolist() == [
        [-1.0, -1.0, -1.0, 7.0],
        [5.0, 5.0, 6.0, 8.0],
        [6.0, 9.0, 10.0, 9.0],
    ]
    with pytest.raises(ValueError, match="shape mismatch"):
        t[0] = [1, 2]


def test_overlapping_assignment():
    t = ts.arange(5)
    t[:] = t[::-1]
    assert t.tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]


def test_view_sharing_is_visible_both_ways():
    a = ts.zeros(2, 3)
    b = a.reshape(6)
    b[4] = 1
    a[0, 0] = 2
    assert a.tolist() == [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert b.tolist() == [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_reshape_and_view():
    t = ts.arange(6)
    assert t.reshape(2, 3).shape == (2, 3)
    assert t.reshape((3, -1)).shape == (3, 2)
    assert t.view(-1, 1).shape == (6, 1)
    with pytest.raises(ValueError):
        t.reshape(4, -1)

    tt = t.reshape(2, 3).T
    assert not tt.is_contiguous()
    with pytest.raises(ValueError, match="use reshape"):
        tt.view(6)
    flat = tt.reshape(6)
    assert flat.tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    assert not flat.shares_memory(t)


def test_transpose_permute():
    t = ts.arange(24).reshape(2, 3, 4)
    assert t.transpose(0, 2).shape == (4, 3, 2)
    assert t.transpose().shape == (4, 3, 2)
    assert t.T.shape == (4, 3, 2)
    assert t.permute(1, 2, 0).shape == (3, 4, 2)
    assert t.permute(1, 2, 0)[2, 3, 1] == t[1, 2, 3]
    assert t.strides == (12, 4, 1)
    assert t.T.strides == (1, 4, 12)
    with pytest.raises(IndexError):
        t.transpose(0, 3)
    with pytest.raises(TypeError):
        t.transpose(0, 1, 2)


def test_squeeze_unsqueeze_flatten():
    t = ts.zeros(1, 3, 1)
    assert t.squeeze().shape == (3,)
    assert t.squeeze(0).shape == (3, 1)
    assert t.unsqueeze(-1).shape == (1, 3, 1, 1)
    assert ts.zeros(2, 3, 4).flatten().shape == (24,)
    assert ts.zeros(2, 3, 4).flatten(1).shape == (2, 12)


def test_contiguous():
    t = ts.arange(6).reshape(2, 3)
    assert t.contiguous().shares_memory(t)
    c = t.T.contiguous()
    assert c.is_contiguous() and not c.shares_memory(t)
    assert c.tolist() == t.T.tolist()
    assert t[1].storage_offset() == 3


def test_iteration():
    t, data = grid()
    assert [row.tolist() for row in t] == data
    assert list(ts.tensor([1, 2])) == [1.0, 2.0]

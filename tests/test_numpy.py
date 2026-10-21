import gc

import pytest

import tensorsc as ts

np = pytest.importorskip("numpy")

DTYPES = ["bool", "uint8", "int32", "int64", "float32", "float64"]


def sample(dtype, shape=(2, 3)):
    values = np.arange(np.prod(shape)).reshape(shape) % 5
    return values.astype(dtype)


# ---- tensor -> NumPy ----


@pytest.mark.parametrize("dtype", DTYPES)
def test_asarray_keeps_dtype_and_values(dtype):
    t = ts.tensor(sample(dtype).tolist(), dtype=dtype)
    a = np.asarray(t)
    assert a.dtype == np.dtype(dtype)
    assert a.tolist() == t.tolist()


def test_asarray_shares_memory_both_ways():
    t = ts.zeros(2, 3)
    a = np.asarray(t)
    a[0, 1] = 5
    t[1, 2] = 7
    assert t.tolist() == [[0, 5, 0], [0, 0, 7]]
    assert a.tolist() == t.tolist()
    assert t.numpy().base is not None, "numpy() is a view, not a copy"


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t.T,
        lambda t: t[::-1],
        lambda t: t[:, ::2],
        lambda t: t[1],
        lambda t: t.permute(1, 0)[::-1, 1:],
    ],
    ids=["transposed", "reversed", "strided", "row", "mixed"],
)
def test_views_export_without_copying(make):
    t = ts.arange(12.0).reshape(3, 4)
    view = make(t)
    a = np.asarray(view)
    assert a.tolist() == view.tolist()
    a.flat[0] = -1  # write through NumPy...
    assert view.tolist() == a.tolist()  # ...and the tensor sees it


def test_read_only_exports():
    for t in (ts.tensor([True, False]), ts.tensor([1.0]).expand(3)):
        a = np.asarray(t)
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a[0] = 0


def test_memoryview_and_lifetime():
    m = memoryview(ts.tensor([1, 2, 3], dtype=ts.int32))
    assert m.format == "i" and m.tolist() == [1, 2, 3]
    a = np.asarray(ts.arange(5.0))  # the tensor is only reachable via `a`
    gc.collect()
    assert a.tolist() == [0, 1, 2, 3, 4]


def test_ufuncs_accept_tensors():
    t = ts.tensor([0.0, 1.0])
    assert np.allclose(np.exp(t), [1.0, np.e])


# ---- NumPy -> tensor ----


@pytest.mark.parametrize("dtype", DTYPES)
def test_from_numpy_dtypes(dtype):
    a = sample(dtype)
    t = ts.from_numpy(a)
    assert t.dtype == dtype
    assert t.tolist() == a.tolist()


def test_from_numpy_shares_memory_both_ways():
    a = np.zeros((2, 3), dtype=np.float32)
    t = ts.from_numpy(a)
    t[0] = 1
    a[1, 2] = 9
    assert a.tolist() == [[1, 1, 1], [0, 0, 9]]
    assert t.tolist() == a.tolist()


@pytest.mark.parametrize(
    "make",
    [lambda a: a.T, lambda a: a[::-1, ::-2], lambda a: a[:, 1:3], lambda a: a[1]],
    ids=["transposed", "reversed", "sliced", "row"],
)
def test_from_numpy_views(make):
    a = make(np.arange(20, dtype=np.int64).reshape(4, 5))
    t = ts.from_numpy(a)
    assert t.shape == a.shape
    assert t.tolist() == a.tolist()
    t += 100
    assert t.tolist() == a.tolist()


def test_from_numpy_bool_is_a_copy():
    a = np.array([True, False])
    t = ts.from_numpy(a)
    assert t.tolist() == [True, False]
    a[0] = False
    assert t[0] is True


def test_read_only_arrays_give_read_only_tensors():
    a = np.arange(3.0)
    a.flags.writeable = False
    t = ts.from_numpy(a)
    assert not t.is_writable
    assert (t * 2).tolist() == [0, 2, 4], "reading is fine"
    with pytest.raises(ValueError, match="read-only"):
        t += 1
    with pytest.raises(ValueError, match="read-only"):
        t[0] = 5


def test_lifetime_after_the_array_is_gone():
    t = ts.from_numpy(np.arange(4.0))
    gc.collect()
    assert t.tolist() == [0, 1, 2, 3]


def test_unsupported_arrays():
    with pytest.raises(TypeError, match="unsupported element type"):
        ts.from_numpy(np.zeros(2, dtype=np.float16))
    with pytest.raises(TypeError, match="byte order"):
        ts.from_numpy(np.zeros(2, dtype=">f4"))
    with pytest.raises(TypeError, match="NumPy array"):
        ts.from_numpy([1, 2])


def test_empty_and_0d():
    assert ts.from_numpy(np.zeros((0, 3))).shape == (0, 3)
    t = ts.from_numpy(np.array(2.5))
    assert t.shape == () and t.item() == 2.5


def test_round_trip_aliasing_is_detected():
    # Two tensor objects over the same memory, through NumPy.
    t = ts.arange(4.0)
    u = ts.from_numpy(np.asarray(t))
    assert u.shares_memory(t) and t.shares_memory(u)
    t += u
    assert t.tolist() == [0, 2, 4, 6] and u.tolist() == t.tolist()
    t[1:] += u[:-1]  # reads the original values, as in NumPy
    assert t.tolist() == [0, 2, 6, 10]


# ---- copying and mixing ----


def test_tensor_copies_arrays():
    a = np.arange(3, dtype=np.int32)
    t = ts.tensor(a)
    assert t.dtype == ts.int32 and not t.shares_memory(ts.from_numpy(a))
    a[0] = 9
    assert t.tolist() == [0, 1, 2]
    assert ts.tensor(a, dtype=ts.float64).dtype == ts.float64
    assert ts.tensor(np.float32(1.5)).dtype == ts.float32


def test_mixed_operators_give_tensors():
    t = ts.tensor([1.0, 2.0])
    a = np.array([10.0, 20.0], dtype=np.float32)
    for result in (t + a, a + t, a * t, t @ a):
        assert isinstance(result, ts.Tensor)
    assert (a - t).tolist() == [9.0, 18.0]
    assert (a > t).tolist() == [True, True]


def test_numpy_scalars_keep_their_dtype():
    # Unlike plain Python numbers, NumPy scalars are typed (as in NumPy 2).
    assert (ts.tensor([1]) + np.float64(0.5)).dtype == ts.float64
    assert (ts.tensor([1], dtype=ts.int32) + np.int64(1)).dtype == ts.int64
    assert (ts.tensor([1], dtype=ts.int32) + 1).dtype == ts.int32
    assert ts.full(2, np.float32(1.5)).tolist() == [1.5, 1.5]
    assert ts.arange(np.int64(3)).tolist() == [0, 1, 2]


def test_numpy_dtypes_are_accepted():
    assert ts.zeros(2, dtype=np.float64).dtype == ts.float64
    assert ts.tensor([1, 2]).astype(np.int32).dtype == ts.int32
    assert ts.tensor([1.5]).to(np.dtype("int64")).dtype == ts.int64
    assert ts.float32 == np.float32 and ts.float32 == np.dtype("float32")
    assert ts.bool == np.bool_


def test_numpy_keys():
    t = ts.arange(6).reshape(2, 3)
    assert t[np.array([True, False])].tolist() == [[0, 1, 2]]
    assert t[np.array([1, 0])].tolist() == [[3, 4, 5], [0, 1, 2]]
    row = t[np.int64(1)]
    assert row.shares_memory(t), "a NumPy int scalar is basic indexing"
    t[np.array([[True, False, True], [False, False, False]])] = 0
    assert t.tolist() == [[0, 1, 0], [3, 4, 5]]

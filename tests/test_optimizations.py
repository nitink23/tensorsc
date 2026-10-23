"""Regression coverage for allocation-free comparisons and view layouts."""
import numpy as np
import pytest
import tensorsc as ts

@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int32', 'int64', 'float32', 'float64'])
def test_allclose_layouts(dtype):
    a = np.arange(24).reshape(4, 6).astype(dtype)
    x = ts.tensor(a)
    for actual, expected in [(x, a), (x.T, a.T), (x[::-1, ::2], a[::-1, ::2]), (x[1:], a[1:])]:
        assert actual.allclose(ts.tensor(expected))
    assert ts.zeros(0, 3, dtype=dtype).allclose(ts.ones(1, 3, dtype=dtype))
    assert x.allclose(x[0]) == bool(np.allclose(a, a[0]))

@pytest.mark.parametrize('a,b', [(np.inf,np.inf),(-np.inf,-np.inf),(1.,np.inf),(np.inf,-np.inf),(np.nan,np.nan),(1.,1.000001)])
def test_allclose_nonfinite(a,b):
    assert ts.allclose(a,b) == bool(np.allclose(np.float32(a),np.float32(b)))

@pytest.mark.parametrize('left,right', [('float32','float64'),('int64','float32'),('int32','int64')])
def test_allclose_mixed(left,right):
    a = np.array([[1],[2]],dtype=left)
    b = np.array([1,2,3],dtype=right)
    assert ts.tensor(a).allclose(ts.tensor(b)) == bool(np.allclose(a,b))
    assert ts.tensor(a).allclose(ts.tensor(a.astype(right)))

def test_allclose_mismatch_and_late_difference():
    with pytest.raises(ValueError): ts.zeros(2,3).allclose(ts.ones(4))
    a = np.ones(200, dtype=np.float32)
    b = a.copy(); b[-1]=2
    assert not ts.tensor(a).allclose(ts.tensor(b))
    b = a.copy(); b[0]=2
    assert not ts.tensor(a).allclose(ts.tensor(b))

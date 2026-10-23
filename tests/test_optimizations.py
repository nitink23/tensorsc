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
@pytest.mark.parametrize('dtype', ['float32','float64','int32','int64'])
@pytest.mark.parametrize('shape', [(4,16,32),(5,17,33),(2,3,16,32)])
def test_matmul_grouped_rows(dtype,shape):
    m,k,n=shape[-3:]
    rng=np.random.default_rng(7)
    a=rng.integers(-3,4,size=(m,k)).astype(dtype)
    b=rng.integers(-3,4,size=(k,n)).astype(dtype)
    if len(shape)==4:
        a=np.stack([a,a+1]); b=np.stack([b,b-1])
    np.testing.assert_allclose((ts.tensor(a)@ts.tensor(b)).numpy(),a@b,rtol=1e-5,atol=1e-5)

def test_matmul_grouped_gradients():
    rng=np.random.default_rng(3)
    a=rng.normal(size=(5,17)).astype('float64')
    b=rng.normal(size=(17,33)).astype('float64')
    x=ts.tensor(a,requires_grad=True); y=ts.tensor(b,requires_grad=True)
    (x@y).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(),np.ones((5,33))@b.T,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(y.grad.numpy(),a.T@np.ones((5,33)),rtol=1e-10,atol=1e-10)

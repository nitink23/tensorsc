"""DLPack ownership, layout, protocol and gradient regression tests."""
import gc
import numpy as np
import pytest
import tensorsc as ts

DTYPES=['uint8','int32','int64','float32','float64','bool']

@pytest.mark.parametrize('dtype',DTYPES)
def test_numpy_round_trip(dtype):
    a=np.arange(12).reshape(3,4).astype(dtype)
    x=ts.tensor(a)
    b=np.from_dlpack(x)
    assert b.dtype == a.dtype
    np.testing.assert_array_equal(b,a)
    assert np.shares_memory(b,np.asarray(x))
    y=ts.from_dlpack(a)
    np.testing.assert_array_equal(y.numpy(),a)
    if dtype=='bool':
        assert not np.shares_memory(y.numpy(),a)
        with pytest.raises(BufferError): ts.from_dlpack(a,copy=False)
    else:
        assert np.shares_memory(y.numpy(),a)
        y[0,0]=7
        assert a[0,0]==7
        a[0,1]=9
        assert y[0,1]==9

@pytest.mark.parametrize('make',[lambda x:x,lambda x:x.T,lambda x:x[::-1,::-1],lambda x:x[:,::2],lambda x:x[1:3],lambda x:x.reshape(2,2,3)])
def test_layouts(make):
    a=make(np.arange(12,dtype='float32').reshape(3,4))
    x=ts.from_dlpack(a)
    np.testing.assert_array_equal(x.numpy(),a)
    np.testing.assert_array_equal(np.from_dlpack(ts.tensor(a)),a)
    # A non-dense stride-2 source requires a copy with our storage model.
    if a.strides[-1]==8:
        with pytest.raises(BufferError): ts.from_dlpack(a,copy=False)
    else:
        assert np.shares_memory(x.numpy(),a)

@pytest.mark.parametrize('shape',[(),(0,),(2,0,3)])
def test_empty_and_scalar(shape):
    a=np.zeros(shape,dtype='float64')
    x=ts.from_dlpack(a)
    assert x.shape==shape
    np.testing.assert_array_equal(np.from_dlpack(x),a)

@pytest.mark.parametrize('version',[None,(1,0),(1,3)])
def test_capsules_single_use_and_independent_exports(version):
    x=ts.arange(6,dtype='float32').reshape(2,3)
    kw={} if version is None else {'max_version':version}
    capsule=x.__dlpack__(**kw)
    y=ts.from_dlpack(capsule)
    assert y.shares_memory(x)
    with pytest.raises((ValueError,TypeError,BufferError)): ts.from_dlpack(capsule)
    z=ts.from_dlpack(x.__dlpack__(**kw))
    del x,capsule
    gc.collect()
    y[0,0]=11
    assert z[0,0]==11


def test_copy_and_owner_lifetimes():
    a=np.arange(10,dtype='int64')
    x=ts.from_dlpack(a,copy=False)
    copied=ts.from_dlpack(a,copy=True)
    assert not np.shares_memory(copied.numpy(),a)
    del a
    gc.collect()
    assert x.tolist()==list(range(10))
    b=np.from_dlpack(x)
    del x
    gc.collect()
    assert b.tolist()==list(range(10))


def test_readonly_and_expanded():
    a=np.arange(4,dtype='float32'); a.flags.writeable=False
    x=ts.from_dlpack(a)
    assert not x.is_writable
    with pytest.raises((ValueError,BufferError)): x[0]=1
    # Legacy cannot communicate read-only flags.
    with pytest.raises(BufferError): x.__dlpack__()
    e=ts.ones(1,3).expand(4,3)
    b=np.from_dlpack(e)
    assert not b.flags.writeable
    np.testing.assert_array_equal(b,np.ones((4,3)))
    y=ts.from_dlpack(e)
    assert y.strides==(0,1)
    assert not y.is_writable


def test_export_options_and_errors():
    x=ts.ones(2)
    assert x.__dlpack_device__()==(1,0)
    with pytest.raises(BufferError): x.__dlpack__(copy=True)
    with pytest.raises(BufferError): x.__dlpack__(stream=1)
    with pytest.raises(BufferError): x.__dlpack__(dl_device=(2,0))
    with pytest.raises(BufferError): ts.from_dlpack(np.ones(2,dtype='complex64'))
    with pytest.raises(BufferError): ts.from_dlpack(np.ones(2,dtype='float16'))


def test_foreign_device_rejected_before_export():
    class GPU:
        def __dlpack_device__(self): return (2,0)
        def __dlpack__(self,**kwargs): raise AssertionError('must not export GPU data')
    with pytest.raises(BufferError): ts.from_dlpack(GPU())


def test_autograd_detach_and_foreign_saved_values():
    x=ts.tensor([2.0],requires_grad=True)
    with pytest.raises(BufferError): x.__dlpack__()
    ts.from_dlpack(x.detach())[0]=3
    assert x.item()==3
    # Foreign shared values are snapshotted at forward time.
    a=np.array([2.0],dtype='float32')
    x=ts.from_dlpack(a).requires_grad_()
    y=(x*x).sum()
    a[0]=3
    y.backward()
    assert x.grad.tolist()==[4.0]


def test_numpy_export_before_and_after_forward():
    x=ts.tensor([2.0],requires_grad=True)
    a=x.numpy()
    y=(x*x).sum()
    a[0]=3
    y.backward()
    assert x.grad.tolist()==[4.0]
    x=ts.tensor([2.0],requires_grad=True)
    y=(x*x).sum()
    a=x.detach().numpy()
    a[0]=3
    with pytest.raises(ValueError,match='modified in place'): y.backward()


def test_torch_interop():
    torch=pytest.importorskip('torch')
    for dtype in ['float32','float64','int32','int64','uint8']:
        x=ts.tensor([1,2,3],dtype=dtype)
        y=torch.from_dlpack(x)
        y[0]=7
        assert x[0]==7
        z=ts.from_dlpack(y)
        z[1]=8
        assert y[1].item()==8

@pytest.mark.parametrize('dtype',DTYPES)
def test_offset_and_reverse_exports_share(dtype):
    a=np.arange(20).astype(dtype)
    x=ts.tensor(a)[2:18].reshape(4,4).T[::-1]
    result=np.from_dlpack(x)
    np.testing.assert_array_equal(result,a[2:18].reshape(4,4).T[::-1])
    assert np.shares_memory(result,np.asarray(x))


def test_empty_readonly_and_copy_policy():
    a=np.zeros((2,0,3),dtype='float32'); a.flags.writeable=False
    x=ts.from_dlpack(a,copy=False)
    assert not x.is_writable
    assert ts.from_dlpack(a,copy=True).is_writable


def test_dlp_export_after_forward_is_detected():
    x=ts.tensor([2.0],requires_grad=True)
    y=(x*x).sum()
    alias=ts.from_dlpack(x.detach())
    alias[0]=3
    with pytest.raises(ValueError,match='modified in place'): y.backward()


def test_torch_gradients_use_foreign_value_snapshots():
    torch=pytest.importorskip('torch')
    source=torch.tensor([2.0,3.0])
    x=ts.from_dlpack(source).requires_grad_()
    loss=(x*x).sum()
    source.add_(1)
    loss.backward()
    assert x.grad.tolist()==[4.0,6.0]

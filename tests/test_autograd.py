import math
import random

import pytest

import tensorsc as ts


def close(a, b, tol=1e-5):
    a, b = (x.tolist() if isinstance(x, ts.Tensor) else x for x in (a, b))
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def numeric_grad(f, x, eps=1e-6):
    """Central differences of scalar-valued f at float64 tensor x."""
    flat = x.detach().reshape(-1).tolist()
    grads = []
    for i in range(len(flat)):
        hi, lo = list(flat), list(flat)
        hi[i] += eps
        lo[i] -= eps
        with ts.no_grad():
            up = f(ts.tensor(hi, dtype=ts.float64).reshape(*x.shape)).item()
            down = f(ts.tensor(lo, dtype=ts.float64).reshape(*x.shape)).item()
        grads.append((up - down) / (2 * eps))
    return ts.tensor(grads, dtype=ts.float64).reshape(*x.shape)


def test_basic_backward():
    x = ts.tensor([2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    assert y.requires_grad and not y.is_leaf and x.is_leaf
    assert y.grad_fn == "SumBackward"
    y.backward()
    assert x.grad.tolist() == [4.0, 6.0]
    assert repr(x) == "tensor([2.0, 3.0], requires_grad=True)"
    assert "grad_fn=<SumBackward>" in repr(y)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x * x.sin() + x.exp() / 3).sum(),
        lambda x: (x @ x.T).tanh().mean(),
        lambda x: x[1:, ::-1].sigmoid().sum() + x.max(),
        lambda x: ts.where(x > 0, x**2, -x).sum(),
        lambda x: x.abs().clamp(min=0.2, max=0.8).sum(),
        lambda x: x.reshape(-1)[[0, 0, 3]].sum(),
        lambda x: (x - x.mean(dim=1, keepdim=True)).pow(2).sum(),
    ],
    ids=["elementwise", "matmul", "views", "where", "clamp", "gather", "centered"],
)
def test_gradients_match_finite_differences(f):
    rng = random.Random(0)
    x = ts.tensor([[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)], dtype=ts.float64, requires_grad=True)
    f(x).backward()
    assert close(x.grad, numeric_grad(f, x))


def test_gradients_accumulate_and_reset():
    w = ts.tensor([1.0], requires_grad=True)
    (w * 3).sum().backward()
    (w * 3).sum().backward()
    assert w.grad.tolist() == [6.0]
    w.grad = None
    assert w.grad is None


def test_explicit_gradient_for_non_scalars():
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    y = x * 10
    with pytest.raises(ValueError, match="gradient argument"):
        y.backward()
    y.backward(gradient=[1.0, 0.5])
    assert x.grad.tolist() == [10.0, 5.0]


def test_training_a_linear_model():
    rng = random.Random(1)
    xs = ts.tensor([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(64)])
    true_w, true_b = ts.tensor([[2.0], [-3.0]]), 0.5
    ys = xs @ true_w + true_b

    w = ts.zeros(2, 1, requires_grad=True)
    b = ts.zeros(1, requires_grad=True)
    for _ in range(300):
        loss = ((xs @ w + b - ys) ** 2).mean()
        loss.backward()
        with ts.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
        w.grad = None
        b.grad = None
    assert loss.item() < 1e-6
    assert close(w, [[2.0], [-3.0]], 1e-3) and close(b, [0.5], 1e-3)


def test_no_grad_context_and_decorator():
    x = ts.tensor([1.0], requires_grad=True)
    with ts.no_grad():
        assert not ts.is_grad_enabled()
        assert not (x * 2).requires_grad
        with ts.enable_grad():
            assert (x * 2).requires_grad
        assert not ts.is_grad_enabled()
    assert ts.is_grad_enabled()

    @ts.no_grad()
    def predict(t):
        return t * 2

    assert not predict(x).requires_grad
    assert ts.is_grad_enabled()


def test_detach_and_requires_grad_():
    x = ts.tensor([1.0, 2.0])
    assert x.requires_grad_() is x and x.requires_grad
    d = x.detach()
    assert not d.requires_grad and d.shares_memory(x), "same data, outside the graph"
    assert not (x * 2).detach().requires_grad
    x.requires_grad = False
    assert not (x * 2).requires_grad


def test_errors():
    with pytest.raises(ValueError, match="floating point"):
        ts.tensor([1, 2], requires_grad=True)
    x = ts.tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="in-place"):
        x += 1
    with pytest.raises(ValueError, match="in-place"):
        x[0] = 5
    with pytest.raises(ValueError, match="leaf"):
        (x * 2).requires_grad = False
    with pytest.raises(ValueError, match="doesn't require grad"):
        ts.tensor([1.0]).sum().backward()


def test_modified_saved_tensors_are_caught():
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    y = x.exp()
    z = (y * y).sum()
    with ts.no_grad():
        y += 1
    with pytest.raises(ValueError, match="modified in place"):
        z.backward()


def test_non_differentiable_ops_cut_the_graph():
    x = ts.tensor([1.0, -2.0], requires_grad=True)
    assert not (x > 0).requires_grad
    assert not x.argmax().requires_grad
    assert not x.floor().requires_grad
    with pytest.raises(ValueError, match="prod"):
        x.prod()


def test_numpy_operands_are_constants():
    np = pytest.importorskip("numpy")
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    (x * np.array([3.0, 4.0], dtype=np.float32)).sum().backward()
    assert x.grad.tolist() == [3.0, 4.0]

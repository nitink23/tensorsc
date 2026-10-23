"""Focused optimization measurements; build with maturin develop --release first.
Inputs and validation are outside timing; reports median of five batches.
"""
import argparse
import json
import platform
import statistics
import timeit
import numpy as np
import tensorsc as ts


def measure(fn):
    timer = timeit.Timer(fn)
    count = 1
    while timer.timeit(count) < 0.02 and count < 32768:
        count *= 2
    return statistics.median(timer.repeat(repeat=5, number=count)) / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', required=True)
    args = parser.parse_args()
    if ts._debug_build:
        raise SystemExit('Build with maturin develop --release first')
    rows = []
    for side in (4, 128, 512):
        a = np.linspace(.1, 1, side*side, dtype=np.float32).reshape(side, side)
        x, y = ts.tensor(a), ts.tensor(a)
        b = a.copy()
        cases = {
            'add': (lambda: x+y, lambda: a+b),
            'mul_scalar': (lambda: x*2., lambda: a*2.),
            'exp': (lambda: x.exp(), lambda: np.exp(a)),
            'sum': (lambda: x.sum(), lambda: a.sum()),
            'sum_axis0': (lambda: x.sum(axis=0), lambda: a.sum(axis=0)),
            'sum_axis1': (lambda: x.sum(axis=1), lambda: a.sum(axis=1)),
            'allclose_equal': (lambda: x.allclose(y), lambda: np.allclose(a,b)),
            'reshape_view': (lambda: x.reshape(-1), lambda: a.reshape(-1)),
            'tolist': (lambda: x.tolist(), lambda: a.tolist()),
        }
        if side <= 128:
            cases['matmul'] = (lambda: x@y, lambda: a@b)
        for name, (ours, numpy) in cases.items():
            actual = ours()
            if isinstance(actual, ts.Tensor): actual = actual.numpy()
            np.testing.assert_allclose(actual, numpy(), rtol=2e-4, atol=2e-3)
            tt, nt = measure(ours), measure(numpy)
            rows.append(dict(operation=name, side=side, tensorsc_s=tt, numpy_s=nt))
            print(f'{name:18} {side:4}: {tt*1e6:10.2f} us (NumPy {nt*1e6:.2f} us)', flush=True)
    with open(args.save,'w') as f:
        json.dump(dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), build='release', statistic='median of five batches; adaptive batch length >=20ms unless capped', results=rows),f,indent=2)

if __name__ == '__main__': main()

"""Benchmarks tensorsc against NumPy.

Build tensorsc optimized first, or the numbers are meaningless:

    pip install -r benchmarks/requirements.txt
    maturin develop --release
    python benchmarks/bench.py                          # everything
    python benchmarks/bench.py -k add --size large      # a subset
    python benchmarks/bench.py --save benchmarks/results/phase3.json
    python benchmarks/bench.py --compare benchmarks/results/phase3.json

Before timing, each operation runs once in both libraries and the results are
checked against each other, so neither side can win by doing less work.
"""

import argparse
import datetime
import json
import math
import operator
import platform
import random
import sys
import timeit
from types import SimpleNamespace

import numpy as np

import tensorsc as ts

SIZES = {
    "small": (10, 10),
    "medium": (100, 100),
    "large": (1000, 1000),
}


def both(fn):
    """The same expression works for both libraries."""
    return (fn, fn)


# name -> (tensorsc version, numpy version). Each takes the prepared inputs
# (see `prepare`) and returns the result, which is what gets timed.
BENCHMARKS = {
    # creation
    "from_list": (
        lambda x: ts.tensor(x.a_list),
        lambda x: np.array(x.a_list, dtype=np.float32),
    ),
    "zeros": (
        lambda x: ts.zeros(x.shape),
        lambda x: np.zeros(x.shape, dtype=np.float32),
    ),
    # elementwise math
    "add": both(lambda x: x.a + x.b),
    "add_broadcast": both(lambda x: x.a + x.row),
    "mul_scalar": both(lambda x: x.a * 2.0),
    "iadd": both(lambda x: operator.iadd(x.scratch, x.b)),
    "exp": (lambda x: x.a.exp(), lambda x: np.exp(x.a)),
    # NumPy has no sigmoid/relu; these are the usual NumPy spellings.
    "sigmoid": (lambda x: x.a.sigmoid(), lambda x: 1 / (1 + np.exp(-x.a))),
    "relu": (lambda x: x.a.relu(), lambda x: np.maximum(x.a, 0)),
    # views and layout
    "slice_view": both(lambda x: x.a[::2, 1:]),
    "reshape_view": both(lambda x: x.a.reshape(-1)),
    "transpose_copy": (
        lambda x: x.a.T.contiguous(),
        lambda x: np.ascontiguousarray(x.a.T),
    ),
    # element access and conversion
    "getitem": both(lambda x: x.a[x.i, x.j]),
    "tolist": both(lambda x: x.a.tolist()),
}

LIBRARIES = ("tensorsc", "numpy")


def make_lists(shape, seed=0):
    rng = random.Random(seed)
    rows, cols = shape

    def matrix():
        return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

    return {"a": matrix(), "b": matrix(), "row": [rng.uniform(-1, 1) for _ in range(cols)]}


def prepare(library, lists, shape):
    """Fresh inputs for one library, built from the same Python data."""
    if library == "tensorsc":
        convert = ts.tensor
    else:
        convert = lambda data: np.array(data, dtype=np.float32)  # noqa: E731
    return SimpleNamespace(
        a=convert(lists["a"]),
        b=convert(lists["b"]),
        row=convert(lists["row"]),
        scratch=convert(lists["a"]),  # mutated by in-place benchmarks
        a_list=lists["a"],
        shape=shape,
        i=shape[0] // 2,
        j=shape[1] // 2,
    )


def to_python(value):
    if isinstance(value, (ts.Tensor, np.ndarray)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def close(x, y):
    if isinstance(x, list) or isinstance(y, list):
        return (
            isinstance(x, list)
            and isinstance(y, list)
            and len(x) == len(y)
            and all(close(a, b) for a, b in zip(x, y))
        )
    return math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6)


def measure(fn, inputs, repeat):
    """Best-of-`repeat` seconds per call."""
    timer = timeit.Timer(lambda: fn(inputs))
    number, _ = timer.autorange()  # enough calls to take >= 0.2 s
    return min(timer.repeat(repeat=repeat, number=number)) / number


def format_time(seconds):
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def format_ratio(ours, theirs):
    """How tensorsc compares: '3.2x slower' or '1.5x faster'."""
    if ours >= theirs:
        return f"{ours / theirs:.2g}x slower"
    return f"{theirs / ours:.2g}x faster"


def run(names, sizes, repeat, baseline):
    results = []
    for size in sizes:
        shape = SIZES[size]
        lists = make_lists(shape)
        print(f"\n{size}: {shape[0]}x{shape[1]} ({shape[0] * shape[1]:,} elements)")
        header = f"{'op':<16}{'tensorsc':>12}{'numpy':>12}   {'tensorsc vs numpy':<20}"
        if baseline:
            header += "tensorsc vs baseline"
        print(header)
        print("-" * len(header))

        for name in names:
            fns = dict(zip(LIBRARIES, BENCHMARKS[name]))
            outputs = {lib: to_python(fns[lib](prepare(lib, lists, shape))) for lib in LIBRARIES}
            if not close(outputs["tensorsc"], outputs["numpy"]):
                sys.exit(f"error: {name} ({size}) gives different results in tensorsc and numpy")

            times = {lib: measure(fns[lib], prepare(lib, lists, shape), repeat) for lib in LIBRARIES}
            for lib, seconds in times.items():
                results.append({"op": name, "size": size, "library": lib, "seconds": seconds})

            line = (
                f"{name:<16}{format_time(times['tensorsc']):>12}{format_time(times['numpy']):>12}"
                f"   {format_ratio(times['tensorsc'], times['numpy']):<20}"
            )
            before = baseline.get((name, size))
            if before:
                line += format_ratio(times["tensorsc"], before)
            print(line, flush=True)
    return results


def load_baseline(path):
    with open(path) as f:
        data = json.load(f)
    return {
        (r["op"], r["size"]): r["seconds"] for r in data["results"] if r["library"] == "tensorsc"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", metavar="TEXT", help="only run benchmarks whose name contains TEXT")
    parser.add_argument("--size", choices=SIZES, action="append", help="sizes to run (repeatable; default: all)")
    parser.add_argument("--repeat", type=int, default=3, help="timing rounds per benchmark (default: 3)")
    parser.add_argument("--save", metavar="FILE", help="write results as JSON")
    parser.add_argument("--compare", metavar="FILE", help="also compare tensorsc against a saved run")
    args = parser.parse_args()

    if getattr(ts, "_debug_build", True):
        print(
            "warning: tensorsc is a debug build, so these numbers are meaningless.\n"
            "         rebuild with: maturin develop --release",
            file=sys.stderr,
        )

    names = [n for n in BENCHMARKS if not args.k or args.k in n]
    if not names:
        sys.exit(f"no benchmarks match {args.k!r}")
    sizes = args.size or list(SIZES)
    baseline = load_baseline(args.compare) if args.compare else {}

    print(f"tensorsc {getattr(ts, '__version__', 'dev')} vs numpy {np.__version__}, "
          f"Python {platform.python_version()} on {platform.machine()}")
    results = run(names, sizes, args.repeat, baseline)

    if args.save:
        meta = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorsc_debug_build": getattr(ts, "_debug_build", None),
        }
        with open(args.save, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nsaved {len(results)} results to {args.save}")


if __name__ == "__main__":
    main()

# Optimization work

Branch: optimizations. Base: main at 0a284c4. No new tensor features or dependencies.

## Changes, in order
1. Single-row traversal: recognize contiguous, broadcast-scalar and reversed flat layouts before allocating coalesced shape/stride vectors. Existing general strided fallback remains.
2. allclose: compare storage values directly for equal dtypes, converting each value to f64 without allocating converted tensors or a boolean mask. Mixed dtypes retain the previous f64 conversion semantics. Stop testing values after a mismatch. Correct matching-infinity and unequal-infinity behavior; NaNs remain unequal.
3. Matmul: reuse each B value across four output rows. Preserve ascending reduction order and use the original loop for small/vector products and remaining rows. An experimental cache-blocked implementation showed marginal benefit and was discarded.
4. Full reductions: fold contiguous full reductions into one scalar without an accumulator vector or output-stride layout. General axis/strided reductions retain their existing implementation.
5. Python scalar conversion: scalar tolist returns its Python value without a temporary vector. Storage guards are released before Python object creation. Singleton-axis tensors still return lists.

## Local measurements
Windows, Python 3.13.5, release build, NumPy 2.5.4. Inputs prepared outside timing. General benchmark reports median of five adaptive batches; matmul benchmark reports median of five batches of five calls. These are local observations, not portable performance guarantees. No changes to compiler flags, BLAS dependencies or thread configuration.

- Equal float32 allclose, 512x512: 1663 us before, 179 us after (about 9.3x faster).
- Equal float32 allclose, 128x128: 22.8 us before, 11.8 us after.
- Small full sum, 4x4: 0.39 us before, 0.21 us after. Large sums did not show a consistent improvement.
- Float32 matmul, 512x512: 11.30 ms before matrix tuning, 8.61 ms after (about 24% lower time).
- Float32 matmul, 256x256: 1.28 ms before matrix tuning, 1.00 ms after.
- Scalar float32 tolist: 74 ns before, 48 ns after (100000 calls per batch, median of five batches).
- Small add, 4x4: 0.75 us before, 0.67 us after. Large elementwise results varied with allocation/system effects; do not attribute all observed changes to stride setup.

Raw snapshots are in benchmarks/results/optimizations-*.json. The matrix snapshots isolate matrix tuning after layout/allclose work. optimizations-final.json was measured after reduction tuning and before scalar tolist tuning; it contains no scalar tolist case. Other snapshots are successive branch states, not independent one-change experiments. Existing NumPy baselines remain available for comparison; this branch does not claim to outperform NumPy across operations.

## Validation
- cargo test: 100 passed; one documentation example ignored.
- pytest after rebuilding the release extension: 511 passed.
- cargo fmt --check and git diff --check passed.
- New tests cover allclose across all dtypes, mixed dtypes, infinities, NaNs, broadcasting, empty/negative-stride/offset/transposed layouts, early/late mismatches; matmul row tails, batches, dtypes and gradients; Python scalar values/types versus singleton lists.

## Reproduce
Activate .venv, then run:

    maturin develop --release
    cargo test
    python -m pytest -q
    python benchmarks/optimization_baseline.py --save results.json
    python benchmarks/optimization_matmul.py matmul-results.json

## Remaining boundaries
External NumPy mutation still bypasses autograd version tracking; it was not changed by this performance work. Same-dtype allclose avoids bulk conversions, while mixed-dtype allclose still converts. General strided reductions, advanced-index selection buffers, graph accumulation, and the GIL/foreign-memory ownership contract retain their existing behavior. Further optimization should be measured independently on this branch, with aliasing and gradient regressions preserved.

# DLPack interoperability

The `dlpack-interop` branch adds dlpark 0.10.0 on top of `optimizations`.
It uses dlpark's current PyO3 0.30 support directly; no PyO3 downgrade or fork is needed.

## Python

    import numpy as np
    import tensorsc as ts
    import torch

    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    a = np.from_dlpack(x)          # shares CPU memory; NumPy chooses read-only
    b = torch.from_dlpack(x)       # shares CPU memory
    y = ts.from_dlpack(b)          # shares dense numeric data; no foreign graph
    owned = ts.from_dlpack(a, copy=True)

Existing `ts.from_numpy`, `t.numpy()`, `np.asarray(t)` and `memoryview(t)` keep their buffer-protocol implementations. DLPack adds cross-framework interoperability; it does not replace Tensor operators, constructors or autograd bindings.

`Tensor.__dlpack__` and `__dlpack_device__` implement the standard CPU protocol. dlpark handles legacy/versioned negotiation, capsules, single consumption and deleters. Each export creates fresh owned metadata and retains the underlying storage; consumers may outlive the source Python tensor. Exports are zero-copy only: `copy=True`, non-CPU destination devices, CPU stream arguments, and exporting a tracked tensor raise BufferError. Use `tensor.detach()` explicitly to leave the graph. A DLPack capsule can be consumed once; a producer tensor can export repeatedly.

## Import policy

`ts.from_dlpack(obj, copy=None)` accepts protocol producers or raw capsules. Only CPU device 0 and bool/uint8/int32/int64/float32/float64 are supported. Non-native/unsupported representations must be converted by the source library. A producer must provide valid initialized elements and honor the DLPack lifetime contract; raw metadata cannot prove allocation bounds.

- `copy=None`: share dense numeric layouts, including transposed/reversed views and offset subviews. Bool imports normalize bytes into an owned bool allocation. Gapped or otherwise non-dense layouts copy element by element; this avoids forming Rust typed slices over potentially uninitialized holes.
- `copy=False`: require sharing. Reject bool/non-dense imports needing conversion and versioned producer results explicitly flagged as copies. Empty tensors retain metadata/ownership without accessing data.
- `copy=True`: make a local owned contiguous copy, including for legacy producers that cannot negotiate a copy keyword.

Read-only versioned inputs remain read-only unless explicitly copied. Bool/expanded/read-only exports require the versioned ABI, since the legacy ABI cannot convey read-only flags. Consumers have different supported layouts; for example, do not assume another framework accepts negative strides just because tensorsc and DLPack can represent them.

## Autograd and concurrency

Import creates a tensor outside the source framework's graph. It can later become a tensorsc leaf with `requires_grad_()` if floating point. DLPack does not connect autograd graphs.

Storage now remembers that it has been exposed externally. Values saved for backward from externally shared storage are copied into independent snapshots. Export after a value has already been saved bumps the version, conservatively making that saved value invalid; backward raises an error even if the export was only read. This also covers existing NumPy buffer exports/imports. Ordinary internal-only tensors retain the existing version-counter path. Snapshot protection has a memory/time cost during training with foreign-shared tensors.

Foreign producers/consumers must not mutate concurrently with Rust storage access. Python calls hold the GIL and do not invoke Python while a storage slice is borrowed, but foreign native worker threads are outside those locks. DLPack does not provide synchronization between arbitrary CPU writers. The public unsafe Rust import API documents this requirement.

## Rust and features

`dlpack = ["dep:dlpark"]` enables `tensorsc::interop::dlpack` independently of Python. `python` enables DLPack plus dlpark's PyO3 feature. No CUDA, ndarray, candle or other dlpark backends are enabled. Default `cargo test` remains independent of both Python and dlpark.

The Rust exporter uses copied dynamic metadata plus an Arc<Storage> context. The unsafe importer accepts validated descriptor metadata and a retained owner; callers must uphold pointer, owner, flags and access-exclusivity contracts. Python negotiation lives in src/python/dlpack.rs; metadata/layout conversion lives in src/interop/dlpack.rs.

## Validation and measurements

- Default cargo test: 100 passed; one documentation example ignored.
- cargo test --features dlpack: 103 passed; one documentation example ignored.
- Rebuilt release extension and pytest: 545 passed, including NumPy 2.5.4 and PyTorch 2.14.1+cpu interoperability.
- Clippy passes with only the pre-existing `chunks_exact_to_as_chunks` lint in reduce.rs allowed. An unqualified -D warnings run reports that existing lint.
- cargo fmt --check and git diff --check pass.

The release conversion benchmark prepares inputs and validates values/pointer sharing outside timing, then reports the median of five batches of 10000 calls. On this Windows/Python 3.13 machine, dense NumPy DLPack imports took about 1.2 us versus 0.3-0.4 us for the existing buffer bridge. Export to NumPy via DLPack took about 0.31-0.32 us. Sizes of 16 and 262144 elements had similar conversion times, consistent with verified zero-copy sharing. These results support retaining the buffer path and adding DLPack for framework interoperability, not claiming every conversion becomes faster.

Reproduce:

    maturin develop --release
    cargo test
    cargo test --features dlpack
    python -m pytest -q
    python benchmarks/bench_dlpack.py --save benchmarks/results/dlpack-interop.json

NumPy is needed for Python tests; CPU PyTorch is an optional test dependency. Tests skip the PyTorch-specific cases when it is unavailable. Raw results are checked into benchmarks/results/dlpack-interop.json.

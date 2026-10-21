//! Typed elementwise loops over strided memory.
//!
//! These are generic over the element type and assume their inputs already
//! have the right dtype (callers convert first; `Storage::read` asserts it).
//! Deciding result dtypes is `ops.rs`'s job. Each loop works a row at a time
//! (see `strided.rs`), with a slice fast path for stride-1 rows.

use std::sync::Arc;

use crate::dtype::Element;
use crate::error::{Result, TensorError};
use crate::ops::broadcast_shapes;
use crate::strided::{for_each_row, positions};
use crate::tensor::Tensor;

// Locking rule: code that holds two storage locks at once takes them in
// address order, so two threads can never wait on each other's locks.

/// Runs `f` with read access to both tensors' elements. When they share
/// storage it takes the lock once: read-locking the same `RwLock` twice from
/// one thread may deadlock or panic.
pub(crate) fn read_both<T: Element, R>(
    a: &Tensor,
    b: &Tensor,
    f: impl FnOnce(&[T], &[T]) -> R,
) -> R {
    if a.shares_storage(b) {
        let data = a.storage.read::<T>();
        return f(&data, &data);
    }
    let (da, db);
    if Arc::as_ptr(&a.storage) < Arc::as_ptr(&b.storage) {
        da = a.storage.read::<T>();
        db = b.storage.read::<T>();
    } else {
        db = b.storage.read::<T>();
        da = a.storage.read::<T>();
    }
    f(&da, &db)
}

impl Tensor {
    /// Fails for views where several elements share one memory location
    /// (stride 0 along an axis longer than 1, from `expand`): writing through
    /// them elementwise would be ambiguous.
    pub(crate) fn ensure_writable_elementwise(&self) -> Result<()> {
        let overlapping = self
            .shape
            .iter()
            .zip(&self.strides)
            .any(|(&dim, &stride)| dim > 1 && stride == 0);
        if overlapping {
            return Err(TensorError::InvalidArgument(
                "cannot write in place to an expanded tensor: several of its elements \
                 share one memory location (call .contiguous() first)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// `f` applied to every element (of dtype `S`), in logical row-major order.
    pub(crate) fn map_to_vec<S: Element, U>(&self, f: impl Fn(S) -> U) -> Vec<U> {
        let data = self.storage.read::<S>();
        let mut out = Vec::with_capacity(self.numel());
        for_each_row(
            &self.shape,
            [&self.strides],
            [self.offset],
            |[p], len, [s]| match s {
                1 => out.extend(data[p..p + len].iter().map(|&x| f(x))),
                _ => out.extend(positions(p, len, s).map(|i| f(data[i]))),
            },
        );
        out
    }

    /// A new tensor of `f` applied to every element.
    pub(crate) fn map_elements<S: Element, U: Element>(&self, f: impl Fn(S) -> U) -> Tensor {
        Tensor::from_vec(self.map_to_vec(f), &self.shape).expect("length matches shape")
    }

    /// True if `pred` holds for any element (of dtype `T`).
    pub(crate) fn any_element<T: Element>(&self, pred: impl Fn(T) -> bool) -> bool {
        let data = self.storage.read::<T>();
        let mut found = false;
        for_each_row(
            &self.shape,
            [&self.strides],
            [self.offset],
            |[p], len, [s]| {
                found = found || positions(p, len, s).any(|i| pred(data[i]));
            },
        );
        found
    }

    /// Sets every element of this view to `value`.
    pub(crate) fn fill_with<T: Element>(&self, value: T) -> Result<()> {
        let mut data = self.storage.write::<T>()?;
        for_each_row(
            &self.shape,
            [&self.strides],
            [self.offset],
            |[p], len, [s]| match s {
                1 => data[p..p + len].fill(value),
                _ => positions(p, len, s).for_each(|i| data[i] = value),
            },
        );
        Ok(())
    }

    /// A new tensor combining `self` and `other` (both of dtype `A`)
    /// elementwise after broadcasting.
    pub(crate) fn zip_map<A: Element, U: Element>(
        &self,
        other: &Tensor,
        f: impl Fn(A, A) -> U,
    ) -> Result<Tensor> {
        let shape = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&shape)?;
        let b = other.broadcast_to(&shape)?;
        let mut out = Vec::with_capacity(a.numel());
        read_both::<A, _>(&a, &b, |da, db| {
            for_each_row(
                &shape,
                [&a.strides, &b.strides],
                [a.offset, b.offset],
                |[pa, pb], len, [sa, sb]| match (sa, sb) {
                    (1, 1) => out.extend(
                        da[pa..pa + len]
                            .iter()
                            .zip(&db[pb..pb + len])
                            .map(|(&x, &y)| f(x, y)),
                    ),
                    // A broadcast scalar on one side, e.g. `a * 2`.
                    (1, 0) => {
                        let y = db[pb];
                        out.extend(da[pa..pa + len].iter().map(|&x| f(x, y)));
                    }
                    (0, 1) => {
                        let x = da[pa];
                        out.extend(db[pb..pb + len].iter().map(|&y| f(x, y)));
                    }
                    _ => out.extend(
                        positions(pa, len, sa)
                            .zip(positions(pb, len, sb))
                            .map(|(i, j)| f(da[i], db[j])),
                    ),
                },
            )
        });
        Tensor::from_vec(out, &shape)
    }

    /// In-place [`Tensor::zip_map`]: writes `f(self, other)` back through this
    /// view. Both have dtype `A`; `other` must broadcast to this tensor's
    /// shape (not the reverse).
    pub(crate) fn zip_map_assign<A: Element>(
        &self,
        other: &Tensor,
        f: impl Fn(A, A) -> A,
    ) -> Result<()> {
        self.ensure_writable_elementwise()?;
        let b = other
            .broadcast_to(&self.shape)
            .map_err(|_| TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            })?;
        if self.overlaps_memory(&b) {
            // Writing while reading the same memory could read values that were
            // already overwritten (`a[1:] += a[:-1]`), and would alias a mutable
            // slice, so work from a copy, as NumPy does.
            return self.zip_map_assign(&b.deep_copy(), f);
        }
        let mut dst;
        let src;
        if Arc::as_ptr(&self.storage) < Arc::as_ptr(&b.storage) {
            dst = self.storage.write::<A>()?;
            src = b.storage.read::<A>();
        } else {
            src = b.storage.read::<A>();
            dst = self.storage.write::<A>()?;
        }
        for_each_row(
            &self.shape,
            [&self.strides, &b.strides],
            [self.offset, b.offset],
            |[pa, pb], len, [sa, sb]| match (sa, sb) {
                (1, 1) => {
                    for (x, &y) in dst[pa..pa + len].iter_mut().zip(&src[pb..pb + len]) {
                        *x = f(*x, y);
                    }
                }
                (1, 0) => {
                    let y = src[pb];
                    for x in &mut dst[pa..pa + len] {
                        *x = f(*x, y);
                    }
                }
                _ => {
                    for (i, j) in positions(pa, len, sa).zip(positions(pb, len, sb)) {
                        dst[i] = f(dst[i], src[j]);
                    }
                }
            },
        );
        Ok(())
    }

    /// `f(cond, a, b)` elementwise after broadcasting all three, where `cond`
    /// is bool and `a`, `b` have dtype `A`.
    pub(crate) fn zip3_map<A: Element>(
        cond: &Tensor,
        a: &Tensor,
        b: &Tensor,
        f: impl Fn(bool, A, A) -> A,
    ) -> Result<Tensor> {
        let shape = broadcast_shapes(&broadcast_shapes(&cond.shape, &a.shape)?, &b.shape)?;
        // Read the condition up front, so at most two locks are held below.
        let picks = cond.broadcast_to(&shape)?.to_vec::<bool>();
        let (a, b) = (a.broadcast_to(&shape)?, b.broadcast_to(&shape)?);
        let mut out = Vec::with_capacity(picks.len());
        read_both::<A, _>(&a, &b, |da, db| {
            for_each_row(
                &shape,
                [&a.strides, &b.strides],
                [a.offset, b.offset],
                |[pa, pb], len, [sa, sb]| {
                    let row = &picks[out.len()..out.len() + len];
                    out.extend(
                        row.iter()
                            .zip(positions(pa, len, sa).zip(positions(pb, len, sb)))
                            .map(|(&c, (i, j))| f(c, da[i], db[j])),
                    )
                },
            )
        });
        Tensor::from_vec(out, &shape)
    }
}

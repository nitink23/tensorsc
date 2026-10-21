//! Reverse-mode automatic differentiation, with PyTorch's model.
//!
//! A tensor that requires grad carries [`AutogradMeta`]. Leaves (tensors the
//! user created and marked) accumulate `.grad`; every other such tensor was
//! produced by an operation and points at a [`Node`]: that operation's
//! backward function plus links to its inputs' metadata. [`Tensor::backward`]
//! walks those links in reverse topological order.
//!
//! Differentiable operations compute their forward pass with recording off,
//! then call [`record`] once with their real inputs. `record` builds the
//! backward closure only if grad mode is on and an input requires grad, so
//! code that never asks for gradients pays almost nothing.
//!
//! Two safety rules, as in PyTorch: in-place writes to a tensor that requires
//! grad are errors (outside `no_grad`), and tensors saved for the backward
//! pass remember their storage's version, so modifying one in place before
//! `backward()` is reported instead of producing wrong gradients.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtype::DType;
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

// ---- grad mode ----

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Turns recording on or off for this thread; returns the previous setting.
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| g.replace(enabled))
}

/// Runs `f` without recording operations (like `with torch.no_grad():`).
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_grad_enabled(self.0);
        }
    }
    let _restore = Restore(set_grad_enabled(false));
    f()
}

// ---- the graph ----

/// Gradient of the output in, gradients of each input out (`None` where an
/// input doesn't need one).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync>;

pub(crate) struct Node {
    name: &'static str,
    /// One per input; `None` for inputs that don't require grad.
    inputs: Vec<Option<Arc<AutogradMeta>>>,
    backward: BackwardFn,
}

pub struct AutogradMeta {
    /// Only ever false for leaves that had `requires_grad` turned off.
    requires_grad: AtomicBool,
    /// Accumulated gradient (leaves only, as in PyTorch).
    grad: Mutex<Option<Tensor>>,
    /// The operation that produced this tensor; `None` for leaves.
    grad_fn: Option<Arc<Node>>,
}

impl fmt::Debug for AutogradMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutogradMeta")
            .field("requires_grad", &self.requires_grad.load(Ordering::Relaxed))
            .field("grad_fn", &self.grad_fn.as_ref().map(|n| n.name))
            .finish_non_exhaustive()
    }
}

/// A tensor kept for the backward pass, with the version of its storage at
/// the time, so later in-place modification is caught.
pub(crate) struct Saved {
    tensor: Tensor,
    version: u64,
}

impl Saved {
    pub(crate) fn new(tensor: &Tensor) -> Saved {
        Saved {
            tensor: tensor.detach(),
            version: tensor.storage.version(),
        }
    }

    pub(crate) fn get(&self) -> Result<&Tensor> {
        if self.tensor.storage.version() != self.version {
            return Err(TensorError::InvalidArgument(
                "a tensor needed to compute gradients was modified in place after it was \
                 used; clone() it before modifying, or modify it before using it"
                    .into(),
            ));
        }
        Ok(&self.tensor)
    }

    /// Shape and dtype don't change in place, so they need no version check.
    pub(crate) fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    pub(crate) fn dtype(&self) -> DType {
        self.tensor.dtype()
    }
}

/// Attaches `out` to the graph if grad mode is on and any input requires
/// grad. Only in that case is `backward` called, with the output (to save,
/// if the gradient needs it) and which inputs need gradients, to build the
/// backward function.
pub(crate) fn record<const N: usize>(
    out: Tensor,
    inputs: [&Tensor; N],
    name: &'static str,
    backward: impl FnOnce(&Tensor, [bool; N]) -> BackwardFn,
) -> Tensor {
    if !is_grad_enabled() || !inputs.iter().any(|t| t.requires_grad()) {
        return out;
    }
    let needs = inputs.map(|t| t.requires_grad());
    let backward = backward(&out, needs);
    let node = Node {
        name,
        inputs: inputs
            .iter()
            .map(|t| t.autograd_meta().filter(|_| t.requires_grad()))
            .collect(),
        backward,
    };
    let meta = AutogradMeta {
        requires_grad: AtomicBool::new(true),
        grad: Mutex::new(None),
        grad_fn: Some(Arc::new(node)),
    };
    out.attach_meta(meta)
}

/// Sums `grad` down to `shape`, undoing broadcasting: leading axes that were
/// added and axes that were stretched from length 1.
pub(crate) fn sum_to(grad: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if grad.shape() == shape {
        return Ok(grad.shallow_clone());
    }
    let lead = grad.ndim() - shape.len();
    let mut axes: Vec<isize> = (0..lead as isize).collect();
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 1 && grad.shape()[lead + axis] != 1 {
            axes.push((lead + axis) as isize);
        }
    }
    let summed = grad.sum(Some(&axes), true)?;
    summed.reshape(&shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
}

/// `grad` reshaped/summed to match `input`'s shape and dtype.
pub(crate) fn fit(grad: &Tensor, input: &Saved) -> Result<Tensor> {
    Ok(sum_to(grad, input.shape())?.to_dtype(input.dtype()))
}

/// `Ok(Some(f()))` if the gradient is needed, else `Ok(None)`.
pub(crate) fn when(need: bool, f: impl FnOnce() -> Result<Tensor>) -> Result<Option<Tensor>> {
    if need { f().map(Some) } else { Ok(None) }
}

/// A 0-d constant of `like`'s dtype.
pub(crate) fn constant(value: f64, like: &Tensor) -> Tensor {
    Tensor::full(&[], value, like.dtype())
}

// ---- tensor API ----

impl Tensor {
    pub(crate) fn autograd_meta(&self) -> Option<Arc<AutogradMeta>> {
        self.autograd.get().cloned()
    }

    fn attach_meta(self, meta: AutogradMeta) -> Tensor {
        let _ = self.autograd.set(Arc::new(meta));
        self
    }

    pub fn requires_grad(&self) -> bool {
        self.autograd
            .get()
            .is_some_and(|m| m.requires_grad.load(Ordering::Relaxed))
    }

    /// True for tensors the user created (not produced by a recorded operation).
    pub fn is_leaf(&self) -> bool {
        self.autograd.get().is_none_or(|m| m.grad_fn.is_none())
    }

    /// Name of the operation that produced this tensor, e.g. `"MulBackward"`.
    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.autograd.get()?.grad_fn.as_ref().map(|n| n.name)
    }

    /// Marks a leaf as needing gradients (or not). Only float tensors can.
    pub fn set_requires_grad(&self, requires: bool) -> Result<()> {
        if !self.is_leaf() {
            return Err(TensorError::InvalidArgument(
                "requires_grad can only be changed on leaf tensors; use detach() \
                 to get a tensor outside the graph"
                    .into(),
            ));
        }
        if requires && !self.dtype().is_float() {
            return Err(TensorError::InvalidArgument(format!(
                "only floating point tensors can require gradients, not {}",
                self.dtype()
            )));
        }
        match self.autograd.get() {
            Some(meta) => meta.requires_grad.store(requires, Ordering::Relaxed),
            None if requires => {
                let _ = self.autograd.set(Arc::new(AutogradMeta {
                    requires_grad: AtomicBool::new(true),
                    grad: Mutex::new(None),
                    grad_fn: None,
                }));
            }
            None => {}
        }
        Ok(())
    }

    /// The gradient accumulated by `backward()` (leaves only).
    pub fn grad(&self) -> Option<Tensor> {
        self.autograd
            .get()?
            .grad
            .lock()
            .expect("grad lock poisoned")
            .as_ref()
            .map(Tensor::shallow_clone)
    }

    /// Replaces the accumulated gradient; `None` clears it (zeroing gradients).
    pub fn set_grad(&self, grad: Option<Tensor>) -> Result<()> {
        if let Some(g) = &grad
            && g.shape() != self.shape()
        {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: g.shape().to_vec(),
            });
        }
        match self.autograd.get() {
            Some(meta) => *meta.grad.lock().expect("grad lock poisoned") = grad,
            None if grad.is_none() => {}
            None => {
                return Err(TensorError::InvalidArgument(
                    "can't set .grad on a tensor that doesn't require grad".into(),
                ));
            }
        }
        Ok(())
    }

    /// The same data, outside the graph: shares memory, never requires grad.
    pub fn detach(&self) -> Tensor {
        self.with_layout(self.shape.clone(), self.strides.clone(), self.offset)
    }

    /// Computes gradients of this tensor with respect to every leaf that
    /// requires grad, adding them to the leaves' `.grad`. `grad` is the
    /// gradient of whatever this tensor feeds into; it may be omitted for
    /// one-element tensors (it's then 1). The graph is kept, so calling
    /// `backward()` again accumulates again.
    pub fn backward(&self, grad: Option<&Tensor>) -> Result<()> {
        let Some(root) = self.autograd_meta().filter(|_| self.requires_grad()) else {
            return Err(TensorError::InvalidArgument(
                "this tensor doesn't require grad (no input to it has requires_grad=True)".into(),
            ));
        };
        let seed = match grad {
            Some(g) if g.shape() != self.shape() => {
                return Err(TensorError::ShapeMismatch {
                    expected: self.shape().to_vec(),
                    got: g.shape().to_vec(),
                });
            }
            Some(g) => g.to_dtype(self.dtype()).detach(),
            None if self.numel() == 1 => Tensor::ones(self.shape(), self.dtype()),
            None => {
                return Err(TensorError::InvalidArgument(
                    "backward() needs a gradient argument for tensors with more than one \
                     element (pass ones_like(t) to sum over everything)"
                        .into(),
                ));
            }
        };
        no_grad(|| run_backward(&root, seed))
    }
}

/// Every node reachable from `root`, inputs before the nodes that use them.
fn topological_order(root: &Arc<AutogradMeta>) -> Vec<Arc<AutogradMeta>> {
    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Iterative post-order DFS: (meta, inputs already pushed?).
    let mut stack = vec![(Arc::clone(root), false)];
    while let Some((meta, expanded)) = stack.pop() {
        if expanded {
            order.push(meta);
            continue;
        }
        if !seen.insert(Arc::as_ptr(&meta)) {
            continue;
        }
        stack.push((Arc::clone(&meta), true));
        if let Some(node) = &meta.grad_fn {
            for input in node.inputs.iter().flatten() {
                if !seen.contains(&Arc::as_ptr(input)) {
                    stack.push((Arc::clone(input), false));
                }
            }
        }
    }
    order
}

fn run_backward(root: &Arc<AutogradMeta>, seed: Tensor) -> Result<()> {
    let mut pending: HashMap<*const AutogradMeta, Tensor> = HashMap::new();
    pending.insert(Arc::as_ptr(root), seed);
    for meta in topological_order(root).iter().rev() {
        let Some(grad) = pending.remove(&Arc::as_ptr(meta)) else {
            continue;
        };
        match &meta.grad_fn {
            None => {
                let mut slot = meta.grad.lock().expect("grad lock poisoned");
                *slot = Some(match slot.take() {
                    Some(existing) => existing.add(&grad)?,
                    // A fresh copy: `.grad` must not alias anything else.
                    None => grad.deep_copy(),
                });
            }
            Some(node) => {
                let input_grads = (node.backward)(&grad)?;
                debug_assert_eq!(input_grads.len(), node.inputs.len());
                for (input, input_grad) in node.inputs.iter().zip(input_grads) {
                    let (Some(input), Some(input_grad)) = (input, input_grad) else {
                        continue;
                    };
                    let key = Arc::as_ptr(input);
                    let total = match pending.remove(&key) {
                        Some(existing) => existing.add(&input_grad)?,
                        None => input_grad,
                    };
                    pending.insert(key, total);
                }
            }
        }
    }
    Ok(())
}

/// For new tensors: an empty autograd slot.
pub(crate) fn empty_slot() -> OnceLock<Arc<AutogradMeta>> {
    OnceLock::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::where_;
    use crate::views::TensorIndex;

    /// Deterministic values in [-1, 1).
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn data(shape: &[usize], seed: u64) -> Tensor {
        Tensor::from_vec(values(shape.iter().product(), seed), shape).unwrap()
    }

    fn leaf(shape: &[usize], seed: u64) -> Tensor {
        let t = data(shape, seed);
        t.set_requires_grad(true).unwrap();
        t
    }

    /// A leaf with values in [0.5, 1.5), for functions that need positive input.
    fn positive_leaf(shape: &[usize], seed: u64) -> Tensor {
        let t = data(shape, seed)
            .abs()
            .add(&Tensor::scalar(0.5f64))
            .unwrap()
            .detach();
        t.set_requires_grad(true).unwrap();
        t
    }

    /// Checks `f`'s gradients against central differences. The output is
    /// weighted by fixed random values and summed, so every element counts.
    fn gradcheck(inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Result<Tensor>) {
        let out = f(inputs).unwrap();
        assert!(out.requires_grad(), "output should require grad");
        let weights = data(out.shape(), 99);
        let loss = |xs: &[Tensor]| {
            no_grad(|| {
                f(xs)
                    .unwrap()
                    .mul(&weights)
                    .unwrap()
                    .sum(None, false)
                    .unwrap()
            })
            .item()
            .unwrap()
            .to_f64()
        };
        out.mul(&weights)
            .unwrap()
            .sum(None, false)
            .unwrap()
            .backward(None)
            .unwrap();
        for (k, x) in inputs.iter().enumerate() {
            let analytic = x.grad().expect("input got a gradient").to_vec::<f64>();
            assert_eq!(x.grad().unwrap().shape(), x.shape());
            let base = x.to_vec::<f64>();
            for i in 0..x.numel() {
                let at = |delta: f64| {
                    let xs: Vec<Tensor> = inputs
                        .iter()
                        .enumerate()
                        .map(|(j, t)| {
                            let mut v = if j == k {
                                base.clone()
                            } else {
                                t.to_vec::<f64>()
                            };
                            if j == k {
                                v[i] += delta;
                            }
                            Tensor::from_vec(v, t.shape()).unwrap()
                        })
                        .collect();
                    loss(&xs)
                };
                let eps = 1e-6;
                let numeric = (at(eps) - at(-eps)) / (2.0 * eps);
                assert!(
                    (numeric - analytic[i]).abs() <= 1e-5 * (1.0 + numeric.abs()),
                    "input {k}, element {i}: numeric {numeric}, analytic {}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn arithmetic_with_broadcasting() {
        let (a, b) = (leaf(&[2, 3], 1), leaf(&[3], 2));
        gradcheck(&[a.detach_leaf(), b.detach_leaf()], |x| x[0].add(&x[1]));
        gradcheck(&[a.detach_leaf(), b.detach_leaf()], |x| x[0].sub(&x[1]));
        gradcheck(&[a.detach_leaf(), b.detach_leaf()], |x| x[0].mul(&x[1]));
        let c = positive_leaf(&[2, 1], 3);
        gradcheck(&[a.detach_leaf(), c], |x| x[0].div(&x[1]));
        gradcheck(&[a.detach_leaf(), b.detach_leaf()], |x| x[0].maximum(&x[1]));
        gradcheck(&[a.detach_leaf(), b.detach_leaf()], |x| x[0].minimum(&x[1]));
    }

    #[test]
    fn powers() {
        let (base, exponent) = (positive_leaf(&[2, 3], 4), leaf(&[3], 5));
        gradcheck(&[base.detach_leaf(), exponent], |x| x[0].pow(&x[1]));
        gradcheck(&[base], |x| x[0].pow(&Tensor::scalar(3.0f64)));
    }

    #[test]
    fn unary_functions() {
        let x = leaf(&[2, 3], 6);
        let p = positive_leaf(&[2, 3], 7);
        for f in [
            Tensor::exp,
            Tensor::sin,
            Tensor::cos,
            Tensor::tanh,
            Tensor::sigmoid,
            Tensor::abs,
        ] {
            gradcheck(&[x.detach_leaf()], |t| Ok(f(&t[0])));
        }
        gradcheck(&[p.detach_leaf()], |t| Ok(t[0].log()));
        gradcheck(&[p], |t| Ok(t[0].sqrt()));
        gradcheck(&[x.detach_leaf()], |t| t[0].neg());
        gradcheck(&[x.detach_leaf()], |t| t[0].relu());
        let (lo, hi) = (Tensor::scalar(-0.3f64), Tensor::scalar(0.4f64));
        gradcheck(&[x.detach_leaf()], |t| t[0].clamp(Some(&lo), Some(&hi)));
        let cond = data(&[2, 3], 8).gt(&Tensor::scalar(0.0f64)).unwrap();
        gradcheck(&[x.detach_leaf(), leaf(&[3], 9)], |t| {
            where_(&cond, &t[0], &t[1])
        });
    }

    #[test]
    fn reductions() {
        let x = leaf(&[2, 3, 4], 10);
        gradcheck(&[x.detach_leaf()], |t| t[0].sum(None, false));
        gradcheck(&[x.detach_leaf()], |t| t[0].sum(Some(&[0, 2]), false));
        gradcheck(&[x.detach_leaf()], |t| t[0].mean(Some(&[1]), true));
        gradcheck(&[x.detach_leaf()], |t| t[0].max(Some(&[2]), false));
        gradcheck(&[x.detach_leaf()], |t| t[0].min(None, false));
    }

    #[test]
    fn max_splits_ties() {
        let x = Tensor::from_vec(vec![1.0f64, 3.0, 3.0, 2.0], &[4]).unwrap();
        x.set_requires_grad(true).unwrap();
        x.max(None, false).unwrap().backward(None).unwrap();
        assert_eq!(x.grad().unwrap().to_vec::<f64>(), vec![0.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn matmul_in_every_shape() {
        for (a, b) in [
            (vec![2, 3], vec![3, 4]),
            (vec![3], vec![3, 4]),
            (vec![2, 3], vec![3]),
            (vec![3], vec![3]),
            (vec![2, 1, 2, 3], vec![4, 3, 2]),
            (vec![5, 2, 3], vec![3]),
        ] {
            gradcheck(&[leaf(&a, 11), leaf(&b, 12)], |t| t[0].matmul(&t[1]));
        }
    }

    #[test]
    fn views() {
        let x = leaf(&[2, 3, 4], 13);
        let checks: Vec<fn(&Tensor) -> Result<Tensor>> = vec![
            |t| t.reshape(&[4, -1]),
            |t| t.permute(&[2, 0, 1]),
            |t| t.transpose(0, 2),
            |t| Ok(t.reversed_axes()),
            |t| t.unsqueeze(1)?.squeeze(None),
            |t| t.flatten(1, 2),
            |t| t.select(1, -1),
            |t| t.slice(2, None, None, -2),
            |t| {
                t.index(&[
                    TensorIndex::NewAxis,
                    TensorIndex::Index(0),
                    TensorIndex::FULL,
                ])
            },
            |t| t.reversed_axes().contiguous().reshape(&[-1]),
            |t| t.select(2, 0)?.unsqueeze(-1)?.expand(&[2, 3, 5]),
        ];
        for f in checks {
            gradcheck(&[x.detach_leaf()], |t| f(&t[0]));
        }
    }

    #[test]
    fn advanced_indexing() {
        let x = leaf(&[4, 3], 14);
        let picks = Tensor::from_vec(vec![2i64, 0, 2], &[3]).unwrap();
        gradcheck(&[x.detach_leaf()], |t| t[0].index_select(0, &picks));
        gradcheck(&[x.detach_leaf()], |t| t[0].index_select(1, &picks));
        let mask = data(&[4, 3], 15).gt(&Tensor::scalar(0.0f64)).unwrap();
        gradcheck(&[x], |t| t[0].index_mask(&mask));
    }

    #[test]
    fn a_small_network() {
        let (x, w, b) = (data(&[5, 3], 16), leaf(&[3, 4], 17), leaf(&[4], 18));
        gradcheck(&[w, b], |t| {
            Ok(x.matmul(&t[0])?
                .add(&t[1])?
                .tanh()
                .sum(Some(&[1]), false)?
                .sigmoid())
        });
    }

    #[test]
    fn gradients_accumulate_and_can_be_cleared() {
        let x = Tensor::from_vec(vec![2.0f32], &[1]).unwrap();
        x.set_requires_grad(true).unwrap();
        // y = x*x + x reaches x along three paths: dy/dx = 2x + 1.
        let y = x.mul(&x).unwrap().add(&x).unwrap();
        y.backward(None).unwrap();
        assert_eq!(x.grad().unwrap().to_vec::<f32>(), vec![5.0]);
        y.backward(None).unwrap();
        assert_eq!(x.grad().unwrap().to_vec::<f32>(), vec![10.0], "accumulates");
        x.set_grad(None).unwrap();
        assert!(x.grad().is_none());
        assert_eq!(x.grad().map(|g| g.dtype()), None);
    }

    #[test]
    fn graph_metadata() {
        let x = leaf(&[2], 19);
        let y = x.exp();
        assert!(x.is_leaf() && !y.is_leaf());
        assert_eq!(y.grad_fn_name(), Some("ExpOp"));
        assert!(y.grad().is_none(), "only leaves keep .grad");
        assert!(!y.detach().requires_grad());
        assert!(
            y.gt(&x).unwrap().grad_fn_name().is_none(),
            "comparisons aren't differentiable"
        );
        assert!(!x.argmax(None, false).unwrap().requires_grad());
    }

    #[test]
    fn no_grad_records_nothing() {
        let x = leaf(&[2], 20);
        let y = no_grad(|| x.mul(&x).unwrap());
        assert!(!y.requires_grad());
        assert!(is_grad_enabled(), "restored afterwards");
    }

    #[test]
    fn errors() {
        let ints = Tensor::from_vec(vec![1i64], &[1]).unwrap();
        assert!(ints.set_requires_grad(true).is_err());
        let x = leaf(&[2], 21);
        let y = x.exp();
        assert!(y.set_requires_grad(false).is_err(), "only leaves");
        assert!(y.backward(None).is_err(), "needs a gradient for 2 elements");
        assert!(y.backward(Some(&Tensor::ones(&[3], DType::F64))).is_err());
        assert!(
            data(&[2], 1).backward(None).is_err(),
            "doesn't require grad"
        );
        // In-place writes to tracked tensors are refused, except under no_grad.
        assert!(x.add_(&Tensor::scalar(1.0f64)).is_err());
        assert!(x.fill(0.0).is_err());
        no_grad(|| x.sub_(&Tensor::scalar(1.0f64))).unwrap();
        assert!(x.prod(None, false).is_err(), "prod has no gradient yet");
    }

    #[test]
    fn modifying_a_saved_tensor_is_caught() {
        let x = leaf(&[2], 22);
        let y = x.exp();
        let z = y.mul(&y).unwrap().sum(None, false).unwrap();
        no_grad(|| y.add_(&Tensor::scalar(1.0f64))).unwrap();
        let err = z.backward(None).unwrap_err();
        assert!(err.to_string().contains("modified in place"), "{err}");
    }

    #[test]
    fn gradients_keep_the_input_dtype() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        x.set_requires_grad(true).unwrap();
        let y = x.to_dtype(DType::F64).mul(&Tensor::scalar(3.0f64)).unwrap();
        y.sum(None, false).unwrap().backward(None).unwrap();
        let g = x.grad().unwrap();
        assert_eq!(g.dtype(), DType::F32);
        assert_eq!(g.to_vec::<f32>(), vec![3.0, 3.0]);
    }

    impl Tensor {
        /// A fresh leaf with the same values (tests reuse inputs across checks).
        fn detach_leaf(&self) -> Tensor {
            let t = self.detach().deep_copy();
            t.set_requires_grad(true).unwrap();
            t
        }
    }
}

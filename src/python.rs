//! Python bindings. Everything here converts between Python objects and the
//! core types; tensor logic itself belongs in the core modules.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyFloat, PyIterator, PyList, PySlice, PyTuple};

use crate::error::TensorError;
use crate::tensor::Tensor;
use crate::views::TensorIndex;

impl From<TensorError> for PyErr {
    fn from(err: TensorError) -> PyErr {
        match err {
            TensorError::IndexRankMismatch { .. }
            | TensorError::TooManyIndices { .. }
            | TensorError::IndexOutOfBounds { .. }
            | TensorError::AxisOutOfRange { .. } => PyIndexError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

#[pyclass(name = "Tensor", module = "tensorsc", frozen)]
pub struct PyTensor {
    inner: Tensor,
}

impl From<Tensor> for PyTensor {
    fn from(inner: Tensor) -> Self {
        PyTensor { inner }
    }
}

/// Walks a nested list/tuple, recording its shape and flattening its numbers.
/// Every list at the same depth must have the same length.
fn flatten_nested(
    obj: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    data: &mut Vec<f32>,
) -> PyResult<()> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let len = obj.len()?;
        if depth == shape.len() {
            // First list seen at this depth sets the axis length, unless numbers
            // already appeared at a shallower depth.
            if !data.is_empty() {
                return Err(ragged());
            }
            shape.push(len);
        } else if shape[depth] != len {
            return Err(ragged());
        }
        for item in obj.try_iter()? {
            flatten_nested(&item?, depth + 1, shape, data)?;
        }
        Ok(())
    } else {
        if depth != shape.len() {
            return Err(ragged());
        }
        let value = obj.extract::<f32>().map_err(|_| {
            let type_name = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".into());
            PyTypeError::new_err(format!("tensor elements must be numbers, got {type_name}"))
        })?;
        data.push(value);
        Ok(())
    }
}

fn ragged() -> PyErr {
    PyValueError::new_err("nested sequences must be rectangular (every row the same length)")
}

fn tensor_from_object(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(other) = obj.cast::<PyTensor>() {
        return Ok(other.get().inner.deep_copy());
    }
    let mut shape = Vec::new();
    let mut data = Vec::new();
    flatten_nested(obj, 0, &mut shape, &mut data)?;
    Ok(Tensor::from_vec(data, &shape)?)
}

/// Accepts both `f(2, 3)` and `f((2, 3))`, like PyTorch.
fn parse_dims(args: &Bound<'_, PyTuple>) -> PyResult<Vec<isize>> {
    // Check the type rather than trying one form and falling back: a failed
    // extraction builds a Python exception, which costs more than the work.
    if args.len() == 1 {
        let first = args.get_item(0)?;
        if first.is_instance_of::<PyList>() || first.is_instance_of::<PyTuple>() {
            return first.extract();
        }
    }
    args.iter().map(|dim| dim.extract()).collect()
}

/// Like [`parse_dims`], for sizes that can't be negative.
fn parse_shape(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    parse_dims(args)?
        .into_iter()
        .map(|d| {
            usize::try_from(d)
                .map_err(|_| PyValueError::new_err(format!("negative dimension {d} in shape")))
        })
        .collect()
}

fn parse_index_item(obj: &Bound<'_, PyAny>) -> PyResult<TensorIndex> {
    if obj.is_none() {
        return Ok(TensorIndex::NewAxis);
    }
    if obj.is_instance_of::<PyEllipsis>() {
        return Ok(TensorIndex::Ellipsis);
    }
    if let Ok(slice) = obj.cast::<PySlice>() {
        // CPython's own slice decoding (NumPy uses it too): fills in defaults
        // for missing bounds and clamps huge ones, as list slicing does.
        let (mut start, mut stop, mut step) = (0, 0, 0);
        // SAFETY: `slice` is a live slice object and the pointers are to locals.
        if unsafe { ffi::PySlice_Unpack(slice.as_ptr(), &mut start, &mut stop, &mut step) } < 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        return Ok(TensorIndex::Slice {
            start: Some(start),
            stop: Some(stop),
            step,
        });
    }
    obj.extract::<isize>()
        .map(TensorIndex::Index)
        .map_err(|_| PyTypeError::new_err("tensor indices must be integers, slices, None or ..."))
}

/// A key like `1`, `1:3` or `(0, ..., None)` as a list of index items.
fn parse_index(key: &Bound<'_, PyAny>) -> PyResult<Vec<TensorIndex>> {
    match key.cast::<PyTuple>() {
        Ok(tuple) => tuple.iter().map(|item| parse_index_item(&item)).collect(),
        Err(_) => Ok(vec![parse_index_item(key)?]),
    }
}

fn to_nested<'py>(py: Python<'py>, data: &[f32], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    match shape.split_first() {
        None => Ok(PyFloat::new(py, data[0] as f64).into_any()),
        // Innermost lists come straight from the floats, without first
        // collecting a Vec of Python objects.
        Some((_, [])) => Ok(PyList::new(py, data.iter().map(|&x| x as f64))?.into_any()),
        Some((&len, rest)) => {
            let chunk: usize = rest.iter().product();
            let items = (0..len)
                .map(|i| to_nested(py, &data[i * chunk..(i + 1) * chunk], rest))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

/// The other side of an arithmetic op: a tensor (shared, not copied), a number,
/// or a nested list. `None` means "not something we combine with", so binary
/// operators can return `NotImplemented` and let Python try the other operand.
fn operand(obj: &Bound<'_, PyAny>) -> PyResult<Option<Tensor>> {
    if let Ok(t) = obj.cast::<PyTensor>() {
        return Ok(Some(t.get().inner.shallow_clone()));
    }
    if let Ok(x) = obj.extract::<f32>() {
        return Ok(Some(Tensor::scalar(x)));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return tensor_from_object(obj).map(Some);
    }
    Ok(None)
}

/// [`operand`] for places where anything else is a `TypeError`.
fn require_operand(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    operand(obj)?.ok_or_else(|| {
        let type_name = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        PyTypeError::new_err(format!(
            "expected a tensor, number or list, got {type_name}"
        ))
    })
}

/// Runs a binary op if `other` is usable, otherwise returns `NotImplemented`.
fn binary_op<'py>(
    py: Python<'py>,
    other: &Bound<'py, PyAny>,
    op: impl FnOnce(&Tensor) -> crate::Result<Tensor>,
) -> PyResult<Bound<'py, PyAny>> {
    match operand(other)? {
        Some(other) => Ok(Bound::new(py, PyTensor::from(op(&other)?))?.into_any()),
        None => Ok(py.NotImplemented().into_bound(py)),
    }
}

impl PyTensor {
    /// Indexes and converts the result: a plain float when every axis got an
    /// integer (like NumPy), otherwise a tensor view.
    fn index_to_py<'py>(
        &self,
        py: Python<'py>,
        items: &[TensorIndex],
    ) -> PyResult<Bound<'py, PyAny>> {
        let view = self.inner.index(items)?;
        let all_ints = items.iter().all(|i| matches!(i, TensorIndex::Index(_)));
        if all_ints && view.ndim() == 0 {
            return Ok(PyFloat::new(py, view.item()? as f64).into_any());
        }
        Ok(Bound::new(py, PyTensor::from(view))?.into_any())
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn py_new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(tensor_from_object(data)?.into())
    }

    // ---- metadata ----

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    /// Strides in elements, like PyTorch's `Tensor.stride()`.
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.strides())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().name()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Where this view starts in its storage, in elements.
    fn storage_offset(&self) -> usize {
        self.inner.offset()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// True when both tensors read and write the same memory.
    fn shares_memory(&self, other: &PyTensor) -> bool {
        self.inner.shares_storage(&other.inner)
    }

    // ---- copies ----

    /// A copy with its own memory (PyTorch spelling).
    fn clone(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

    /// A copy with its own memory (NumPy spelling).
    fn copy(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

    /// This tensor if it's already contiguous, otherwise a contiguous copy.
    fn contiguous(&self) -> PyTensor {
        self.inner.contiguous().into()
    }

    // ---- views ----

    /// A view with a new shape; one dimension may be -1. Raises if the memory
    /// layout doesn't allow it (use `reshape`, which copies when needed).
    #[pyo3(signature = (*shape))]
    fn view(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        Ok(self.inner.view(&parse_dims(shape)?)?.into())
    }

    /// A tensor with a new shape: a view when possible, otherwise a copy.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        Ok(self.inner.reshape(&parse_dims(shape)?)?.into())
    }

    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<PyTensor> {
        Ok(self.inner.flatten(start_dim, end_dim)?.into())
    }

    #[pyo3(signature = (*dims))]
    fn permute(&self, dims: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        Ok(self.inner.permute(&parse_dims(dims)?)?.into())
    }

    /// `transpose(d0, d1)` swaps two axes (PyTorch); `transpose()` reverses
    /// all of them (NumPy).
    #[pyo3(signature = (*dims))]
    fn transpose(&self, dims: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        match parse_dims(dims)?.as_slice() {
            [] => Ok(self.inner.reversed_axes().into()),
            &[a, b] => Ok(self.inner.transpose(a, b)?.into()),
            _ => Err(PyTypeError::new_err(
                "transpose takes no axes (reverse all) or two axes to swap; \
                 use permute to reorder every axis",
            )),
        }
    }

    /// All axes reversed.
    #[getter(T)]
    fn reversed(&self) -> PyTensor {
        self.inner.reversed_axes().into()
    }

    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<isize>) -> PyResult<PyTensor> {
        Ok(self.inner.squeeze(dim)?.into())
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<PyTensor> {
        Ok(self.inner.unsqueeze(dim)?.into())
    }

    // ---- elements ----

    fn item(&self) -> PyResult<f32> {
        Ok(self.inner.item()?)
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_nested(py, &self.inner.to_vec(), self.inner.shape())
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    /// Iterates over the first axis: views for N-d tensors, floats for 1-d.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let items = (0..self.__len__()? as isize)
            .map(|i| self.index_to_py(py, &[TensorIndex::Index(i)]))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)?.try_iter()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.index_to_py(py, &parse_index(key)?)
    }

    /// Writes through the indexed view. `value` can be a number, a tensor or a
    /// nested list, and is broadcast to the view's shape.
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = self.inner.index(&parse_index(key)?)?;
        Ok(view.copy_from(&require_operand(value)?)?)
    }

    fn __float__(&self) -> PyResult<f32> {
        self.item()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    // ---- broadcasting ----

    /// A view stretched to `shape` (stride 0 along broadcast axes); -1 keeps a size.
    #[pyo3(signature = (*shape))]
    fn expand(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        Ok(self.inner.expand(&parse_dims(shape)?)?.into())
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        Ok(self.inner.broadcast_to(&shape)?.into())
    }

    // ---- arithmetic operators ----

    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| self.inner.add(o))
    }

    fn __radd__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| o.add(&self.inner))
    }

    fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| self.inner.sub(o))
    }

    fn __rsub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| o.sub(&self.inner))
    }

    fn __mul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| self.inner.mul(o))
    }

    fn __rmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| o.mul(&self.inner))
    }

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| self.inner.div(o))
    }

    fn __rtruediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| o.div(&self.inner))
    }

    fn __pow__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
        _modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| self.inner.pow(o))
    }

    fn __rpow__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
        _modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        binary_op(py, other, |o| o.pow(&self.inner))
    }

    // In-place operators write through to the shared storage, so every view
    // of this tensor sees the change (like NumPy and PyTorch).

    fn __iadd__(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.inner.add_(&require_operand(other)?)?)
    }

    fn __isub__(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.inner.sub_(&require_operand(other)?)?)
    }

    fn __imul__(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.inner.mul_(&require_operand(other)?)?)
    }

    fn __itruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(self.inner.div_(&require_operand(other)?)?)
    }

    fn __ipow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Ok(self.inner.pow_(&require_operand(other)?)?)
    }

    fn __neg__(&self) -> PyTensor {
        self.inner.neg().into()
    }

    fn __pos__(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

    fn __abs__(&self) -> PyTensor {
        self.inner.abs().into()
    }

    // ---- named math (also available as module functions) ----

    fn add(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.add(&require_operand(other)?)?.into())
    }

    fn sub(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.sub(&require_operand(other)?)?.into())
    }

    fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.mul(&require_operand(other)?)?.into())
    }

    fn div(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.div(&require_operand(other)?)?.into())
    }

    fn pow(&self, exponent: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.pow(&require_operand(exponent)?)?.into())
    }

    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.maximum(&require_operand(other)?)?.into())
    }

    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.minimum(&require_operand(other)?)?.into())
    }

    fn abs(&self) -> PyTensor {
        self.inner.abs().into()
    }

    fn neg(&self) -> PyTensor {
        self.inner.neg().into()
    }

    fn exp(&self) -> PyTensor {
        self.inner.exp().into()
    }

    fn log(&self) -> PyTensor {
        self.inner.log().into()
    }

    fn sqrt(&self) -> PyTensor {
        self.inner.sqrt().into()
    }

    fn sin(&self) -> PyTensor {
        self.inner.sin().into()
    }

    fn cos(&self) -> PyTensor {
        self.inner.cos().into()
    }

    fn tanh(&self) -> PyTensor {
        self.inner.tanh().into()
    }

    fn sigmoid(&self) -> PyTensor {
        self.inner.sigmoid().into()
    }

    fn relu(&self) -> PyTensor {
        self.inner.relu().into()
    }

    #[pyo3(signature = (min=None, max=None))]
    fn clamp(&self, min: Option<f32>, max: Option<f32>) -> PyTensor {
        self.inner.clamp(min, max).into()
    }

    #[pyo3(signature = (other, rtol=1e-5, atol=1e-8))]
    fn allclose(&self, other: &Bound<'_, PyAny>, rtol: f32, atol: f32) -> PyResult<bool> {
        Ok(self.inner.allclose(&require_operand(other)?, rtol, atol)?)
    }
}

/// Builds a tensor from a number, a (nested) list/tuple, or another tensor.
#[pyfunction]
fn tensor(data: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    Ok(tensor_from_object(data)?.into())
}

#[pyfunction]
#[pyo3(signature = (*shape))]
fn zeros(shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
    Ok(Tensor::zeros(&parse_shape(shape)?).into())
}

#[pyfunction]
#[pyo3(signature = (*shape))]
fn ones(shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
    Ok(Tensor::ones(&parse_shape(shape)?).into())
}

#[pyfunction]
fn full(shape: &Bound<'_, PyAny>, fill_value: f32) -> PyResult<PyTensor> {
    let shape: Vec<usize> = match shape.extract::<usize>() {
        Ok(n) => vec![n],
        Err(_) => shape.extract()?,
    };
    Ok(Tensor::full(&shape, fill_value).into())
}

/// `arange(stop)`, `arange(start, stop)` or `arange(start, stop, step)`.
#[pyfunction]
#[pyo3(signature = (start, end=None, step=1.0))]
fn arange(start: f32, end: Option<f32>, step: f32) -> PyResult<PyTensor> {
    let (start, end) = match end {
        Some(end) => (start, end),
        None => (0.0, start),
    };
    Ok(Tensor::arange(start, end, step)?.into())
}

// ---- math functions: `ts.exp(x)` etc. accept tensors, numbers and lists ----

macro_rules! unary_functions {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            fn $name(x: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
                Ok(require_operand(x)?.$name().into())
            }
        )*
    };
}

unary_functions!(abs, exp, log, sqrt, sin, cos, tanh, sigmoid, relu);

#[pyfunction]
fn maximum(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    Ok(require_operand(a)?.maximum(&require_operand(b)?)?.into())
}

#[pyfunction]
fn minimum(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    Ok(require_operand(a)?.minimum(&require_operand(b)?)?.into())
}

#[pyfunction]
#[pyo3(signature = (x, min=None, max=None))]
fn clamp(x: &Bound<'_, PyAny>, min: Option<f32>, max: Option<f32>) -> PyResult<PyTensor> {
    Ok(require_operand(x)?.clamp(min, max).into())
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8))]
fn allclose(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, rtol: f32, atol: f32) -> PyResult<bool> {
    Ok(require_operand(a)?.allclose(&require_operand(b)?, rtol, atol)?)
}

#[pyfunction]
fn broadcast_to(x: &Bound<'_, PyAny>, shape: Vec<usize>) -> PyResult<PyTensor> {
    Ok(require_operand(x)?.broadcast_to(&shape)?.into())
}

/// The shape that the given shapes broadcast to.
#[pyfunction]
#[pyo3(signature = (*shapes))]
fn broadcast_shapes<'py>(
    py: Python<'py>,
    shapes: Vec<Vec<usize>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut out = Vec::new();
    for shape in &shapes {
        out = crate::ops::broadcast_shapes(&out, shape)?;
    }
    PyTuple::new(py, out)
}

#[pymodule]
fn tensorsc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    // Lets benchmarks warn when they're timing an unoptimized build.
    m.add("_debug_build", cfg!(debug_assertions))?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_shapes, m)?)?;
    Ok(())
}

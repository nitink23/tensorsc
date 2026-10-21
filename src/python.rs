//! Python bindings. Everything here converts between Python objects and the
//! core types; tensor logic itself belongs in the core modules.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
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
    if args.len() == 1
        && let Ok(dims) = args.get_item(0)?.extract::<Vec<isize>>()
    {
        return Ok(dims);
    }
    args.extract()
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
        let step: Option<isize> = slice.getattr("step")?.extract()?;
        return Ok(TensorIndex::Slice {
            start: slice.getattr("start")?.extract()?,
            stop: slice.getattr("stop")?.extract()?,
            step: step.unwrap_or(1),
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
        Some((&len, rest)) => {
            let chunk: usize = rest.iter().product();
            let items = (0..len)
                .map(|i| to_nested(py, &data[i * chunk..(i + 1) * chunk], rest))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
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

    /// Writes through the indexed view. `value` can be a number (fills the
    /// view), a tensor, or a nested list of the view's shape.
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = self.inner.index(&parse_index(key)?)?;
        if let Ok(number) = value.extract::<f32>() {
            view.fill(number);
        } else if let Ok(src) = value.cast::<PyTensor>() {
            view.copy_from(&src.get().inner)?;
        } else {
            view.copy_from(&tensor_from_object(value)?)?;
        }
        Ok(())
    }

    fn __float__(&self) -> PyResult<f32> {
        self.item()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
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

#[pymodule]
fn tensorsc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    Ok(())
}

//! Python bindings. Everything here converts between Python objects and the
//! core types; tensor logic itself belongs in the core modules.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PyTuple};

use crate::error::TensorError;
use crate::tensor::Tensor;

impl From<TensorError> for PyErr {
    fn from(err: TensorError) -> PyErr {
        match err {
            TensorError::IndexRankMismatch { .. } | TensorError::IndexOutOfBounds { .. } => {
                PyIndexError::new_err(err.to_string())
            }
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
fn parse_shape(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    if args.len() == 1
        && let Ok(shape) = args.get_item(0)?.extract::<Vec<usize>>()
    {
        return Ok(shape);
    }
    args.extract()
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
    /// Turns an int or tuple of ints (negative allowed) into a full index.
    fn parse_index(&self, key: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let raw: Vec<isize> = match key.extract::<isize>() {
            Ok(i) => vec![i],
            Err(_) => key.extract().map_err(|_| {
                PyTypeError::new_err("tensor indices must be integers or tuples of integers")
            })?,
        };
        let shape = self.inner.shape();
        if raw.len() != shape.len() {
            let mut msg = TensorError::IndexRankMismatch {
                expected: shape.len(),
                got: raw.len(),
            }
            .to_string();
            if raw.len() < shape.len() {
                msg.push_str(" (selecting whole rows/slices is coming with views)");
            }
            return Err(PyIndexError::new_err(msg));
        }
        raw.iter()
            .zip(shape)
            .enumerate()
            .map(|(axis, (&i, &size))| {
                let resolved = if i < 0 { i + size as isize } else { i };
                if resolved < 0 || resolved as usize >= size {
                    Err(PyIndexError::new_err(format!(
                        "index {i} is out of bounds for axis {axis} with size {size}"
                    )))
                } else {
                    Ok(resolved as usize)
                }
            })
            .collect()
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn py_new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(tensor_from_object(data)?.into())
    }

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

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// True when both tensors read and write the same memory.
    fn shares_memory(&self, other: &PyTensor) -> bool {
        self.inner.shares_storage(&other.inner)
    }

    /// A copy with its own memory (PyTorch spelling).
    fn clone(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

    /// A copy with its own memory (NumPy spelling).
    fn copy(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

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

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<f32> {
        Ok(self.inner.get(&self.parse_index(key)?)?)
    }

    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: f32) -> PyResult<()> {
        Ok(self.inner.set(&self.parse_index(key)?, value)?)
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

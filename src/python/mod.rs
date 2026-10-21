//! Python bindings. Everything here converts between Python objects and the
//! core types; tensor logic itself belongs in the core modules.

mod convert;
mod pytensor;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::dtype::{DType, Scalar};
use crate::python::convert::parse_reduce_axis;
use crate::python::convert::{
    PyDType, operands, parse_optional_dtype, parse_shape, parse_shape_arg, require_operand,
    scalar_from_py, tensor_from_object, type_name,
};
use crate::python::pytensor::{PyTensor, clamp_impl, reduce_with};
use crate::tensor::Tensor;

// ---- creation ----

/// Builds a new tensor (a copy) from a number, a nested list/tuple, or
/// another tensor. Without `dtype`: bools stay bool, ints become int64,
/// floats float32.
#[pyfunction]
#[pyo3(signature = (data, dtype=None))]
fn tensor(data: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    Ok(tensor_from_object(data, parse_optional_dtype(dtype)?)?.into())
}

#[pyfunction]
#[pyo3(signature = (*shape, dtype=None))]
fn zeros(shape: &Bound<'_, PyTuple>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    let dtype = parse_optional_dtype(dtype)?.unwrap_or(DType::DEFAULT_FLOAT);
    Ok(Tensor::zeros(&parse_shape(shape)?, dtype).into())
}

#[pyfunction]
#[pyo3(signature = (*shape, dtype=None))]
fn ones(shape: &Bound<'_, PyTuple>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    let dtype = parse_optional_dtype(dtype)?.unwrap_or(DType::DEFAULT_FLOAT);
    Ok(Tensor::ones(&parse_shape(shape)?, dtype).into())
}

fn number(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Scalar> {
    scalar_from_py(obj)?.ok_or_else(|| {
        PyTypeError::new_err(format!("{what} must be a number, got {}", type_name(obj)))
    })
}

/// Without `dtype`, the dtype follows `fill_value` (as in PyTorch and NumPy).
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
fn full(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let value = number(fill_value, "fill_value")?;
    let dtype = parse_optional_dtype(dtype)?.unwrap_or(value.default_dtype());
    Ok(Tensor::full(&parse_shape_arg(shape)?, value, dtype).into())
}

/// `arange(stop)`, `arange(start, stop)` or `arange(start, stop, step)`.
/// Without `dtype`: int64 if every argument is an int, else float32.
#[pyfunction]
#[pyo3(signature = (start, end=None, step=None, dtype=None))]
fn arange(
    start: &Bound<'_, PyAny>,
    end: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let first = number(start, "start")?;
    let (start, end) = match end {
        Some(end) => (first, number(end, "end")?),
        None => (Scalar::Int(0), first),
    };
    let step = step
        .map(|s| number(s, "step"))
        .transpose()?
        .unwrap_or(Scalar::Int(1));
    let all_ints = [start, end, step]
        .iter()
        .all(|s| !matches!(s, Scalar::Float(_)));
    let inferred = if all_ints {
        DType::I64
    } else {
        DType::DEFAULT_FLOAT
    };
    let dtype = parse_optional_dtype(dtype)?.unwrap_or(inferred);
    Ok(Tensor::arange(start, end, step, dtype)?.into())
}

// ---- math functions: `ts.exp(x)` etc. accept tensors, numbers and lists ----

macro_rules! unary_functions {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            fn $name(x: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
                Ok(require_operand(x, DType::DEFAULT_FLOAT)?.$name().into())
            }
        )*
    };
}

unary_functions!(
    abs,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tanh,
    sigmoid,
    floor,
    ceil,
    round,
    trunc,
    isnan,
    isinf,
    isfinite,
    logical_not,
);

#[pyfunction]
fn relu(x: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    Ok(require_operand(x, DType::DEFAULT_FLOAT)?.relu()?.into())
}

macro_rules! binary_functions {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            fn $name(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
                let [a, b] = operands([a, b])?;
                Ok(a.$name(&b)?.into())
            }
        )*
    };
}

binary_functions!(maximum, minimum, logical_and, logical_or, logical_xor);

/// `cond ? a : b` elementwise. Also spelled `where` in Python.
#[pyfunction]
#[pyo3(name = "where")]
fn where_(
    cond: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let cond = require_operand(cond, DType::Bool)?;
    let [a, b] = operands([a, b])?;
    Ok(crate::ops::where_(&cond, &a, &b)?.into())
}

#[pyfunction]
#[pyo3(signature = (x, min=None, max=None))]
fn clamp(
    x: &Bound<'_, PyAny>,
    min: Option<&Bound<'_, PyAny>>,
    max: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let x = require_operand(x, DType::DEFAULT_FLOAT)?;
    Ok(clamp_impl(&x, min, max)?.into())
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8))]
fn allclose(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, rtol: f64, atol: f64) -> PyResult<bool> {
    let [a, b] = operands([a, b])?;
    Ok(a.allclose(&b, rtol, atol)?)
}

#[pyfunction]
fn broadcast_to(x: &Bound<'_, PyAny>, shape: Vec<usize>) -> PyResult<PyTensor> {
    Ok(require_operand(x, DType::DEFAULT_FLOAT)?
        .broadcast_to(&shape)?
        .into())
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

// ---- reductions and linear algebra ----

macro_rules! reduce_functions {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            #[pyo3(signature = (x, dim=None, keepdim=false, *, axis=None, keepdims=None))]
            fn $name(
                x: &Bound<'_, PyAny>,
                dim: Option<&Bound<'_, PyAny>>,
                keepdim: bool,
                axis: Option<&Bound<'_, PyAny>>,
                keepdims: Option<bool>,
            ) -> PyResult<PyTensor> {
                let x = require_operand(x, DType::DEFAULT_FLOAT)?;
                reduce_with(&x, Tensor::$name, dim, keepdim, axis, keepdims)
            }
        )*
    };
}

reduce_functions!(sum, prod, mean, max, min, all, any);

macro_rules! arg_functions {
    ($($name:ident),* $(,)?) => {
        $(
            #[pyfunction]
            #[pyo3(signature = (x, dim=None, keepdim=false, *, axis=None, keepdims=None))]
            fn $name(
                x: &Bound<'_, PyAny>,
                dim: Option<&Bound<'_, PyAny>>,
                keepdim: bool,
                axis: Option<&Bound<'_, PyAny>>,
                keepdims: Option<bool>,
            ) -> PyResult<PyTensor> {
                let x = require_operand(x, DType::DEFAULT_FLOAT)?;
                let axis = parse_reduce_axis(dim, axis)?;
                Ok(x.$name(axis, keepdims.unwrap_or(keepdim))?.into())
            }
        )*
    };
}

arg_functions!(argmax, argmin);

#[pyfunction]
fn matmul(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let [a, b] = operands([a, b])?;
    Ok(a.matmul(&b)?.into())
}

macro_rules! add_functions {
    ($m:expr, $($f:ident),* $(,)?) => {
        $($m.add_function(wrap_pyfunction!($f, $m)?)?;)*
    };
}

#[pymodule]
fn tensorsc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyDType>()?;
    for dtype in DType::ALL {
        m.add(dtype.name(), PyDType { dtype })?;
    }
    // Lets benchmarks warn when they're timing an unoptimized build.
    m.add("_debug_build", cfg!(debug_assertions))?;
    add_functions!(
        m,
        tensor,
        zeros,
        ones,
        full,
        arange,
        abs,
        exp,
        log,
        sqrt,
        sin,
        cos,
        tanh,
        sigmoid,
        relu,
        floor,
        ceil,
        round,
        trunc,
        isnan,
        isinf,
        isfinite,
        logical_not,
        maximum,
        minimum,
        logical_and,
        logical_or,
        logical_xor,
        where_,
        clamp,
        allclose,
        broadcast_to,
        broadcast_shapes,
        sum,
        prod,
        mean,
        max,
        min,
        all,
        any,
        argmax,
        argmin,
        matmul,
    );
    Ok(())
}

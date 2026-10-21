//! Converting between Python objects and the core types.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyEllipsis, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple, PyType};

use crate::dtype::{DType, Element, Scalar, with_dtype};
use crate::error::TensorError;
use crate::python::buffer::{has_buffer, tensor_from_buffer};
use crate::python::pytensor::PyTensor;
use crate::tensor::Tensor;
use crate::views::TensorIndex;

impl From<TensorError> for PyErr {
    fn from(err: TensorError) -> PyErr {
        let msg = err.to_string();
        match err {
            TensorError::IndexRankMismatch { .. }
            | TensorError::TooManyIndices { .. }
            | TensorError::IndexOutOfBounds { .. }
            | TensorError::AxisOutOfRange { .. }
            | TensorError::InvalidIndex(_) => PyIndexError::new_err(msg),
            TensorError::DivisionByZero => PyZeroDivisionError::new_err(msg),
            TensorError::UnsupportedDType { .. } => PyTypeError::new_err(msg),
            _ => PyValueError::new_err(msg),
        }
    }
}

pub(crate) fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into())
}

// ---- dtypes ----

/// A dtype as a Python object: `tensorsc.float32` etc.
#[pyclass(name = "dtype", module = "tensorsc", frozen)]
pub(crate) struct PyDType {
    pub(crate) dtype: DType,
}

#[pymethods]
impl PyDType {
    #[getter]
    fn name(&self) -> &'static str {
        self.dtype.name()
    }

    /// Size of one element in bytes.
    #[getter]
    fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    #[getter]
    fn is_floating_point(&self) -> bool {
        self.dtype.is_float()
    }

    /// Equal to the same dtype however it's spelled: `tensorsc.float32 ==
    /// "float32"`, and NumPy dtypes too.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        parse_dtype(other).is_ok_and(|d| d == self.dtype)
    }

    fn __hash__(&self) -> u64 {
        self.dtype as u64
    }

    fn __repr__(&self) -> String {
        format!("tensorsc.{}", self.dtype)
    }

    fn __str__(&self) -> &'static str {
        self.dtype.name()
    }
}

pub(crate) fn dtype_to_py(py: Python<'_>, dtype: DType) -> PyResult<Bound<'_, PyDType>> {
    Bound::new(py, PyDType { dtype })
}

/// Reads a dtype from a `tensorsc` dtype, its name (`"int64"`), a Python type
/// (`bool`, `int` and `float` mean bool, int64 and float64, as in NumPy), or
/// a NumPy dtype or scalar type.
pub(crate) fn parse_dtype(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    let unknown = || {
        let names: Vec<&str> = DType::ALL.iter().map(|d| d.name()).collect();
        PyTypeError::new_err(format!(
            "unsupported dtype {}; expected one of {}",
            obj.repr().map(|r| r.to_string()).unwrap_or_default(),
            names.join(", ")
        ))
    };
    if let Ok(d) = obj.cast::<PyDType>() {
        return Ok(d.get().dtype);
    }
    let from_name = |name: &str| match name {
        // NumPy 1.x spells its bool type `bool_`.
        "bool_" => Some(DType::Bool),
        name => DType::from_name(name),
    };
    if let Ok(name) = obj.cast::<PyString>() {
        return from_name(name.to_str()?).ok_or_else(unknown);
    }
    let py = obj.py();
    if let Ok(ty) = obj.cast::<PyType>() {
        if ty.is(py.get_type::<PyBool>()) {
            return Ok(DType::Bool);
        }
        if ty.is(py.get_type::<PyInt>()) {
            return Ok(DType::I64);
        }
        if ty.is(py.get_type::<PyFloat>()) {
            return Ok(DType::F64);
        }
    }
    // NumPy: `np.dtype("float32").name` or `np.float32.__name__`.
    for attr in ["name", "__name__"] {
        if let Ok(name) = obj.getattr(attr)
            && let Ok(name) = name.extract::<PyBackedStr>()
            && let Some(dtype) = from_name(&name)
        {
            return Ok(dtype);
        }
    }
    Err(unknown())
}

pub(crate) fn parse_optional_dtype(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<DType>> {
    obj.filter(|o| !o.is_none()).map(parse_dtype).transpose()
}

// ---- scalars ----

/// A Python `bool`, `int` or `float` (checked in that order, since `bool` is
/// a subclass of `int`). `None` for anything else.
pub(crate) fn scalar_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Option<Scalar>> {
    if let Ok(float) = obj.cast_exact::<PyFloat>() {
        // An exact float can't fail to convert, so read it directly.
        return Ok(Some(Scalar::Float(float.value())));
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Some(Scalar::Bool(obj.extract()?)));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Some(Scalar::Int(obj.extract()?)));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Some(Scalar::Float(obj.extract()?)));
    }
    Ok(None)
}

pub(crate) fn scalar_to_py(py: Python<'_>, value: Scalar) -> PyResult<Bound<'_, PyAny>> {
    Ok(match value {
        Scalar::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
        Scalar::Int(i) => i.into_pyobject(py)?.into_any(),
        Scalar::Float(x) => PyFloat::new(py, x).into_any(),
    })
}

/// The dtype a Python number takes next to a tensor of `other`, following
/// PyTorch: it adopts the tensor's dtype unless it's of a higher kind (a
/// float next to ints), in which case it gets the default for its kind. So
/// `uint8 + 1` stays uint8, `float64 + 0.1` keeps full precision, and
/// `int * 2.5` is float32.
pub(crate) fn weak_scalar_dtype(value: Scalar, other: DType) -> DType {
    if value.default_dtype().kind() > other.kind() {
        value.default_dtype()
    } else {
        other
    }
}

// ---- nested lists ----

/// Numbers gathered from a nested list, upgraded as needed: all bools stay
/// bools, ints and bools become ints, any float makes everything floats.
enum Gathered {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Float(Vec<f64>),
}

impl Gathered {
    fn len(&self) -> usize {
        match self {
            Gathered::Bool(v) => v.len(),
            Gathered::Int(v) => v.len(),
            Gathered::Float(v) => v.len(),
        }
    }

    fn reserve(&mut self, additional: usize) {
        match self {
            Gathered::Bool(v) => v.reserve(additional),
            Gathered::Int(v) => v.reserve(additional),
            Gathered::Float(v) => v.reserve(additional),
        }
    }

    fn push(&mut self, value: Scalar) {
        match (&mut *self, value) {
            (Gathered::Bool(v), Scalar::Bool(b)) => v.push(b),
            (Gathered::Int(v), Scalar::Int(i)) => v.push(i),
            (Gathered::Int(v), Scalar::Bool(b)) => v.push(b as i64),
            (Gathered::Float(v), value) => v.push(value.to_f64()),
            (Gathered::Bool(v), Scalar::Int(i)) => {
                let mut ints: Vec<i64> = v.iter().map(|&b| b as i64).collect();
                ints.push(i);
                *self = Gathered::Int(ints);
            }
            (Gathered::Bool(v), Scalar::Float(x)) => {
                let mut floats: Vec<f64> = v.iter().map(|&b| b as u8 as f64).collect();
                floats.push(x);
                *self = Gathered::Float(floats);
            }
            (Gathered::Int(v), Scalar::Float(x)) => {
                let mut floats: Vec<f64> = v.iter().map(|&i| i as f64).collect();
                floats.push(x);
                *self = Gathered::Float(floats);
            }
        }
    }

    fn default_dtype(&self) -> DType {
        match self {
            Gathered::Bool(_) => DType::Bool,
            Gathered::Int(_) => DType::I64,
            Gathered::Float(_) => DType::DEFAULT_FLOAT,
        }
    }

    fn into_tensor(self, shape: &[usize], dtype: DType) -> crate::Result<Tensor> {
        with_dtype!(dtype, T => {
            let data: Vec<T> = match self {
                Gathered::Bool(v) => v.into_iter().map(T::from_bool).collect(),
                Gathered::Int(v) => v.into_iter().map(T::from_i64).collect(),
                Gathered::Float(v) => v.into_iter().map(T::from_f64).collect(),
            };
            Tensor::from_vec(data, shape)
        })
    }
}

/// Walks a nested list/tuple, recording its shape and gathering its numbers.
/// Every list at the same depth must have the same length.
fn flatten_nested(
    obj: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    data: &mut Gathered,
) -> PyResult<()> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let len = obj.len()?;
        if depth == shape.len() {
            // First list seen at this depth sets the axis length, unless numbers
            // already appeared at a shallower depth.
            if data.len() > 0 {
                return Err(ragged());
            }
            shape.push(len);
        } else if shape[depth] != len {
            return Err(ragged());
        }
        for item in obj.try_iter()? {
            flatten_nested(&item?, depth + 1, shape, data)?;
        }
        return Ok(());
    }
    if depth != shape.len() {
        return Err(ragged());
    }
    if data.len() == 0 {
        // The first number comes after one list at every depth, so the whole
        // shape is known: allocate once instead of growing.
        data.reserve(shape.iter().product());
    }
    let value = scalar_from_py(obj)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "tensor elements must be numbers, got {}",
            type_name(obj)
        ))
    })?;
    data.push(value);
    Ok(())
}

fn ragged() -> PyErr {
    PyValueError::new_err("nested sequences must be rectangular (every row the same length)")
}

/// Builds a new tensor (always a copy) from a tensor, a NumPy array (or
/// other buffer), a number or a nested list. Without `dtype`, it's inferred:
/// buffers keep theirs, bools stay bool, ints become int64, floats float32.
pub(crate) fn tensor_from_object(obj: &Bound<'_, PyAny>, dtype: Option<DType>) -> PyResult<Tensor> {
    let copy = |t: &Tensor| match dtype {
        Some(d) if d != t.dtype() => t.to_dtype(d),
        _ => t.deep_copy(),
    };
    if let Ok(other) = obj.cast::<PyTensor>() {
        return Ok(copy(&other.get().inner));
    }
    if has_buffer(obj) {
        return Ok(copy(&tensor_from_buffer(obj)?));
    }
    let mut shape = Vec::new();
    let mut data = Gathered::Bool(Vec::new());
    flatten_nested(obj, 0, &mut shape, &mut data)?;
    // An empty list holds no values to infer from; it's float, as in PyTorch.
    let inferred = if data.len() == 0 {
        DType::DEFAULT_FLOAT
    } else {
        data.default_dtype()
    };
    Ok(data.into_tensor(&shape, dtype.unwrap_or(inferred))?)
}

// ---- operands ----

/// A 0-d tensor for a Python number used next to a tensor of `other`.
pub(crate) fn weak_scalar(value: Scalar, other: DType) -> Tensor {
    Tensor::full(&[], value, weak_scalar_dtype(value, other))
}

/// A plain Python number, which adopts the dtype of the tensor it meets (see
/// [`weak_scalar_dtype`]). NumPy scalars aren't plain: they carry their own
/// dtype (as in NumPy 2), so they're treated like 0-d arrays instead.
pub(crate) fn weak_number(obj: &Bound<'_, PyAny>) -> PyResult<Option<Scalar>> {
    if has_buffer(obj) {
        return Ok(None);
    }
    scalar_from_py(obj)
}

/// The other side of an operation with a tensor of dtype `other`: a tensor or
/// NumPy array (shared, not copied), a number (see [`weak_scalar_dtype`]), or
/// a nested list. `None` means "not something we combine with", so binary
/// operators can return `NotImplemented` and let Python try the other operand.
pub(crate) fn operand(obj: &Bound<'_, PyAny>, other: DType) -> PyResult<Option<Tensor>> {
    if let Ok(t) = obj.cast::<PyTensor>() {
        return Ok(Some(t.get().inner.shallow_clone()));
    }
    if has_buffer(obj) {
        return tensor_from_buffer(obj).map(Some);
    }
    if let Some(value) = scalar_from_py(obj)? {
        return Ok(Some(weak_scalar(value, other)));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return tensor_from_object(obj, None).map(Some);
    }
    Ok(None)
}

/// [`operand`] for places where anything else is a `TypeError`.
pub(crate) fn require_operand(obj: &Bound<'_, PyAny>, other: DType) -> PyResult<Tensor> {
    operand(obj, other)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "expected a tensor, number or list, got {}",
            type_name(obj)
        ))
    })
}

/// Operands for functions of several inputs (`ts.maximum(a, b)`): numbers
/// take their dtype from the first input that isn't a plain number.
pub(crate) fn operands<const N: usize>(objs: [&Bound<'_, PyAny>; N]) -> PyResult<[Tensor; N]> {
    let mut anchor = None;
    for obj in objs {
        if weak_number(obj)?.is_none() {
            anchor = Some(require_operand(obj, DType::DEFAULT_FLOAT)?.dtype());
            break;
        }
    }
    let mut out = Vec::with_capacity(N);
    for obj in objs {
        out.push(match (anchor, weak_number(obj)?) {
            (Some(dtype), Some(value)) => weak_scalar(value, dtype),
            (None, Some(value)) => Tensor::full(&[], value, value.default_dtype()),
            (_, None) => require_operand(obj, DType::DEFAULT_FLOAT)?,
        });
    }
    Ok(out
        .try_into()
        .unwrap_or_else(|_| unreachable!("N operands pushed")))
}

// ---- shapes ----

/// Accepts both `f(2, 3)` and `f((2, 3))`, like PyTorch.
pub(crate) fn parse_dims(args: &Bound<'_, PyTuple>) -> PyResult<Vec<isize>> {
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
pub(crate) fn parse_shape(args: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    to_sizes(parse_dims(args)?)
}

/// An int or a sequence of ints, as a shape.
pub(crate) fn parse_shape_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dims = match obj.extract::<isize>() {
        Ok(n) => vec![n],
        Err(_) => obj.extract()?,
    };
    to_sizes(dims)
}

fn to_sizes(dims: Vec<isize>) -> PyResult<Vec<usize>> {
    dims.into_iter()
        .map(|d| {
            usize::try_from(d)
                .map_err(|_| PyValueError::new_err(format!("negative dimension {d} in shape")))
        })
        .collect()
}

// ---- indexing ----

/// What a `t[key]` key asks for.
pub(crate) enum Key {
    /// Ints, slices, `None` and `...`: a view.
    Basic(Vec<TensorIndex>),
    /// A bool tensor (or list, or a bare `True`/`False`): a copy.
    Mask(Tensor),
    /// An int tensor (or list) of positions along the first axis: a copy.
    Positions(Tensor),
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
    if obj.is_instance_of::<PyTensor>() || obj.is_instance_of::<PyList>() {
        return Err(PyIndexError::new_err(
            "index arrays and masks can't be combined with other indices yet; \
             index with the array alone (t[idx]) or use index_select",
        ));
    }
    obj.extract::<isize>().map(TensorIndex::Index).map_err(|_| {
        PyIndexError::new_err(format!(
            "tensor indices must be integers, slices, None, ..., or bool/int tensors, not {}",
            type_name(obj)
        ))
    })
}

pub(crate) fn parse_key(key: &Bound<'_, PyAny>) -> PyResult<Key> {
    let array = if let Ok(t) = key.cast::<PyTensor>() {
        Some(t.get().inner.shallow_clone())
    } else if key.is_instance_of::<PyList>() || key.is_instance_of::<PyBool>() {
        Some(tensor_from_object(key, None)?)
    } else if has_buffer(key) && key.extract::<isize>().is_err() {
        // NumPy masks and index arrays. NumPy integer scalars (which convert
        // to an index) stay basic indexing, so `t[np.int64(1)]` is a view.
        Some(tensor_from_buffer(key)?)
    } else {
        None
    };
    if let Some(array) = array {
        return match array.dtype() {
            DType::Bool => Ok(Key::Mask(array)),
            d if d.is_float() => Err(PyIndexError::new_err(
                "only integer and bool tensors can be used as indices",
            )),
            _ => Ok(Key::Positions(array)),
        };
    }
    match key.cast::<PyTuple>() {
        Ok(tuple) => Ok(Key::Basic(
            tuple
                .iter()
                .map(|item| parse_index_item(&item))
                .collect::<PyResult<_>>()?,
        )),
        Err(_) => Ok(Key::Basic(vec![parse_index_item(key)?])),
    }
}

// ---- nested lists out ----

/// How each element type becomes a Python object in `tolist`.
pub(crate) trait ToPython: Element {
    type Py: for<'py> IntoPyObject<'py>;
    fn to_python(self) -> Self::Py;
}

macro_rules! to_python {
    ($($t:ty => $py:ty),*) => {
        $(impl ToPython for $t {
            type Py = $py;
            fn to_python(self) -> $py {
                self as $py
            }
        })*
    };
}

to_python!(bool => bool, u8 => i64, i32 => i64, i64 => i64, f32 => f64, f64 => f64);

pub(crate) fn to_nested<'py, T: ToPython>(
    py: Python<'py>,
    data: &[T],
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    match shape.split_first() {
        None => data[0].to_python().into_bound_py_any(py),
        // Innermost lists come straight from the values, without first
        // collecting a Vec of Python objects.
        Some((_, [])) => Ok(PyList::new(py, data.iter().map(|&x| x.to_python()))?.into_any()),
        Some((&len, rest)) => {
            let chunk: usize = rest.iter().product();
            let items = (0..len)
                .map(|i| to_nested(py, &data[i * chunk..(i + 1) * chunk], rest))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

// ---- reduction arguments ----

/// Axes for a reduction, from PyTorch's `dim` or NumPy's `axis` (not both):
/// `None` (everything), an int, or a sequence of ints.
pub(crate) fn parse_reduce_axes(
    dim: Option<&Bound<'_, PyAny>>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<isize>>> {
    let given = match (dim.filter(|d| !d.is_none()), axis.filter(|a| !a.is_none())) {
        (Some(_), Some(_)) => {
            return Err(PyTypeError::new_err("pass dim or axis, not both"));
        }
        (Some(d), None) | (None, Some(d)) => d,
        (None, None) => return Ok(None),
    };
    match given.extract::<isize>() {
        Ok(axis) => Ok(Some(vec![axis])),
        Err(_) => given.extract().map(Some),
    }
}

/// A single axis (for `argmax`/`argmin`), from `dim` or `axis`.
pub(crate) fn parse_reduce_axis(
    dim: Option<&Bound<'_, PyAny>>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<isize>> {
    match parse_reduce_axes(dim, axis)?.as_deref() {
        None => Ok(None),
        Some(&[axis]) => Ok(Some(axis)),
        Some(_) => Err(PyTypeError::new_err("expected a single axis")),
    }
}

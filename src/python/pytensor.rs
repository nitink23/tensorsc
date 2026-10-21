//! The Python `Tensor` class.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyIterator, PyList, PyTuple};

use crate::dtype::{DType, with_dtype};
use crate::python::convert::{
    Key, PyDType, dtype_to_py, operand, parse_dims, parse_dtype, parse_key, parse_reduce_axes,
    parse_reduce_axis, require_operand, scalar_to_py, tensor_from_object, to_nested,
};
use crate::tensor::Tensor;
use crate::views::TensorIndex;

#[pyclass(name = "Tensor", module = "tensorsc", frozen)]
pub(crate) struct PyTensor {
    pub(crate) inner: Tensor,
}

impl From<Tensor> for PyTensor {
    fn from(inner: Tensor) -> Self {
        PyTensor { inner }
    }
}

type BinaryFn = fn(&Tensor, &Tensor) -> crate::Result<Tensor>;
type InPlaceFn = fn(&Tensor, &Tensor) -> crate::Result<()>;

pub(crate) fn wrap(py: Python<'_>, tensor: Tensor) -> PyResult<Bound<'_, PyAny>> {
    Ok(Bound::new(py, PyTensor::from(tensor))?.into_any())
}

impl PyTensor {
    /// `self op other` (or `other op self` when `reflected`), returning
    /// `NotImplemented` for operands we don't understand so Python can try
    /// the other side.
    fn binary<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
        op: BinaryFn,
        reflected: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(other) = operand(other, self.inner.dtype())? else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        let result = if reflected {
            op(&other, &self.inner)
        } else {
            op(&self.inner, &other)
        };
        wrap(py, result?)
    }

    /// In-place ops write through to the shared storage, so every view of this
    /// tensor sees the change (as in NumPy and PyTorch).
    fn in_place(&self, other: &Bound<'_, PyAny>, op: InPlaceFn) -> PyResult<()> {
        Ok(op(
            &self.inner,
            &require_operand(other, self.inner.dtype())?,
        )?)
    }

    /// A named method taking one more operand (`t.maximum(x)`).
    fn with(&self, other: &Bound<'_, PyAny>, op: BinaryFn) -> PyResult<PyTensor> {
        Ok(op(&self.inner, &require_operand(other, self.inner.dtype())?)?.into())
    }

    /// Basic indexing: a plain Python number when every axis got an integer
    /// (as in NumPy), otherwise a tensor view.
    fn index_to_py<'py>(
        &self,
        py: Python<'py>,
        items: &[TensorIndex],
    ) -> PyResult<Bound<'py, PyAny>> {
        let view = self.inner.index(items)?;
        let all_ints = items.iter().all(|i| matches!(i, TensorIndex::Index(_)));
        if all_ints && view.ndim() == 0 {
            return scalar_to_py(py, view.item()?);
        }
        wrap(py, view)
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, dtype=None))]
    fn py_new(data: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let dtype = dtype.map(parse_dtype).transpose()?;
        Ok(tensor_from_object(data, dtype)?.into())
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
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDType>> {
        dtype_to_py(py, self.inner.dtype())
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

    fn is_floating_point(&self) -> bool {
        self.inner.dtype().is_float()
    }

    /// True when both tensors read and write the same memory.
    fn shares_memory(&self, other: &PyTensor) -> bool {
        self.inner.shares_storage(&other.inner)
    }

    // ---- copies and conversion ----

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

    /// A copy converted to `dtype` (NumPy's `astype`: always copies).
    fn astype(&self, dtype: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let dtype = parse_dtype(dtype)?;
        Ok(if dtype == self.inner.dtype() {
            self.inner.deep_copy()
        } else {
            self.inner.to_dtype(dtype)
        }
        .into())
    }

    /// This tensor as `dtype` (PyTorch's `to`: shares memory if nothing changes).
    fn to(&self, dtype: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        Ok(self.inner.to_dtype(parse_dtype(dtype)?).into())
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

    /// A view stretched to `shape` (stride 0 along broadcast axes); -1 keeps a size.
    #[pyo3(signature = (*shape))]
    fn expand(&self, shape: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        Ok(self.inner.expand(&parse_dims(shape)?)?.into())
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        Ok(self.inner.broadcast_to(&shape)?.into())
    }

    // ---- elements ----

    /// The value of a one-element tensor as a Python bool, int or float.
    fn item<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        scalar_to_py(py, self.inner.item()?)
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = self.inner.shape();
        with_dtype!(self.inner.dtype(), T => to_nested(py, &self.inner.to_vec::<T>(), shape))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    /// Iterates over the first axis: views for N-d tensors, numbers for 1-d.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let items = (0..self.__len__()? as isize)
            .map(|i| self.index_to_py(py, &[TensorIndex::Index(i)]))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)?.try_iter()
    }

    /// Ints, slices, `None` and `...` give views; a bool mask or an array of
    /// positions gives a copy (as in NumPy).
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match parse_key(key)? {
            Key::Basic(items) => self.index_to_py(py, &items),
            Key::Mask(mask) => wrap(py, self.inner.index_mask(&mask)?),
            Key::Positions(positions) => wrap(py, self.inner.index_select(0, &positions)?),
        }
    }

    /// Writes through to this tensor. `value` (a number, tensor or nested
    /// list) is converted to this dtype and broadcast to the indexed shape.
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let values = require_operand(value, self.inner.dtype())?;
        match parse_key(key)? {
            Key::Basic(items) => self.inner.index(&items)?.copy_from(&values)?,
            Key::Mask(mask) => self.inner.index_mask_put(&mask, &values)?,
            Key::Positions(positions) => self.inner.index_put(0, &positions, &values)?,
        }
        Ok(())
    }

    fn index_select(&self, dim: isize, index: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let index = require_operand(index, DType::I64)?;
        Ok(self.inner.index_select(dim, &index)?.into())
    }

    /// `if t:` works for one-element tensors only, as in NumPy and PyTorch.
    fn __bool__(&self) -> PyResult<bool> {
        match self.inner.numel() {
            1 => Ok(self.inner.item()?.to_bool()),
            0 => Err(PyValueError::new_err(
                "the truth value of an empty tensor is ambiguous",
            )),
            _ => Err(PyValueError::new_err(
                "the truth value of a tensor with more than one element is ambiguous; \
                 use .any() or .all()",
            )),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        Ok(self.inner.item()?.to_f64())
    }

    fn __int__(&self) -> PyResult<i64> {
        Ok(self.inner.item()?.to_i64())
    }

    /// Lets one-element integer tensors be used as list indices.
    fn __index__(&self) -> PyResult<i64> {
        if self.inner.dtype().is_float() {
            return Err(PyTypeError::new_err(
                "only integer tensors of a single element can be used as an index",
            ));
        }
        self.__int__()
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    // ---- operators ----

    fn __add__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::add, false)
    }

    fn __radd__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::add, true)
    }

    fn __sub__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::sub, false)
    }

    fn __rsub__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::sub, true)
    }

    fn __mul__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::mul, false)
    }

    fn __rmul__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::mul, true)
    }

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::div, false)
    }

    fn __rtruediv__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::div, true)
    }

    fn __floordiv__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::floor_divide, false)
    }

    fn __rfloordiv__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::floor_divide, true)
    }

    fn __mod__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::remainder, false)
    }

    fn __rmod__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::remainder, true)
    }

    fn __pow__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
        _modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::pow, false)
    }

    fn __rpow__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
        _modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::pow, true)
    }

    fn __and__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_and, false)
    }

    fn __rand__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_and, true)
    }

    fn __or__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_or, false)
    }

    fn __ror__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_or, true)
    }

    fn __xor__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_xor, false)
    }

    fn __rxor__<'py>(&self, py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::bitwise_xor, true)
    }

    fn __iadd__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::add_)
    }

    fn __isub__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::sub_)
    }

    fn __imul__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::mul_)
    }

    fn __itruediv__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::div_)
    }

    fn __ifloordiv__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::floor_divide_)
    }

    fn __imod__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::remainder_)
    }

    fn __ipow__(&self, o: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.in_place(o, Tensor::pow_)
    }

    fn __iand__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::bitwise_and_)
    }

    fn __ior__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::bitwise_or_)
    }

    fn __ixor__(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        self.in_place(o, Tensor::bitwise_xor_)
    }

    /// Comparisons are elementwise and return bool tensors, so tensors aren't
    /// hashable (like NumPy arrays).
    fn __richcmp__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
        op: CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        let compare: BinaryFn = match op {
            CompareOp::Eq => Tensor::eq,
            CompareOp::Ne => Tensor::ne,
            CompareOp::Lt => Tensor::lt,
            CompareOp::Le => Tensor::le,
            CompareOp::Gt => Tensor::gt,
            CompareOp::Ge => Tensor::ge,
        };
        self.binary(py, o, compare, false)
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        Ok(self.inner.neg()?.into())
    }

    fn __pos__(&self) -> PyTensor {
        self.inner.deep_copy().into()
    }

    fn __abs__(&self) -> PyTensor {
        self.inner.abs().into()
    }

    /// `~t`: logical not for bools, bitwise not for integers.
    fn __invert__(&self) -> PyResult<PyTensor> {
        Ok(self.inner.bitwise_not()?.into())
    }

    // ---- named math (most are also module functions) ----

    fn add(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::add)
    }

    fn sub(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::sub)
    }

    fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::mul)
    }

    fn div(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::div)
    }

    fn floor_divide(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::floor_divide)
    }

    fn remainder(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::remainder)
    }

    fn pow(&self, exponent: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(exponent, Tensor::pow)
    }

    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::maximum)
    }

    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::minimum)
    }

    fn logical_and(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::logical_and)
    }

    fn logical_or(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::logical_or)
    }

    fn logical_xor(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::logical_xor)
    }

    fn logical_not(&self) -> PyTensor {
        self.inner.logical_not().into()
    }

    fn abs(&self) -> PyTensor {
        self.inner.abs().into()
    }

    fn neg(&self) -> PyResult<PyTensor> {
        Ok(self.inner.neg()?.into())
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

    fn relu(&self) -> PyResult<PyTensor> {
        Ok(self.inner.relu()?.into())
    }

    fn floor(&self) -> PyTensor {
        self.inner.floor().into()
    }

    fn ceil(&self) -> PyTensor {
        self.inner.ceil().into()
    }

    /// Rounds half to even, like NumPy and PyTorch.
    fn round(&self) -> PyTensor {
        self.inner.round().into()
    }

    fn trunc(&self) -> PyTensor {
        self.inner.trunc().into()
    }

    fn isnan(&self) -> PyTensor {
        self.inner.isnan().into()
    }

    fn isinf(&self) -> PyTensor {
        self.inner.isinf().into()
    }

    fn isfinite(&self) -> PyTensor {
        self.inner.isfinite().into()
    }

    #[pyo3(signature = (min=None, max=None))]
    fn clamp(
        &self,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTensor> {
        clamp_impl(&self.inner, min, max).map(Into::into)
    }

    #[pyo3(signature = (other, rtol=1e-5, atol=1e-8))]
    fn allclose(&self, other: &Bound<'_, PyAny>, rtol: f64, atol: f64) -> PyResult<bool> {
        let other = require_operand(other, self.inner.dtype())?;
        Ok(self.inner.allclose(&other, rtol, atol)?)
    }

    // ---- reductions: `dim`/`keepdim` (PyTorch) or `axis`/`keepdims` (NumPy) ----

    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn sum(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::sum, dim, keepdim, axis, keepdims)
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn prod(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::prod, dim, keepdim, axis, keepdims)
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn mean(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::mean, dim, keepdim, axis, keepdims)
    }

    /// The largest values (NumPy's `max`: values only, even with `dim`).
    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn max(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::max, dim, keepdim, axis, keepdims)
    }

    /// The smallest values (NumPy's `min`: values only, even with `dim`).
    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn min(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::min, dim, keepdim, axis, keepdims)
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn all(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::all, dim, keepdim, axis, keepdims)
    }

    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn any(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        reduce_with(&self.inner, Tensor::any, dim, keepdim, axis, keepdims)
    }

    /// Index of the largest value along `dim`, or into the flattened tensor.
    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn argmax(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        let axis = parse_reduce_axis(dim, axis)?;
        Ok(self.inner.argmax(axis, keepdims.unwrap_or(keepdim))?.into())
    }

    /// Index of the smallest value along `dim`, or into the flattened tensor.
    #[pyo3(signature = (dim=None, keepdim=false, *, axis=None, keepdims=None))]
    fn argmin(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        axis: Option<&Bound<'_, PyAny>>,
        keepdims: Option<bool>,
    ) -> PyResult<PyTensor> {
        let axis = parse_reduce_axis(dim, axis)?;
        Ok(self.inner.argmin(axis, keepdims.unwrap_or(keepdim))?.into())
    }

    // ---- linear algebra ----

    fn matmul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.with(other, Tensor::matmul)
    }

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::matmul, false)
    }

    fn __rmatmul__<'py>(
        &self,
        py: Python<'py>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.binary(py, o, Tensor::matmul, true)
    }
}

type ReduceFn = fn(&Tensor, Option<&[isize]>, bool) -> crate::Result<Tensor>;

/// Runs a reduction with PyTorch- or NumPy-style arguments.
pub(crate) fn reduce_with(
    x: &Tensor,
    op: ReduceFn,
    dim: Option<&Bound<'_, PyAny>>,
    keepdim: bool,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: Option<bool>,
) -> PyResult<PyTensor> {
    let axes = parse_reduce_axes(dim, axis)?;
    Ok(op(x, axes.as_deref(), keepdims.unwrap_or(keepdim))?.into())
}

/// `clamp` with bounds given as numbers (taking `x`'s dtype where they fit)
/// or tensors; `None` means unbounded.
pub(crate) fn clamp_impl(
    x: &Tensor,
    min: Option<&Bound<'_, PyAny>>,
    max: Option<&Bound<'_, PyAny>>,
) -> PyResult<Tensor> {
    let bound = |b: Option<&Bound<'_, PyAny>>| -> PyResult<Option<Tensor>> {
        b.filter(|b| !b.is_none())
            .map(|b| require_operand(b, x.dtype()))
            .transpose()
    };
    let (min, max) = (bound(min)?, bound(max)?);
    Ok(x.clamp(min.as_ref(), max.as_ref())?)
}

//! Sharing memory with NumPy (and anything else that speaks Python's buffer
//! protocol) in both directions, without copying.
//!
//! Safety rests on one invariant: tensorsc only holds a storage guard (a Rust
//! slice into the memory) inside its own calls, never while running Python
//! code, and those calls hold the GIL. So NumPy can't read or write memory we
//! share with it while Rust has a slice of that memory.

use std::ffi::{CStr, c_int, c_void};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use pyo3::buffer::ElementType;
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;

use crate::dtype::DType;
use crate::python::pytensor::PyTensor;
use crate::storage::Storage;
use crate::tensor::Tensor;

/// A buffer held on a Python object, released when dropped.
///
/// Like PyO3's `PyUntypedBuffer`, except it accepts 0-d buffers (NumPy
/// scalars and 0-d arrays), whose shape pointer is allowed to be NULL.
struct HeldBuffer(Box<ffi::Py_buffer>);

// SAFETY: the view is only read after it's filled, and released with the
// GIL held (see `Drop`), so it can move and be shared between threads.
unsafe impl Send for HeldBuffer {}
unsafe impl Sync for HeldBuffer {}

impl HeldBuffer {
    fn get(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // SAFETY: a zeroed Py_buffer is a valid "empty" value to fill.
        let mut view = Box::new(unsafe { std::mem::zeroed::<ffi::Py_buffer>() });
        // Strides and format, read-only allowed (the view says if it's writable).
        // SAFETY: `obj` is a valid object and `view` a valid Py_buffer.
        if unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), &mut *view, ffi::PyBUF_RECORDS_RO) } < 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        Ok(HeldBuffer(view))
    }

    fn ndim(&self) -> usize {
        self.0.ndim as usize
    }

    fn shape(&self) -> Vec<usize> {
        if self.ndim() == 0 || self.0.shape.is_null() {
            return vec![];
        }
        // SAFETY: a filled view's `shape` has `ndim` entries.
        unsafe { std::slice::from_raw_parts(self.0.shape, self.ndim()) }
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    /// Strides in bytes. Exporters may leave them out for C-contiguous data.
    fn strides(&self, shape: &[usize]) -> Vec<isize> {
        if self.0.strides.is_null() {
            let mut strides = vec![self.0.itemsize; shape.len()];
            for axis in (0..shape.len().saturating_sub(1)).rev() {
                strides[axis] = strides[axis + 1] * shape[axis + 1] as isize;
            }
            return strides;
        }
        // SAFETY: a filled view's `strides`, when present, has `ndim` entries.
        unsafe { std::slice::from_raw_parts(self.0.strides, self.ndim()) }.to_vec()
    }

    fn format(&self) -> &CStr {
        if self.0.format.is_null() {
            // A missing format means unsigned bytes.
            return c"B";
        }
        // SAFETY: a filled view's `format`, when present, is a C string that
        // lives as long as the view.
        unsafe { CStr::from_ptr(self.0.format) }
    }
}

impl Drop for HeldBuffer {
    fn drop(&mut self) {
        // If Python has already shut down, its memory is gone anyway.
        // SAFETY: the view was filled by PyObject_GetBuffer and not released yet.
        Python::try_attach(|_| unsafe { ffi::PyBuffer_Release(&mut *self.0) });
    }
}

/// True if `obj` exports a buffer (NumPy arrays and scalars, memoryview,
/// bytes, array.array, tensors).
pub(crate) fn has_buffer(obj: &Bound<'_, PyAny>) -> bool {
    // SAFETY: `obj` is a valid object; this only inspects its type.
    unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) == 1 }
}

/// The dtype for a buffer's struct-module format string.
fn buffer_dtype(format: &CStr) -> PyResult<DType> {
    // '>' and '!' mean big-endian, '<' little-endian; no prefix means native.
    let foreign_order: &[u8] = if cfg!(target_endian = "little") {
        b">!"
    } else {
        b"<"
    };
    if format
        .to_bytes()
        .first()
        .is_some_and(|c| foreign_order.contains(c))
    {
        return Err(PyTypeError::new_err(
            "data in non-native byte order isn't supported; convert it first \
             (e.g. arr.astype(arr.dtype.newbyteorder('=')))",
        ));
    }
    Ok(match ElementType::from_format(format) {
        ElementType::Bool => DType::Bool,
        ElementType::UnsignedInteger { bytes: 1 } => DType::U8,
        ElementType::SignedInteger { bytes: 4 } => DType::I32,
        ElementType::SignedInteger { bytes: 8 } => DType::I64,
        ElementType::Float { bytes: 4 } => DType::F32,
        ElementType::Float { bytes: 8 } => DType::F64,
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unsupported element type (buffer format {format:?}); tensorsc supports \
                 bool, uint8, int32, int64, float32 and float64, so convert first \
                 (e.g. arr.astype('float32'))"
            )));
        }
    })
}

/// A tensor over `obj`'s memory, without copying: writes through either side
/// are visible in both, and the memory stays alive as long as the tensor
/// does. Read-only buffers give read-only tensors. Bool data is copied, since
/// bytes other than 0 and 1 aren't valid Rust bools.
pub(crate) fn tensor_from_buffer(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let buffer = HeldBuffer::get(obj)?;
    if !buffer.0.suboffsets.is_null() {
        return Err(PyBufferError::new_err(
            "indirect (suboffset) buffers aren't supported",
        ));
    }
    let dtype = buffer_dtype(buffer.format())?;
    let itemsize = dtype.itemsize();
    if buffer.0.itemsize as usize != itemsize {
        return Err(PyBufferError::new_err(
            "buffer item size doesn't match its format",
        ));
    }
    let shape = buffer.shape();
    if shape.contains(&0) {
        return Ok(Tensor::zeros(&shape, dtype));
    }
    let byte_strides = buffer.strides(&shape);
    let data = buffer.0.buf.cast::<u8>();
    let misaligned =
        data.align_offset(itemsize) != 0 || byte_strides.iter().any(|s| s % itemsize as isize != 0);
    if misaligned {
        return Err(PyBufferError::new_err(
            "the data isn't aligned to its element size; pass a contiguous copy \
             (e.g. np.ascontiguousarray(arr))",
        ));
    }
    let strides: Vec<isize> = byte_strides.iter().map(|s| s / itemsize as isize).collect();
    // The view may run backwards (negative strides), so the storage starts at
    // the lowest element it can reach.
    let (mut lowest, mut highest) = (0isize, 0isize);
    for (&dim, &stride) in shape.iter().zip(&strides) {
        let extent = (dim as isize - 1) * stride;
        if stride < 0 {
            lowest += extent;
        } else {
            highest += extent;
        }
    }
    let base = NonNull::new(data.wrapping_offset(lowest * itemsize as isize))
        .ok_or_else(|| PyBufferError::new_err("buffer has a null data pointer"))?;
    let writable = buffer.0.readonly == 0;
    // Bools are imported as their bytes, then copied into real bools below.
    let storage_dtype = if dtype == DType::Bool {
        DType::U8
    } else {
        dtype
    };
    // SAFETY: the exporter guarantees every element of the view lies in
    // `[base, base + len)`; alignment was checked; `buffer` (the owner) keeps
    // the memory alive and is released when the storage is dropped; and see
    // the module docs for why Python can't touch it while we hold a guard.
    let storage = unsafe {
        Storage::from_external(
            base,
            (highest - lowest + 1) as usize,
            storage_dtype,
            writable,
            Box::new(buffer),
        )
    };
    let tensor = Tensor::from_parts(Arc::new(storage), shape, strides, (-lowest) as usize);
    Ok(if dtype == DType::Bool {
        tensor.to_dtype(DType::Bool)
    } else {
        tensor
    })
}

fn format_of(dtype: DType) -> &'static CStr {
    match dtype {
        DType::Bool => c"?",
        DType::U8 => c"B",
        DType::I32 => c"i",
        DType::I64 => c"q",
        DType::F32 => c"f",
        DType::F64 => c"d",
    }
}

/// Is this view column-major (Fortran) contiguous?
fn is_f_contiguous(t: &Tensor) -> bool {
    let mut expected = 1isize;
    for (&dim, &stride) in t.shape().iter().zip(t.strides()) {
        if dim != 1 && stride != expected {
            return false;
        }
        expected *= dim as isize;
    }
    true
}

/// Fills `view` with a description of the tensor's memory (`__getbuffer__`).
///
/// The buffer is read-only for read-only memory, for bools (so nothing can
/// store bytes other than 0 and 1 in them), and for expanded tensors (where
/// several elements share one memory location).
///
/// # Safety
///
/// `view` must be a valid pointer to a `Py_buffer` to fill, as passed by
/// Python to `__getbuffer__`.
pub(crate) unsafe fn export(
    slf: &Bound<'_, PyTensor>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("view is null"));
    }
    // CPython requires `obj` to be NULL if we fail.
    // SAFETY: `view` is non-null and points to a Py_buffer (see above).
    unsafe { (*view).obj = ptr::null_mut() };
    let t = &slf.get().inner;
    let expanded = t
        .shape()
        .iter()
        .zip(t.strides())
        .any(|(&d, &s)| d > 1 && s == 0);
    let readonly = !t.is_writable() || t.dtype() == DType::Bool || expanded;
    if flags & ffi::PyBUF_WRITABLE != 0 && readonly {
        return Err(PyBufferError::new_err(
            "this tensor can't be written through a buffer (it's read-only, bool, or expanded)",
        ));
    }
    let wants = |flag: c_int| flags & flag == flag;
    let (c_contiguous, f_contiguous) = (t.is_contiguous(), is_f_contiguous(t));
    // Without strides, consumers assume C order.
    let needs_c = !wants(ffi::PyBUF_STRIDES) || wants(ffi::PyBUF_C_CONTIGUOUS);
    if (needs_c && !c_contiguous)
        || (wants(ffi::PyBUF_F_CONTIGUOUS) && !f_contiguous)
        || (wants(ffi::PyBUF_ANY_CONTIGUOUS) && !c_contiguous && !f_contiguous)
    {
        return Err(PyBufferError::new_err(
            "this tensor isn't laid out as requested; call .contiguous() first",
        ));
    }

    let itemsize = t.dtype().itemsize();
    // Shape then strides (in bytes), freed in `release`.
    let mut dims: Vec<isize> = t.shape().iter().map(|&d| d as isize).collect();
    dims.extend(t.strides().iter().map(|&s| s * itemsize as isize));
    let dims = Box::into_raw(Box::new(dims));
    // SAFETY: `view` is valid (checked non-null; the caller passes a real
    // Py_buffer). `dims` lives until `release`. The data pointer is inside the
    // storage, which `obj` (a reference to the tensor) keeps alive.
    unsafe {
        let dims_ptr = (*dims).as_mut_ptr();
        let ndim = t.ndim();
        (*view).obj = slf.clone().into_any().into_ptr();
        (*view).buf = t.storage().as_ptr().add(t.offset() * itemsize) as *mut c_void;
        (*view).len = (t.numel() * itemsize) as isize;
        (*view).readonly = readonly as c_int;
        (*view).itemsize = itemsize as isize;
        (*view).format = if wants(ffi::PyBUF_FORMAT) {
            format_of(t.dtype()).as_ptr().cast_mut()
        } else {
            ptr::null_mut()
        };
        (*view).ndim = ndim as c_int;
        (*view).shape = if wants(ffi::PyBUF_ND) {
            dims_ptr
        } else {
            ptr::null_mut()
        };
        (*view).strides = if wants(ffi::PyBUF_STRIDES) {
            dims_ptr.add(ndim)
        } else {
            ptr::null_mut()
        };
        (*view).suboffsets = ptr::null_mut();
        (*view).internal = dims.cast();
    }
    Ok(())
}

/// Frees what [`export`] allocated (`__releasebuffer__`).
///
/// # Safety
///
/// `view` must be a buffer filled by [`export`] and not released before.
pub(crate) unsafe fn release(view: *mut ffi::Py_buffer) {
    // SAFETY: `internal` is the `Box<Vec<isize>>` `export` leaked.
    unsafe {
        let dims = (*view).internal.cast::<Vec<isize>>();
        if !dims.is_null() {
            drop(Box::from_raw(dims));
            (*view).internal = ptr::null_mut();
        }
    }
}

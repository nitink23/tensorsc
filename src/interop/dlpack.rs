//! DLPack descriptors and ownership, independent of Python.
use crate::dtype::with_dtype;
use crate::strided::{for_each_row, positions};
use crate::{DType, Result, Storage, Tensor, TensorError};
use dlpark::ffi::{DLDataType, DLDevice};
use dlpark::metadata::Dynamic;
use dlpark::{DlpackFlags, Managed, ManagedTensorBase, TensorRef};
use std::ptr::NonNull;
use std::sync::Arc;

fn invalid(message: impl std::fmt::Display) -> TensorError {
    TensorError::InvalidArgument(format!("DLPack: {message}"))
}

pub(crate) fn flags(t: &Tensor) -> DlpackFlags {
    let expanded = t
        .shape()
        .iter()
        .zip(t.strides())
        .any(|(&d, &s)| d > 1 && s == 0);
    if !t.is_writable() || expanded || t.dtype() == DType::Bool {
        DlpackFlags::READ_ONLY
    } else {
        DlpackFlags::empty()
    }
}

fn dtype_to_dl(dtype: DType) -> DLDataType {
    match dtype {
        DType::Bool => DLDataType::BOOL,
        DType::U8 => DLDataType::U8,
        DType::I32 => DLDataType::I32,
        DType::I64 => DLDataType::I64,
        DType::F32 => DLDataType::F32,
        DType::F64 => DLDataType::F64,
    }
}

fn dtype_from_dl(dtype: DLDataType) -> Result<DType> {
    [
        DType::Bool,
        DType::U8,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
    ]
    .into_iter()
    .find(|&d| dtype_to_dl(d) == dtype)
    .ok_or_else(|| invalid(format!("unsupported dtype {dtype:?}")))
}

/// Export shared CPU storage with owned metadata. Tracked tensors must be detached.
pub fn export<M: ManagedTensorBase>(t: &Tensor) -> Result<Managed<M>> {
    if t.requires_grad() {
        return Err(invalid(
            "detach() a tensor requiring gradients before exporting",
        ));
    }
    let mut initialized = Dynamic::new(t.shape(), t.strides())
        .prepare_as::<M>()
        .map_err(invalid)?
        .initialize(Arc::clone(t.storage()));
    initialized
        .set_data(t.storage().as_ptr().cast_mut().cast())
        .set_device(DLDevice::CPU)
        .set_dtype(dtype_to_dl(t.dtype()))
        .set_byte_offset(
            u64::try_from(
                t.offset()
                    .checked_mul(t.dtype().itemsize())
                    .ok_or_else(|| invalid("byte offset overflow"))?,
            )
            .map_err(invalid)?,
        );
    initialized.set_flags(flags(t)).map_err(invalid)?;
    // SAFETY: copied metadata describes our valid view; the context retains the
    // storage allocation. No copy/exclusivity flag is asserted on shared data.
    let managed = unsafe { initialized.finish() };
    managed.validate_export().map_err(invalid)?;
    if managed.flags() != flags(t) {
        return Err(invalid("legacy ABI cannot represent read-only flags"));
    }
    t.storage().mark_externally_shared();
    Ok(managed)
}

/// Import a trusted producer's descriptor while retaining its exactly-once owner.
///
/// # Safety
/// The producer must honor DLPack's pointer/lifetime contract, including valid
/// initialized CPU elements at each reachable address. `owner` must retain this
/// exact allocation, and `flags` must come from its managed descriptor. No foreign thread may
/// mutate the elements concurrently with our Rust reads/writes. We only form a
/// storage slice when all bytes in the reachable span are covered by elements;
/// layouts with holes are copied elementwise (or rejected for copy=false).
pub unsafe fn import(
    view: TensorRef<'_>,
    flags: DlpackFlags,
    owner: Box<dyn std::any::Any + Send + Sync>,
    copy: Option<bool>,
) -> Result<Tensor> {
    if view.device() != DLDevice::CPU {
        return Err(invalid("only CPU device 0 is supported"));
    }
    let dtype = dtype_from_dl(view.dtype())?;
    let shape: Vec<usize> = view
        .shape()
        .iter()
        .map(|&d| usize::try_from(d).map_err(invalid))
        .collect::<Result<_>>()?;
    let strides: Vec<isize> = view
        .strides_or_compact()
        .map_err(invalid)?
        .iter()
        .map(|&s| isize::try_from(s).map_err(invalid))
        .collect::<Result<_>>()?;
    if copy == Some(false) && flags.contains(DlpackFlags::IS_COPIED) {
        return Err(invalid("producer returned a copy despite copy=False"));
    }
    if shape.contains(&0) {
        if copy == Some(true) {
            return Ok(Tensor::zeros(&shape, dtype));
        }
        // No elements are accessed, but preserve layout, read-only status and
        // the producer lifetime. A dangling pointer is valid for a zero slice.
        let ptr = NonNull::<u64>::dangling().cast::<u8>();
        let storage = unsafe {
            Storage::from_external(
                ptr,
                0,
                dtype,
                !flags.contains(DlpackFlags::READ_ONLY),
                owner,
            )
        };
        return Ok(Tensor::from_parts(Arc::new(storage), shape, strides, 0));
    }
    let size = dtype.itemsize();
    let byte_offset = usize::try_from(view.byte_offset()).map_err(invalid)?;
    if byte_offset % size != 0 {
        return Err(invalid("unaligned byte offset"));
    }
    if view.data_ptr().is_null() {
        return Err(invalid("null data pointer"));
    }
    let origin = view.data_ptr().cast::<u8>().wrapping_add(byte_offset);
    if origin.align_offset(size) != 0 {
        return Err(invalid("unaligned data pointer"));
    }
    let (mut low, mut high) = (0i128, 0i128);
    for (&dim, &stride) in shape.iter().zip(&strides) {
        let extent = (dim as i128 - 1)
            .checked_mul(stride as i128)
            .ok_or_else(|| invalid("layout overflow"))?;
        low = low
            .checked_add(extent.min(0))
            .ok_or_else(|| invalid("layout overflow"))?;
        high = high
            .checked_add(extent.max(0))
            .ok_or_else(|| invalid("layout overflow"))?;
    }
    let span = usize::try_from(
        high.checked_sub(low)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| invalid("layout span overflow"))?,
    )
    .map_err(invalid)?;
    if span
        .checked_mul(size)
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(invalid("layout span overflow"));
    }
    let byte_low = isize::try_from(
        low.checked_mul(size as i128)
            .ok_or_else(|| invalid("layout overflow"))?,
    )
    .map_err(invalid)?;
    let offset = usize::try_from(-low).map_err(invalid)?;
    let base = origin.wrapping_offset(byte_low);
    // Dense axes may be permuted, reversed or broadcast. Gaps cannot be
    // represented by Storage's initialized typed slice without extra promises.
    let mut axes: Vec<_> = shape
        .iter()
        .zip(&strides)
        .filter(|&(&d, &s)| d > 1 && s != 0)
        .map(|(&d, &s)| (s.unsigned_abs(), d))
        .collect();
    axes.sort_unstable();
    let mut expected = 1usize;
    let dense = axes.iter().all(|&(stride, dim)| {
        if stride != expected {
            return false;
        }
        match expected.checked_mul(dim) {
            Some(n) => {
                expected = n;
                true
            }
            None => false,
        }
    });
    let needs_copy = copy == Some(true) || dtype == DType::Bool || !dense;
    if needs_copy && copy == Some(false) {
        return Err(invalid("this dtype/layout requires a copy"));
    }
    if needs_copy {
        macro_rules! gather {
            ($ty:ty, $convert:expr) => {{
                let mut values = Vec::with_capacity(view.num_elements());
                for_each_row(&shape, [&strides], [offset], |[p], len, [s]| {
                    for i in positions(p, len, s) {
                        // SAFETY: each position is a reachable element promised
                        // by the producer; no slice is formed over layout gaps.
                        let value = unsafe { base.cast::<$ty>().add(i).read() };
                        values.push(($convert)(value));
                    }
                });
                Tensor::from_vec(values, &shape)
            }};
        }
        return if dtype == DType::Bool {
            gather!(u8, |v| v != 0)
        } else {
            with_dtype!(dtype,T => gather!(T, |v| v))
        };
    }
    let ptr = NonNull::new(base.cast_mut()).ok_or_else(|| invalid("null storage pointer"))?;
    // SAFETY: dense span is initialized and aligned; owner retains the managed
    // allocation; producer access is serialized by the caller's protocol.
    let storage = unsafe {
        Storage::from_external(
            ptr,
            span,
            dtype,
            !flags.contains(DlpackFlags::READ_ONLY),
            owner,
        )
    };
    Ok(Tensor::from_parts(
        Arc::new(storage),
        shape,
        strides,
        offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlpark::ffi::{DLManagedTensor, DLManagedTensorVersioned};

    #[test]
    fn export_retains_storage_until_deleter() {
        let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], &[2, 2]).unwrap();
        let weak = Arc::downgrade(t.storage());
        let managed = export::<DLManagedTensorVersioned>(&t).unwrap();
        assert_eq!(managed.validate().unwrap().shape(), &[2, 2]);
        drop(t);
        assert!(weak.upgrade().is_some());
        drop(managed);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn imports_preserve_dense_negative_and_permuted_views() {
        use crate::TensorIndex;
        let t = Tensor::from_vec((0..12).map(|n| n as f32).collect(), &[3, 4]).unwrap();
        let reversed = t
            .index(&[
                TensorIndex::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                TensorIndex::FULL,
            ])
            .unwrap();
        for source in [t.shallow_clone(), t.transpose(0, 1).unwrap(), reversed] {
            let owner = Arc::new(export::<DLManagedTensorVersioned>(&source).unwrap());
            // SAFETY: descriptor points to our initialized storage; test accesses are sequential.
            let imported = unsafe {
                import(
                    owner.validate().unwrap(),
                    owner.flags(),
                    Box::new(Arc::clone(&owner)),
                    Some(false),
                )
            }
            .unwrap();
            assert_eq!(source.to_vec::<f32>(), imported.to_vec::<f32>());
            assert_eq!(source.strides(), imported.strides());
            assert!(source.overlaps_memory(&imported));
        }
    }

    #[test]
    fn bool_and_gapped_layouts_have_explicit_copy_policy() {
        use crate::TensorIndex;
        let t = Tensor::from_vec((0..8).collect::<Vec<i32>>(), &[8]).unwrap();
        let gapped = t
            .index(&[TensorIndex::Slice {
                start: None,
                stop: None,
                step: 2,
            }])
            .unwrap();
        let owner = Arc::new(export::<DLManagedTensor>(&gapped).unwrap());
        // SAFETY: all reachable elements belong to t; calls are sequential.
        assert!(
            unsafe {
                import(
                    owner.validate().unwrap(),
                    owner.flags(),
                    Box::new(Arc::clone(&owner)),
                    Some(false),
                )
            }
            .is_err()
        );
        let copy = unsafe {
            import(
                owner.validate().unwrap(),
                owner.flags(),
                Box::new(Arc::clone(&owner)),
                None,
            )
        }
        .unwrap();
        assert_eq!(copy.to_vec::<i32>(), vec![0, 2, 4, 6]);
        assert!(!copy.overlaps_memory(&t));
        let b = Tensor::from_vec(vec![true, false], &[2]).unwrap();
        assert!(export::<DLManagedTensor>(&b).is_err());
        assert!(
            export::<DLManagedTensorVersioned>(&b)
                .unwrap()
                .flags()
                .contains(DlpackFlags::READ_ONLY)
        );
    }
}

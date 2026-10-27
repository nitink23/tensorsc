//! Thin Python DLPack negotiation. dlpark owns capsules and their deleters.
use super::pytensor::PyTensor;
use crate::interop::dlpack as bridge;
use dlpark::ffi::{DLDevice, DLManagedTensor, DLManagedTensorVersioned};
use dlpark::python::{DlpackExporter, ExportRequest, ImportRequest};
use dlpark::{DlpackFlags, Managed};
use pyo3::exceptions::{PyAttributeError, PyBufferError};
use pyo3::prelude::*;

pub(crate) fn error(e: impl std::fmt::Display) -> PyErr {
    PyBufferError::new_err(e.to_string())
}

impl DlpackExporter for PyTensor {
    fn device(&self) -> DLDevice {
        DLDevice::CPU
    }
    fn flags(&self) -> DlpackFlags {
        bridge::flags(&self.inner)
    }
    fn prepare_export(&self, _py: Python<'_>, request: &ExportRequest<'_>) -> PyResult<()> {
        if request.stream().is_some_and(|s| !s.is_none()) {
            return Err(error("CPU DLPack export requires stream=None"));
        }
        if self.inner.requires_grad() {
            return Err(error(
                "detach() a tensor requiring gradients before DLPack export",
            ));
        }
        Ok(())
    }
    fn export_legacy(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensor>> {
        bridge::export(&self.inner).map_err(error)
    }
    fn export_versioned(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensorVersioned>> {
        bridge::export(&self.inner).map_err(error)
    }
}

#[pyfunction]
#[pyo3(signature=(obj, *, copy=None))]
pub(crate) fn from_dlpack(obj: &Bound<'_, PyAny>, copy: Option<bool>) -> PyResult<PyTensor> {
    let request = ImportRequest::new(obj.as_borrowed())?;
    match request.device() {
        Ok(device) if device != DLDevice::CPU => {
            return Err(error("only CPU DLPack tensors are supported"));
        }
        Err(e) if !e.is_instance_of::<PyAttributeError>(obj.py()) => return Err(e),
        _ => {}
    }
    // Perform requested copies locally, so legacy producers need not support
    // the newer copy keyword. Negotiation and capsule consumption stay in dlpark.
    let imported = std::sync::Arc::new(request.import(None, None)?);
    let view = imported.validate().map_err(error)?;
    let flags = imported.flags();
    // SAFETY: Python's producer promises DLPack-valid CPU elements. We keep
    // its managed owner alive, hold the GIL, and never call Python with storage
    // slices alive. Consumers must not mutate from foreign native threads.
    unsafe {
        bridge::import(
            view,
            flags,
            Box::new(std::sync::Arc::clone(&imported)),
            copy,
        )
    }
    .map(PyTensor::from)
    .map_err(error)
}

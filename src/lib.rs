//! tensorsc: tensors from scratch.
//!
//! The layers mirror a production engine, bottom to top:
//! - [`Device`]: where memory lives (CPU only, for now).
//! - [`Storage`]: owns a flat buffer on a device. Shared between tensors via `Arc`.
//! - [`Tensor`]: a *view* onto a storage, described by shape, strides, offset and dtype.

pub mod device;
pub mod dtype;
pub mod error;
pub mod storage;
pub mod tensor;
pub mod views;

#[cfg(feature = "python")]
mod python;

pub use device::Device;
pub use dtype::DType;
pub use error::{Result, TensorError};
pub use storage::Storage;
pub use tensor::Tensor;
pub use views::TensorIndex;

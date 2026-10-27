//! tensorsc: tensors from scratch.
//!
//! The layers mirror a production engine, bottom to top:
//! - [`Device`]: where memory lives (CPU only, for now).
//! - [`Storage`]: owns a flat buffer on a device. Shared between tensors via `Arc`.
//! - [`Tensor`]: a *view* onto a storage, described by shape, strides, offset and dtype.

mod arith;
pub mod autograd;
pub mod device;
pub mod dtype;
pub mod error;
pub mod indexing;
#[cfg(feature = "dlpack")]
pub mod interop;
mod kernels;
pub mod linalg;
pub mod ops;
pub mod reduce;
pub mod storage;
mod strided;
pub mod tensor;
pub mod views;

#[cfg(feature = "python")]
mod python;

pub use device::Device;
pub use dtype::DType;
pub use error::{Result, TensorError};
pub use ops::broadcast_shapes;
pub use storage::Storage;
pub use tensor::Tensor;
pub use views::TensorIndex;

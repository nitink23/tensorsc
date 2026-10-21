use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// The number of data elements doesn't match the product of the shape.
    DataShapeMismatch {
        len: usize,
        shape: Vec<usize>,
    },
    /// An index had the wrong number of coordinates.
    IndexRankMismatch {
        expected: usize,
        got: usize,
    },
    /// More axes were indexed than the tensor has.
    TooManyIndices {
        ndim: usize,
        got: usize,
    },
    /// One coordinate of an index is past the end of its axis.
    IndexOutOfBounds {
        axis: usize,
        index: isize,
        size: usize,
    },
    /// An axis (dimension) argument doesn't exist on the tensor.
    AxisOutOfRange {
        axis: isize,
        ndim: usize,
    },
    /// Two tensors needed the same shape and didn't have it.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Two shapes can't be broadcast against each other.
    BroadcastMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::DataShapeMismatch { len, shape } => {
                let expected: usize = shape.iter().product();
                write!(
                    f,
                    "{len} elements cannot fill shape {shape:?} (needs {expected})"
                )
            }
            TensorError::IndexRankMismatch { expected, got } => {
                write!(f, "expected {expected} indices, got {got}")
            }
            TensorError::TooManyIndices { ndim, got } => {
                write!(
                    f,
                    "too many indices: tensor is {ndim}-dimensional, but {got} were indexed"
                )
            }
            TensorError::IndexOutOfBounds { axis, index, size } => {
                write!(
                    f,
                    "index {index} is out of bounds for axis {axis} with size {size}"
                )
            }
            TensorError::AxisOutOfRange { axis, ndim } => {
                write!(
                    f,
                    "axis {axis} is out of range for a {ndim}-dimensional tensor"
                )
            }
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            TensorError::BroadcastMismatch { left, right } => {
                write!(
                    f,
                    "shapes {left:?} and {right:?} cannot be broadcast together"
                )
            }
            TensorError::InvalidArgument(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TensorError {}

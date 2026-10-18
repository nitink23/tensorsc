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
    /// One coordinate of an index is past the end of its axis.
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        size: usize,
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
            TensorError::IndexOutOfBounds { axis, index, size } => {
                write!(
                    f,
                    "index {index} is out of bounds for axis {axis} with size {size}"
                )
            }
            TensorError::InvalidArgument(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TensorError {}

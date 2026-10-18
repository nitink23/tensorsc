use std::fmt;

/// Element type of a tensor. Only `F32` exists today; the field is carried
/// everywhere so adding more types later doesn't change the tensor layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    #[default]
    F32,
}

impl DType {
    /// Size of one element in bytes.
    pub fn itemsize(self) -> usize {
        match self {
            DType::F32 => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

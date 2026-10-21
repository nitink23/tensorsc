use std::fmt;
use std::sync::Arc;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, TensorError};
use crate::storage::Storage;

/// An n-dimensional view onto a [`Storage`].
///
/// The element at index `[i0, i1, ...]` lives at storage position
/// `offset + i0 * strides[0] + i1 * strides[1] + ...`. Changing shape, strides
/// and offset (without touching the buffer) is how reshape, transpose and
/// slicing avoid copying data; see `views.rs`. Strides are signed so reversed
/// views (`t[::-1]`) can be expressed too.
#[derive(Debug)]
pub struct Tensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: usize,
    pub(crate) dtype: DType,
}

/// Row-major strides: the last axis is contiguous in memory.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1] as isize;
    }
    strides
}

impl Tensor {
    // ---- creation ----

    /// Wraps `data` (row-major) in a new tensor with the given shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TensorError::DataShapeMismatch {
                len: data.len(),
                shape: shape.to_vec(),
            });
        }
        let storage = Arc::new(Storage::new(data, Device::Cpu));
        Ok(Tensor::from_parts(
            storage,
            shape.to_vec(),
            contiguous_strides(shape),
            0,
        ))
    }

    /// Builds a view onto existing storage. Callers must make sure every
    /// reachable position is inside the buffer.
    pub(crate) fn from_parts(
        storage: Arc<Storage>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Tensor {
        debug_assert_eq!(shape.len(), strides.len());
        Tensor {
            storage,
            shape,
            strides,
            offset,
            dtype: DType::F32,
        }
    }

    /// Another view onto this tensor's storage with a different layout.
    pub(crate) fn with_layout(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Tensor {
        debug_assert_eq!(shape.len(), strides.len());
        Tensor {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset,
            dtype: self.dtype,
        }
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        let numel = shape.iter().product();
        Tensor::from_vec(vec![value; numel], shape).expect("length matches shape")
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    /// A 0-dimensional tensor holding one value.
    pub fn scalar(value: f32) -> Tensor {
        Tensor::full(&[], value)
    }

    /// Evenly spaced values in `[start, end)`, like `numpy.arange`.
    pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor> {
        if step == 0.0 || !step.is_finite() {
            return Err(TensorError::InvalidArgument(
                "arange step must be a non-zero finite number".into(),
            ));
        }
        let len = ((end - start) / step).ceil().max(0.0) as usize;
        let data = (0..len).map(|i| start + i as f32 * step).collect();
        Tensor::from_vec(data, &[len])
    }

    /// A new tensor with its own contiguous copy of this tensor's elements.
    pub fn deep_copy(&self) -> Tensor {
        Tensor::from_vec(self.to_vec(), &self.shape).expect("length matches shape")
    }

    // ---- metadata ----

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides in elements (not bytes), like PyTorch.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// True when elements are laid out row-major with no gaps, i.e. the view
    /// could be handed to code that expects a plain flat array.
    pub fn is_contiguous(&self) -> bool {
        let expected = contiguous_strides(&self.shape);
        self.shape
            .iter()
            .zip(self.strides.iter().zip(&expected))
            // Axes of length 1 never move, so their stride doesn't matter.
            .all(|(&dim, (&stride, &want))| dim == 1 || stride == want)
    }

    /// True when both tensors are views onto the same buffer.
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    // ---- element access ----

    fn storage_index(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.ndim() {
            return Err(TensorError::IndexRankMismatch {
                expected: self.ndim(),
                got: index.len(),
            });
        }
        let mut pos = self.offset as isize;
        for (axis, (&i, &size)) in index.iter().zip(&self.shape).enumerate() {
            if i >= size {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: i as isize,
                    size,
                });
            }
            pos += i as isize * self.strides[axis];
        }
        Ok(pos as usize)
    }

    pub fn get(&self, index: &[usize]) -> Result<f32> {
        let pos = self.storage_index(index)?;
        Ok(self.storage.read()[pos])
    }

    /// Writes one element. Takes `&self` because storage is shared: the write
    /// is visible through every view of the same buffer.
    pub fn set(&self, index: &[usize], value: f32) -> Result<()> {
        let pos = self.storage_index(index)?;
        self.storage.write()[pos] = value;
        Ok(())
    }

    /// The value of a one-element tensor, whatever its shape.
    pub fn item(&self) -> Result<f32> {
        if self.numel() != 1 {
            return Err(TensorError::InvalidArgument(format!(
                "only one-element tensors can be converted to a scalar, this one has {}",
                self.numel()
            )));
        }
        Ok(self.storage.read()[self.offset])
    }

    /// Sets every element of this view to `value`.
    pub fn fill(&self, value: f32) {
        let positions = self.storage_positions();
        let mut data = self.storage.write();
        for p in positions {
            data[p] = value;
        }
    }

    /// Copies `src` element-by-element into this view. `src` must have the same
    /// shape, or hold a single element (which is then written everywhere).
    /// Safe even when `src` overlaps this view, e.g. `t[0] = t[1]`.
    pub fn copy_from(&self, src: &Tensor) -> Result<()> {
        if src.shape != self.shape {
            if src.numel() == 1 {
                self.fill(src.item()?);
                return Ok(());
            }
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: src.shape.clone(),
            });
        }
        // Read everything before taking the write lock: src may share our storage.
        let values = src.to_vec();
        let positions = self.storage_positions();
        let mut data = self.storage.write();
        for (p, v) in positions.into_iter().zip(values) {
            data[p] = v;
        }
        Ok(())
    }

    /// Storage positions of every element, in logical row-major order.
    pub(crate) fn storage_positions(&self) -> Vec<usize> {
        let numel = self.numel();
        let mut positions = Vec::with_capacity(numel);
        let mut index = vec![0usize; self.ndim()];
        let mut pos = self.offset as isize;
        for _ in 0..numel {
            positions.push(pos as usize);
            // Advance like an odometer: bump the last axis, carry into earlier ones.
            for axis in (0..self.ndim()).rev() {
                index[axis] += 1;
                pos += self.strides[axis];
                if index[axis] < self.shape[axis] {
                    break;
                }
                pos -= self.strides[axis] * self.shape[axis] as isize;
                index[axis] = 0;
            }
        }
        positions
    }

    /// All elements in logical row-major order, regardless of memory layout.
    pub fn to_vec(&self) -> Vec<f32> {
        let data = self.storage.read();
        self.storage_positions()
            .into_iter()
            .map(|p| data[p])
            .collect()
    }
}

fn fmt_nested(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    match shape.split_first() {
        None => write!(f, "{:?}", data[0]),
        Some((&len, rest)) => {
            let chunk: usize = rest.iter().product();
            f.write_str("[")?;
            for i in 0..len {
                if i > 0 {
                    f.write_str(", ")?;
                }
                fmt_nested(f, &data[i * chunk..(i + 1) * chunk], rest)?;
            }
            f.write_str("]")
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor(")?;
        fmt_nested(f, &self.to_vec(), &self.shape)?;
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert_eq!(contiguous_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn from_vec_checks_length() {
        assert_eq!(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err(),
            TensorError::DataShapeMismatch {
                len: 3,
                shape: vec![2, 2]
            }
        );
    }

    #[test]
    fn get_and_set() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(x.get(&[1, 2]).unwrap(), 6.0);
        x.set(&[0, 1], 9.0).unwrap();
        assert_eq!(x.to_vec(), vec![1.0, 9.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(
            x.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds { axis: 0, .. })
        ));
        assert!(matches!(
            x.get(&[0]),
            Err(TensorError::IndexRankMismatch { .. })
        ));
    }

    #[test]
    fn writes_are_shared_between_views() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let flat = Tensor::from_parts(x.storage.clone(), vec![4], vec![1], 0);
        flat.set(&[3], 40.0).unwrap();
        assert_eq!(x.get(&[1, 1]).unwrap(), 40.0);
        assert!(x.shares_storage(&flat));
        assert!(!x.shares_storage(&x.deep_copy()));
    }

    #[test]
    fn strided_views_read_in_logical_order() {
        // [[1, 2, 3], [4, 5, 6]] viewed transposed as [[1, 4], [2, 5], [3, 6]].
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let xt = Tensor::from_parts(x.storage.clone(), vec![3, 2], vec![1, 3], 0);
        assert_eq!(xt.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(!xt.is_contiguous());
        assert!(xt.deep_copy().is_contiguous());

        // Reversed view via a negative stride.
        let rev = Tensor::from_parts(x.storage.clone(), vec![6], vec![-1], 5);
        assert_eq!(rev.to_vec(), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn creation_helpers() {
        assert_eq!(Tensor::zeros(&[2, 2]).to_vec(), vec![0.0; 4]);
        assert_eq!(Tensor::ones(&[3]).to_vec(), vec![1.0; 3]);
        assert_eq!(Tensor::scalar(7.0).item().unwrap(), 7.0);
        assert_eq!(
            Tensor::arange(0.0, 1.0, 0.25).unwrap().to_vec(),
            vec![0.0, 0.25, 0.5, 0.75]
        );
        assert_eq!(
            Tensor::arange(5.0, 0.0, -2.0).unwrap().to_vec(),
            vec![5.0, 3.0, 1.0]
        );
        assert_eq!(Tensor::arange(0.0, -1.0, 1.0).unwrap().numel(), 0);
        assert!(Tensor::arange(0.0, 1.0, 0.0).is_err());
        assert!(Tensor::zeros(&[2, 0]).to_vec().is_empty());
    }

    #[test]
    fn display_is_nested() {
        assert_eq!(
            t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).to_string(),
            "tensor([[1.0, 2.0], [3.0, 4.0]])"
        );
        assert_eq!(Tensor::scalar(2.5).to_string(), "tensor(2.5)");
    }
}

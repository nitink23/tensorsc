//! Advanced indexing: selecting by a bool mask (`t[t > 0]`) or by an array of
//! integer positions (`t[[0, 2]]`).
//!
//! The selected elements aren't evenly spaced, so unlike basic indexing
//! (`views.rs`) the results are copies, as in NumPy. Masks and index arrays
//! are read into plain vectors first, so at most one storage lock is held.

use crate::dtype::{DType, with_dtype};
use crate::error::{Result, TensorError};
use crate::strided::{for_each_row, positions};
use crate::tensor::Tensor;
use crate::views::normalize_axis;

impl Tensor {
    /// One flag per element of `self` (row-major), from a bool `mask` that
    /// covers the leading axes, plus how many flags are set.
    fn mask_flags(&self, mask: &Tensor) -> Result<(Vec<bool>, usize)> {
        if mask.dtype() != DType::Bool {
            return Err(TensorError::InvalidIndex(format!(
                "a mask must be a bool tensor, not {}",
                mask.dtype()
            )));
        }
        if mask.ndim() > self.ndim() || mask.shape() != &self.shape[..mask.ndim()] {
            return Err(TensorError::InvalidIndex(format!(
                "mask of shape {:?} doesn't match the leading dimensions of shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }
        let count = mask.to_vec::<bool>().into_iter().filter(|&m| m).count();
        // Repeat each mask entry across the axes it doesn't cover.
        let mut strides = mask.strides.clone();
        strides.resize(self.ndim(), 0);
        let flags = mask
            .with_layout(self.shape.clone(), strides, mask.offset)
            .to_vec::<bool>();
        Ok((flags, count))
    }

    /// Shape of `self[mask]`: the masked axes collapse into one.
    fn masked_shape(&self, mask: &Tensor, count: usize) -> Vec<usize> {
        let mut shape = vec![count];
        shape.extend_from_slice(&self.shape[mask.ndim()..]);
        shape
    }

    /// The elements (or sub-tensors) where `mask` is true, as a copy. `mask`
    /// covers the leading axes; the result has shape `[trues, ...rest]`.
    pub fn index_mask(&self, mask: &Tensor) -> Result<Tensor> {
        let (flags, count) = self.mask_flags(mask)?;
        let shape = self.masked_shape(mask, count);
        with_dtype!(self.dtype(), T => {
            let data = self.storage.read::<T>();
            let mut out = Vec::with_capacity(shape.iter().product());
            let mut k = 0;
            for_each_row(&self.shape, [&self.strides], [self.offset], |[p], len, [s]| {
                for (i, &keep) in positions(p, len, s).zip(&flags[k..k + len]) {
                    if keep {
                        out.push(data[i]);
                    }
                }
                k += len;
            });
            Tensor::from_vec(out, &shape)
        })
    }

    /// `self[mask] = values`: `values` is broadcast to the shape `self[mask]`
    /// would have and written into the selected positions.
    pub fn index_mask_put(&self, mask: &Tensor, values: &Tensor) -> Result<()> {
        self.ensure_writable_elementwise()?;
        let (flags, count) = self.mask_flags(mask)?;
        let shape = self.masked_shape(mask, count);
        let values = values
            .broadcast_to(&shape)
            .map_err(|_| TensorError::ShapeMismatch {
                expected: shape.clone(),
                got: values.shape().to_vec(),
            })?;
        with_dtype!(self.dtype(), T => {
            // Read the values before locking: they may share our storage.
            let mut values = values.to_vec::<T>().into_iter();
            let mut data = self.storage.write::<T>()?;
            let mut k = 0;
            for_each_row(&self.shape, [&self.strides], [self.offset], |[p], len, [s]| {
                for (i, &keep) in positions(p, len, s).zip(&flags[k..k + len]) {
                    if keep {
                        data[i] = values.next().expect("one value per selected element");
                    }
                }
                k += len;
            });
            Ok(())
        })
    }

    /// Integer `indices` resolved against `axis` (negatives count from the end).
    fn resolve_indices(&self, axis: usize, indices: &Tensor) -> Result<Vec<usize>> {
        if indices.dtype().is_float() || indices.dtype() == DType::Bool {
            return Err(TensorError::InvalidIndex(format!(
                "index arrays must hold integers, not {}",
                indices.dtype()
            )));
        }
        let size = self.shape[axis];
        indices
            .to_vec::<i64>()
            .into_iter()
            .map(|i| {
                let resolved = if i < 0 { i + size as i64 } else { i };
                if resolved < 0 || resolved >= size as i64 {
                    return Err(TensorError::IndexOutOfBounds {
                        axis,
                        index: i as isize,
                        size,
                    });
                }
                Ok(resolved as usize)
            })
            .collect()
    }

    /// A view with `axis` moved to the front.
    fn axis_to_front(&self, axis: usize) -> Tensor {
        let order: Vec<isize> = std::iter::once(axis)
            .chain((0..self.ndim()).filter(|&a| a != axis))
            .map(|a| a as isize)
            .collect();
        self.permute(&order).expect("a valid permutation")
    }

    /// The sub-tensors at `indices` along `axis`, as a copy. The index array's
    /// shape replaces that axis: `[2, 5].index_select(1, [[0, 1], [4, 4]])`
    /// has shape `[2, 2, 2]`.
    pub fn index_select(&self, axis: isize, indices: &Tensor) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.ndim())?;
        let picks = self.resolve_indices(axis, indices)?;
        let front = self.axis_to_front(axis);
        let rest = &front.shape[1..];
        let gathered = with_dtype!(self.dtype(), T => {
            let mut out = Vec::with_capacity(picks.len() * rest.iter().product::<usize>());
            for &i in &picks {
                out.extend(front.select(0, i as isize)?.to_vec::<T>());
            }
            Tensor::from_vec(out, &[indices.shape(), rest].concat())?
        });
        // `gathered` is [index dims, axes before `axis`, axes after]; put the
        // index dims back where `axis` was.
        let k = indices.ndim();
        let order: Vec<isize> = (k..k + axis)
            .chain(0..k)
            .chain(k + axis..gathered.ndim())
            .map(|a| a as isize)
            .collect();
        Ok(gathered.permute(&order)?.contiguous())
    }

    /// `self[indices] = values` along `axis`: `values` is broadcast to the
    /// shape [`Tensor::index_select`] would return. With repeated indices
    /// the last write wins.
    pub fn index_put(&self, axis: isize, indices: &Tensor, values: &Tensor) -> Result<()> {
        self.ensure_writable_elementwise()?;
        let axis = normalize_axis(axis, self.ndim())?;
        let picks = self.resolve_indices(axis, indices)?;
        let front = self.axis_to_front(axis);

        // Lay the values out like `front`'s selected rows: [picks, rest...].
        let mut target = self.shape.clone();
        target.splice(axis..=axis, indices.shape().iter().copied());
        let values = values
            .broadcast_to(&target)
            .map_err(|_| TensorError::ShapeMismatch {
                expected: target.clone(),
                got: values.shape().to_vec(),
            })?;
        let k = indices.ndim();
        let order: Vec<isize> = (axis..axis + k)
            .chain(0..axis)
            .chain(axis + k..values.ndim())
            .map(|a| a as isize)
            .collect();
        let mut rows_shape = vec![picks.len()];
        rows_shape.extend_from_slice(&front.shape[1..]);
        // A copy, so writing rows below can't change values we haven't read yet.
        let rows = values
            .permute(&order)?
            .deep_copy()
            .reshape(&rows_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())?;
        for (row, &i) in picks.iter().enumerate() {
            front
                .select(0, i as isize)?
                .copy_from(&rows.select(0, row as isize)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::error::TensorError;
    use crate::tensor::Tensor;

    fn range(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec((0..n as i64).collect(), shape).unwrap()
    }

    fn ints(data: &[i64], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    fn mask(data: &[bool], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn full_mask_selects_elements() {
        let x = range(&[2, 3]);
        let m = x.gt(&Tensor::scalar(2i64)).unwrap();
        let picked = x.index_mask(&m).unwrap();
        assert_eq!(picked.shape(), &[3]);
        assert_eq!(picked.to_vec::<i64>(), vec![3, 4, 5]);
        assert!(!picked.shares_storage(&x));
        // Strided input, and a mask that selects nothing.
        let t = x.reversed_axes();
        let m = mask(&[true, false, false, true, true, false], &[3, 2]);
        assert_eq!(t.index_mask(&m).unwrap().to_vec::<i64>(), vec![0, 4, 2]);
        let none = Tensor::zeros(&[2, 3], DType::Bool);
        assert_eq!(x.index_mask(&none).unwrap().shape(), &[0]);
    }

    #[test]
    fn leading_mask_selects_rows() {
        let x = range(&[3, 2]);
        let picked = x.index_mask(&mask(&[true, false, true], &[3])).unwrap();
        assert_eq!(picked.shape(), &[2, 2]);
        assert_eq!(picked.to_vec::<i64>(), vec![0, 1, 4, 5]);
    }

    #[test]
    fn bad_masks_are_rejected() {
        let x = range(&[3, 2]);
        assert!(matches!(
            x.index_mask(&mask(&[true, false], &[2])),
            Err(TensorError::InvalidIndex(_))
        ));
        assert!(
            x.index_mask(&ints(&[1, 0, 1], &[3])).is_err(),
            "not a bool tensor"
        );
    }

    #[test]
    fn mask_put_writes_through_views() {
        let x = range(&[2, 3]);
        let m = x.ge(&Tensor::scalar(4i64)).unwrap();
        x.index_mask_put(&m, &Tensor::scalar(-1i64)).unwrap();
        assert_eq!(x.to_vec::<i64>(), vec![0, 1, 2, 3, -1, -1]);
        x.index_mask_put(&mask(&[true, false], &[2]), &ints(&[7, 8, 9], &[3]))
            .unwrap();
        assert_eq!(x.to_vec::<i64>(), vec![7, 8, 9, 3, -1, -1]);
        // Values converted to the tensor's dtype; a wrong count is an error.
        let f = Tensor::zeros(&[3], DType::F32);
        f.index_mask_put(&mask(&[true, false, true], &[3]), &ints(&[1, 2], &[2]))
            .unwrap();
        assert_eq!(f.to_vec::<f32>(), vec![1.0, 0.0, 2.0]);
        assert!(
            f.index_mask_put(&mask(&[true, true, true], &[3]), &ints(&[1, 2], &[2]))
                .is_err()
        );
        // A bool tensor masked by itself.
        let b = mask(&[true, false, true], &[3]);
        b.index_mask_put(&b, &Tensor::scalar(false)).unwrap();
        assert_eq!(b.to_vec::<bool>(), vec![false; 3]);
    }

    #[test]
    fn index_select_along_axes() {
        let x = range(&[3, 4]);
        let rows = x.index_select(0, &ints(&[2, 0, -1], &[3])).unwrap();
        assert_eq!(rows.shape(), &[3, 4]);
        assert_eq!(
            rows.to_vec::<i64>(),
            vec![8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11]
        );
        let cols = x.index_select(1, &ints(&[3, 1], &[2])).unwrap();
        assert_eq!(cols.shape(), &[3, 2]);
        assert_eq!(cols.to_vec::<i64>(), vec![3, 1, 7, 5, 11, 9]);
        let grid = x.index_select(1, &ints(&[0, 1, 3, 3], &[2, 2])).unwrap();
        assert_eq!(grid.shape(), &[3, 2, 2]);
        assert_eq!(grid.to_vec::<i64>()[..4], [0, 1, 3, 3]);
        assert_eq!(
            x.index_select(0, &Tensor::scalar(1i64)).unwrap().shape(),
            &[4]
        );
    }

    #[test]
    fn index_select_errors() {
        let x = range(&[3, 4]);
        assert!(matches!(
            x.index_select(0, &ints(&[3], &[1])),
            Err(TensorError::IndexOutOfBounds { index: 3, .. })
        ));
        assert!(
            x.index_select(0, &Tensor::from_vec(vec![0.0f32], &[1]).unwrap())
                .is_err()
        );
        assert!(x.index_select(2, &ints(&[0], &[1])).is_err());
    }

    #[test]
    fn index_put_writes_rows_and_columns() {
        let x = Tensor::zeros(&[3, 3], DType::I64);
        x.index_put(0, &ints(&[0, 2], &[2]), &ints(&[1, 2, 3], &[3]))
            .unwrap();
        x.index_put(1, &ints(&[1], &[1]), &ints(&[7, 8, 9], &[3, 1]))
            .unwrap();
        assert_eq!(x.to_vec::<i64>(), vec![1, 7, 3, 0, 8, 0, 1, 9, 3]);
        // Repeated indices: the last value wins.
        let y = Tensor::zeros(&[2], DType::I64);
        y.index_put(0, &ints(&[1, 1], &[2]), &ints(&[5, 6], &[2]))
            .unwrap();
        assert_eq!(y.to_vec::<i64>(), vec![0, 6]);
        // Values that alias the destination are read before anything is written.
        let z = range(&[3]);
        z.index_put(
            0,
            &ints(&[0, 1, 2], &[3]),
            &z.slice(0, None, None, -1).unwrap(),
        )
        .unwrap();
        assert_eq!(z.to_vec::<i64>(), vec![2, 1, 0]);
    }
}

//! Matrix multiplication.
//!
//! `matmul` follows NumPy and PyTorch: the last two axes are the matrices,
//! leading axes are batches that broadcast, and a 1-D operand is treated as
//! a row (left) or column (right) vector whose extra axis is dropped from the
//! result.
//!
//! The kernel is the classic i-k-j loop: for each row of the left matrix it
//! adds scaled rows of the right matrix into the output row, so the innermost
//! loop runs over contiguous memory and vectorizes. It's a straightforward,
//! correct kernel, not a tuned BLAS (no cache blocking or threads yet).

use crate::arith::{Numeric, with_numeric};
use crate::dtype::DType;
use crate::error::{Result, TensorError};
use crate::kernels::read_both;
use crate::ops::broadcast_shapes;
use crate::strided::{for_each_row, positions};
use crate::tensor::{Tensor, contiguous_strides};

/// `c += a @ b` for row-major `a` (m x k), `b` (k x n) and `c` (m x n).
fn gemm<T: Numeric>(a: &[T], b: &[T], c: &mut [T], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        for p in 0..k {
            let a_ip = a[i * k + p];
            let b_row = &b[p * n..(p + 1) * n];
            for (c, &b) in c_row.iter_mut().zip(b_row) {
                *c = c.add(a_ip.mul(b));
            }
        }
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() == 0 || other.ndim() == 0 {
            return Err(TensorError::InvalidArgument(
                "matmul needs at least 1-D operands (use * to multiply by a scalar)".into(),
            ));
        }
        let dtype = self.dtype().promote(other.dtype());
        if dtype == DType::Bool {
            return Err(TensorError::UnsupportedDType {
                op: "matmul",
                dtype,
            });
        }
        // Vectors become 1 x k and k x 1 matrices; the extra axes go away at the end.
        let a = match self.ndim() {
            1 => self.unsqueeze(0)?,
            _ => self.shallow_clone(),
        };
        let b = match other.ndim() {
            1 => other.unsqueeze(-1)?,
            _ => other.shallow_clone(),
        };
        let (a_batch, &[m, k]) = a.shape.split_at(a.ndim() - 2) else {
            unreachable!("at least 2-D")
        };
        let (b_batch, &[k2, n]) = b.shape.split_at(b.ndim() - 2) else {
            unreachable!("at least 2-D")
        };
        if k != k2 {
            return Err(TensorError::InvalidArgument(format!(
                "matmul: shapes {:?} and {:?} don't line up ({k} != {k2})",
                self.shape, other.shape
            )));
        }
        let batch = broadcast_shapes(a_batch, b_batch)?;

        // Contiguous matrices for the kernel (a copy only if needed), then
        // broadcast the batch axes as views.
        let a = a
            .to_dtype(dtype)
            .contiguous()
            .broadcast_to(&[&batch[..], &[m, k]].concat())?;
        let b = b
            .to_dtype(dtype)
            .contiguous()
            .broadcast_to(&[&batch[..], &[k, n]].concat())?;
        let batch_numel: usize = batch.iter().product();
        let out_batch_strides = contiguous_strides(&batch);
        let (a_batch_strides, b_batch_strides) =
            (&a.strides[..batch.len()], &b.strides[..batch.len()]);

        let mut out_shape = [&batch[..], &[m, n]].concat();
        let out = with_numeric!(dtype, T => {
            let mut out = vec![T::ZERO; batch_numel * m * n];
            read_both::<T, _>(&a, &b, |da, db| {
                // Visit every batch entry, getting each operand's matrix offset.
                for_each_row(
                    &batch,
                    [&out_batch_strides, a_batch_strides, b_batch_strides],
                    [0, a.offset, b.offset],
                    |[po, pa, pb], len, [so, sa, sb]| {
                        for ((o, i), j) in positions(po, len, so)
                            .zip(positions(pa, len, sa))
                            .zip(positions(pb, len, sb))
                        {
                            let c = &mut out[o * m * n..(o + 1) * m * n];
                            gemm(&da[i..], &db[j..], c, m, k, n);
                        }
                    },
                )
            });
            Tensor::from_vec(out, &out_shape)?
        }, bool => unreachable!("rejected above"));

        // Drop the axes that 1-D operands were given.
        if other.ndim() == 1 {
            out_shape.pop();
        }
        if self.ndim() == 1 {
            out_shape.remove(out_shape.len() - if other.ndim() == 1 { 1 } else { 2 });
        }
        out.reshape(&out_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::tensor::Tensor;

    fn range(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec((0..n as i64).collect(), shape).unwrap()
    }

    /// matmul by the definition, one output element at a time.
    fn reference(a: &Tensor, b: &Tensor) -> Vec<i64> {
        let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
        let mut out = vec![];
        for i in 0..m {
            for j in 0..n {
                out.push(
                    (0..k)
                        .map(|p| {
                            a.get(&[i, p]).unwrap().to_i64() * b.get(&[p, j]).unwrap().to_i64()
                        })
                        .sum(),
                );
            }
        }
        out
    }

    #[test]
    fn matrix_times_matrix() {
        let a = range(&[3, 4]);
        let b = range(&[4, 5]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[3, 5]);
        assert_eq!(c.to_vec::<i64>(), reference(&a, &b));
        // Non-contiguous operands give the same answer.
        let at = range(&[4, 3]).reversed_axes();
        assert_eq!(at.matmul(&b).unwrap().to_vec::<i64>(), reference(&at, &b));
    }

    #[test]
    fn vectors() {
        let v = Tensor::from_vec(vec![1i64, 2, 3], &[3]).unwrap();
        assert_eq!(v.matmul(&v).unwrap().shape(), &[] as &[usize]);
        assert_eq!(v.matmul(&v).unwrap().to_vec::<i64>(), vec![14]);
        let m = range(&[3, 2]);
        assert_eq!(v.matmul(&m).unwrap().shape(), &[2]);
        assert_eq!(v.matmul(&m).unwrap().to_vec::<i64>(), vec![16, 22]);
        let w = Tensor::from_vec(vec![1i64, 1], &[2]).unwrap();
        assert_eq!(m.matmul(&w).unwrap().to_vec::<i64>(), vec![1, 5, 9]);
    }

    #[test]
    fn batches_broadcast() {
        let a = range(&[2, 1, 3, 4]);
        let b = range(&[5, 4, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 5, 3, 2]);
        for i in 0..2 {
            for j in 0..5 {
                let got = c.select(0, i).unwrap().select(0, j).unwrap();
                let ai = a.select(0, i).unwrap().select(0, 0).unwrap();
                let bj = b.select(0, j).unwrap();
                assert_eq!(got.to_vec::<i64>(), reference(&ai, &bj), "batch {i},{j}");
            }
        }
        // A vector against a batch of matrices.
        let v = Tensor::from_vec(vec![1i64, 0, 0, 0], &[4]).unwrap();
        assert_eq!(v.matmul(&b).unwrap().shape(), &[5, 2]);
    }

    #[test]
    fn dtypes_and_errors() {
        let f = Tensor::from_vec(vec![0.5f32, 1.5], &[1, 2]).unwrap();
        let i = Tensor::from_vec(vec![2i64, 4], &[2, 1]).unwrap();
        let c = f.matmul(&i).unwrap();
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.to_vec::<f32>(), vec![7.0]);
        assert!(f.matmul(&f).is_err(), "inner dimensions differ");
        assert!(f.matmul(&Tensor::scalar(2.0f32)).is_err());
        let b = Tensor::from_vec(vec![true], &[1, 1]).unwrap();
        assert!(b.matmul(&b).is_err());
        // Shared storage is fine.
        let sq = range(&[2, 2]);
        assert_eq!(sq.matmul(&sq).unwrap().to_vec::<i64>(), vec![2, 3, 6, 11]);
        // Empty inner dimension gives zeros.
        let e = Tensor::zeros(&[2, 0], DType::F32);
        let z = e.matmul(&Tensor::zeros(&[0, 3], DType::F32)).unwrap();
        assert_eq!(z.to_vec::<f32>(), vec![0.0; 6]);
    }
}

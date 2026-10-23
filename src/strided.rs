//! Walking strided memory.
//!
//! Every elementwise operation visits the elements of one or more same-shape
//! views in row-major order. Instead of computing a storage position per
//! element, the walk is split into *rows*: runs along the innermost axis,
//! each described by a start position, a length and a stride. Callers handle
//! a whole row at once, so a stride-1 row is a plain slice loop the compiler
//! can vectorize, and a stride-0 row is one repeated value.
//!
//! Before walking, neighbouring axes that are laid out back-to-back in every
//! operand are merged. A contiguous tensor of any shape therefore becomes a
//! single row, whatever its dimensionality.

use std::array;

/// Merges neighbouring axes that every operand can walk as one, and drops
/// length-1 axes (they never move). Axes `i` and `i + 1` merge when, for each
/// operand, `stride[i] == stride[i + 1] * shape[i + 1]`.
///
/// Always returns at least one axis, so callers have an innermost axis to walk.
fn coalesce<const N: usize>(
    shape: &[usize],
    strides: [&[isize]; N],
) -> (Vec<usize>, [Vec<isize>; N]) {
    let mut out_shape: Vec<usize> = Vec::with_capacity(shape.len());
    let mut out_strides: [Vec<isize>; N] = array::from_fn(|_| Vec::with_capacity(shape.len()));
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 1 {
            continue;
        }
        if let Some(outer) = out_shape.last_mut() {
            let mergeable =
                (0..N).all(|k| out_strides[k].last() == Some(&(strides[k][axis] * dim as isize)));
            if mergeable {
                *outer *= dim;
                for k in 0..N {
                    *out_strides[k].last_mut().unwrap() = strides[k][axis];
                }
                continue;
            }
        }
        out_shape.push(dim);
        for k in 0..N {
            out_strides[k].push(strides[k][axis]);
        }
    }
    if out_shape.is_empty() {
        // A 0-d tensor, or every axis has length 1: a single element.
        out_shape.push(1);
        for stride in &mut out_strides {
            stride.push(0);
        }
    }
    (out_shape, out_strides)
}

/// Visits every row of `N` same-shape views in row-major order.
///
/// `f(starts, len, inner_strides)` is called once per row with each
/// operand's storage position for the row's first element. Element `i` of the
/// row lives at `starts[k] + i * inner_strides[k]` (see [`positions`]).
pub(crate) fn for_each_row<const N: usize>(
    shape: &[usize],
    strides: [&[isize]; N],
    offsets: [usize; N],
    mut f: impl FnMut([usize; N], usize, [isize; N]),
) {
    if shape.contains(&0) {
        return;
    }
    // Most calls are contiguous, a broadcast scalar, or a reversed flat
    // view. Recognize a single row before allocating coalesced layout vectors.
    let last = shape.iter().rposition(|&dim| dim != 1);
    let inner: [isize; N] = array::from_fn(|k| last.map_or(0, |axis| strides[k][axis]));
    let single_row = (0..N).all(|k| {
        let mut expected = inner[k];
        for axis in (0..shape.len()).rev() {
            if shape[axis] == 1 {
                continue;
            }
            if strides[k][axis] != expected {
                return false;
            }
            expected *= shape[axis] as isize;
        }
        true
    });
    if single_row {
        f(offsets, shape.iter().product(), inner);
        return;
    }
    let (shape, strides) = coalesce(shape, strides);
    let inner_axis = shape.len() - 1;
    let len = shape[inner_axis];
    let inner: [isize; N] = array::from_fn(|k| strides[k][inner_axis]);

    // Odometer over the outer axes, tracking each operand's row start.
    let mut index = vec![0usize; inner_axis];
    let mut pos: [isize; N] = array::from_fn(|k| offsets[k] as isize);
    loop {
        f(array::from_fn(|k| pos[k] as usize), len, inner);
        let mut axis = inner_axis;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            for k in 0..N {
                pos[k] += strides[k][axis];
            }
            if index[axis] < shape[axis] {
                break;
            }
            for k in 0..N {
                pos[k] -= strides[k][axis] * shape[axis] as isize;
            }
            index[axis] = 0;
        }
    }
}

/// Storage positions of the elements of one row.
pub(crate) fn positions(start: usize, len: usize, stride: isize) -> impl Iterator<Item = usize> {
    (0..len).map(move |i| (start as isize + i as isize * stride) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::contiguous_strides;

    /// Every (row start, len, stride) visited for one operand.
    fn rows(shape: &[usize], strides: &[isize], offset: usize) -> Vec<(usize, usize, isize)> {
        let mut out = Vec::new();
        for_each_row(shape, [strides], [offset], |[p], len, [s]| {
            out.push((p, len, s))
        });
        out
    }

    #[test]
    fn contiguous_tensors_are_one_row() {
        assert_eq!(
            rows(&[2, 3, 4], &contiguous_strides(&[2, 3, 4]), 5),
            vec![(5, 24, 1)]
        );
        assert_eq!(rows(&[], &[], 7), vec![(7, 1, 0)]);
        assert_eq!(rows(&[1, 1], &[9, 9], 0), vec![(0, 1, 0)]);
        assert!(rows(&[3, 0], &[0, 1], 0).is_empty());
    }

    #[test]
    fn transposed_rows_are_strided() {
        // [2, 3] transposed: shape [3, 2], strides [1, 3].
        assert_eq!(
            rows(&[3, 2], &[1, 3], 0),
            vec![(0, 2, 3), (1, 2, 3), (2, 2, 3)]
        );
    }

    #[test]
    fn only_axes_that_line_up_merge() {
        // t[:, :2] of a [3, 4] tensor: rows of 2 with a gap of 2 between them.
        assert_eq!(
            rows(&[3, 2], &[4, 1], 0),
            vec![(0, 2, 1), (4, 2, 1), (8, 2, 1)]
        );
        // Reversed contiguous data still merges into one row with stride -1.
        assert_eq!(rows(&[2, 3], &[-3, -1], 5), vec![(5, 6, -1)]);
    }

    #[test]
    fn operands_merge_only_when_all_line_up() {
        // a [2, 3] contiguous plus a row broadcast down axis 0 (stride 0):
        // a alone would be one row, but the broadcast operand can't merge.
        let mut seen = Vec::new();
        for_each_row(
            &[2, 3],
            [&[3, 1], &[0, 1]],
            [0, 10],
            |[a, b], len, [sa, sb]| seen.push((a, b, len, sa, sb)),
        );
        assert_eq!(seen, vec![(0, 10, 3, 1, 1), (3, 10, 3, 1, 1)]);

        // A scalar broadcast to everything merges along with a contiguous operand.
        let mut seen = Vec::new();
        for_each_row(
            &[2, 3],
            [&[3, 1], &[0, 0]],
            [0, 4],
            |[a, b], len, [sa, sb]| seen.push((a, b, len, sa, sb)),
        );
        assert_eq!(seen, vec![(0, 4, 6, 1, 0)]);
    }

    #[test]
    fn positions_follow_the_stride() {
        assert_eq!(positions(5, 3, -2).collect::<Vec<_>>(), vec![5, 3, 1]);
        assert_eq!(positions(4, 3, 0).collect::<Vec<_>>(), vec![4, 4, 4]);
    }
}

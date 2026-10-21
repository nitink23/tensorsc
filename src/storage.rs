//! The memory behind tensors.
//!
//! [`Storage`] is a flat, typed buffer: a pointer, a length and a dtype, plus
//! whatever keeps the memory alive. Usually that's a Rust `Vec` we allocated,
//! but it can also be memory owned by someone else (a NumPy array), which is
//! how tensors share memory with other libraries without copying.
//!
//! This is the only module that touches raw memory. Everything else reads and
//! writes through [`Storage::read`] and [`Storage::write`], which hand out
//! ordinary slices while holding a lock.

use std::any::Any;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::device::Device;
use crate::dtype::{DType, Element, with_dtype};
use crate::error::{Result, TensorError};

enum Owner {
    /// We allocated the buffer as a `Vec` of the storage's dtype.
    Vec { capacity: usize },
    /// Someone else owns it; dropping this releases our hold on it.
    External(#[allow(dead_code)] Box<dyn Any + Send + Sync>),
}

pub struct Storage {
    ptr: NonNull<u8>,
    /// Length in elements.
    len: usize,
    dtype: DType,
    writable: bool,
    device: Device,
    owner: Owner,
    /// Guards the elements behind `ptr` (the other fields never change).
    lock: RwLock<()>,
}

// SAFETY: the elements are only reached through `read` and `write`, which
// take `lock`, so sharing a `Storage` across threads can't cause a data race.
// External owners are required to be `Send + Sync` themselves.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.len)
            .field("dtype", &self.dtype)
            .field("writable", &self.writable)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Storage {
    pub fn from_vec<T: Element>(data: Vec<T>, device: Device) -> Self {
        let mut data = ManuallyDrop::new(data);
        Storage {
            // A Vec's pointer is never null (it's dangling-but-aligned when empty).
            ptr: NonNull::new(data.as_mut_ptr().cast()).expect("Vec pointer is non-null"),
            len: data.len(),
            dtype: T::DTYPE,
            writable: true,
            device,
            owner: Owner::Vec {
                capacity: data.capacity(),
            },
            lock: RwLock::new(()),
        }
    }

    /// Wraps memory owned by someone else. `owner` is kept until the storage
    /// is dropped and should be what keeps the memory alive.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to `len` initialized, properly aligned elements of
    ///   `dtype`, valid for as long as `owner` is alive. For `Bool`, every
    ///   byte must be 0 or 1.
    /// - Nothing else may access that memory while a guard from [`read`] or
    ///   [`write`] is alive (other than reading, while only read guards exist),
    ///   and if `writable` is false nothing may write to it at all.
    ///
    /// [`read`]: Storage::read
    /// [`write`]: Storage::write
    pub unsafe fn from_external(
        ptr: NonNull<u8>,
        len: usize,
        dtype: DType,
        writable: bool,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Self {
        debug_assert_eq!(ptr.as_ptr().align_offset(dtype.itemsize()), 0);
        Storage {
            ptr,
            len,
            dtype,
            writable,
            device: Device::Cpu,
            owner: Owner::External(owner),
            lock: RwLock::new(()),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// Address of the first element, for handing memory to other libraries.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Shared access to the elements. Panics if `T` isn't this storage's dtype
    /// (that's a bug in tensorsc, not a user error).
    pub fn read<T: Element>(&self) -> ReadGuard<'_, T> {
        self.check_type::<T>();
        let lock = self.lock.read().expect("storage lock poisoned");
        // SAFETY: the type matches, the pointer is valid for `len` elements
        // (see the constructors), and the read lock keeps writers out.
        let data = unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast::<T>(), self.len) };
        ReadGuard { _lock: lock, data }
    }

    /// Exclusive access to the elements, or an error for read-only memory.
    pub fn write<T: Element>(&self) -> Result<WriteGuard<'_, T>> {
        self.check_type::<T>();
        if !self.writable {
            return Err(TensorError::ReadOnly);
        }
        let lock = self.lock.write().expect("storage lock poisoned");
        // SAFETY: as in `read`, and the write lock makes this the only access.
        let data = unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<T>(), self.len) };
        Ok(WriteGuard { _lock: lock, data })
    }

    fn check_type<T: Element>(&self) {
        assert_eq!(
            T::DTYPE,
            self.dtype,
            "storage of {} accessed as {}",
            self.dtype,
            T::DTYPE
        );
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Owner::Vec { capacity } = self.owner {
            with_dtype!(self.dtype, T => {
                // SAFETY: these are exactly the parts `from_vec` took apart.
                drop(unsafe { Vec::from_raw_parts(self.ptr.as_ptr().cast::<T>(), self.len, capacity) });
            });
        }
        // External owners release their memory when `owner` is dropped.
    }
}

pub struct ReadGuard<'a, T> {
    _lock: RwLockReadGuard<'a, ()>,
    data: &'a [T],
}

impl<T> Deref for ReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.data
    }
}

pub struct WriteGuard<'a, T> {
    _lock: RwLockWriteGuard<'a, ()>,
    data: &'a mut [T],
}

impl<T> Deref for WriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.data
    }
}

impl<T> DerefMut for WriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn vec_round_trip() {
        let s = Storage::from_vec(vec![1i64, 2, 3], Device::Cpu);
        assert_eq!(s.dtype(), DType::I64);
        s.write::<i64>().unwrap()[1] = 20;
        assert_eq!(&*s.read::<i64>(), &[1, 20, 3]);
        let empty = Storage::from_vec(Vec::<f64>::new(), Device::Cpu);
        assert!(empty.read::<f64>().is_empty());
    }

    #[test]
    #[should_panic(expected = "storage of float32 accessed as int64")]
    fn wrong_type_panics() {
        let s = Storage::from_vec(vec![1.0f32], Device::Cpu);
        let _ = s.read::<i64>();
    }

    #[test]
    fn external_memory_is_released_with_the_storage() {
        struct Owner {
            data: Vec<f32>,
            dropped: Arc<AtomicBool>,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let mut owner = Box::new(Owner {
            data: vec![1.0, 2.0],
            dropped: dropped.clone(),
        });
        let ptr = NonNull::new(owner.data.as_mut_ptr().cast()).unwrap();
        // SAFETY: the Vec's buffer lives as long as `owner`, which the storage keeps.
        let s = unsafe { Storage::from_external(ptr, 2, DType::F32, false, owner) };
        assert_eq!(&*s.read::<f32>(), &[1.0, 2.0]);
        assert!(matches!(s.write::<f32>(), Err(TensorError::ReadOnly)));
        assert!(!dropped.load(Ordering::SeqCst));
        drop(s);
        assert!(dropped.load(Ordering::SeqCst));
    }
}

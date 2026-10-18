use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::device::Device;

/// A flat buffer of elements on a device.
///
/// Storage knows nothing about shape. Tensors hold it behind an `Arc`, so any
/// number of views can share one buffer, and the `RwLock` lets a write through
/// one view be seen by all the others.
#[derive(Debug)]
pub struct Storage {
    data: RwLock<Vec<f32>>,
    device: Device,
}

impl Storage {
    pub fn new(data: Vec<f32>, device: Device) -> Self {
        Storage {
            data: RwLock::new(data),
            device,
        }
    }

    pub fn len(&self) -> usize {
        self.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn read(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data.read().expect("storage lock poisoned")
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.data.write().expect("storage lock poisoned")
    }
}

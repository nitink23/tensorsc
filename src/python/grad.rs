//! Python grad-mode controls: `ts.no_grad()`, `ts.enable_grad()`, and friends.

use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use crate::autograd::{is_grad_enabled, set_grad_enabled};

/// Turns gradient recording off (`no_grad`) or on (`enable_grad`) while
/// active. Works as a context manager (`with ts.no_grad():`) and as a
/// decorator (`@ts.no_grad()`), like PyTorch's.
#[pyclass(name = "grad_mode", module = "tensorsc", frozen)]
pub(crate) struct GradMode {
    enabled: bool,
    /// Settings to restore, one per active `with` (so it can be nested).
    previous: Mutex<Vec<bool>>,
}

impl GradMode {
    pub(crate) fn new(enabled: bool) -> Self {
        GradMode {
            enabled,
            previous: Mutex::new(Vec::new()),
        }
    }
}

#[pymethods]
impl GradMode {
    fn __enter__(&self) {
        let previous = set_grad_enabled(self.enabled);
        self.previous.lock().expect("lock poisoned").push(previous);
    }

    #[pyo3(signature = (*_exc))]
    fn __exit__(&self, _exc: &Bound<'_, PyTuple>) -> bool {
        if let Some(previous) = self.previous.lock().expect("lock poisoned").pop() {
            set_grad_enabled(previous);
        }
        false
    }

    /// Used as a decorator: runs `func` with this grad mode.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCFunction>> {
        let (enabled, func) = (self.enabled, func.unbind());
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>,
                  kwargs: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let previous = set_grad_enabled(enabled);
                let result = func.bind(args.py()).call(args, kwargs);
                set_grad_enabled(previous);
                result.map(Bound::unbind)
            },
        )
    }

    fn __repr__(&self) -> &'static str {
        if self.enabled {
            "tensorsc.enable_grad()"
        } else {
            "tensorsc.no_grad()"
        }
    }
}

/// Disables gradient recording inside `with ts.no_grad():` (or a decorated function).
#[pyfunction]
pub(crate) fn no_grad() -> GradMode {
    GradMode::new(false)
}

/// Re-enables gradient recording inside a `no_grad` block.
#[pyfunction]
pub(crate) fn enable_grad() -> GradMode {
    GradMode::new(true)
}

#[pyfunction(name = "is_grad_enabled")]
pub(crate) fn py_is_grad_enabled() -> bool {
    is_grad_enabled()
}

#[pyfunction(name = "set_grad_enabled")]
pub(crate) fn py_set_grad_enabled(mode: bool) {
    set_grad_enabled(mode);
}

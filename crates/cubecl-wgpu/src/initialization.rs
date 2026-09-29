use std::{any::TypeId, collections::HashMap, sync::Mutex};

use cubecl_common::device::DeviceId;

use crate::{WgpuBackend, WgpuCompiler, WgpuDevice, WgpuServer};

/// A recoverable failure while acquiring or registering a wgpu runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WgpuInitError {
    /// A runtime option or environment default is invalid.
    #[error("Invalid wgpu configuration: {message}")]
    InvalidConfiguration {
        /// The invalid option and why it was rejected.
        message: String,
    },
    /// Selection options cannot be combined with an existing setup.
    #[error("Conflicting wgpu options: {message}")]
    ConflictingOptions {
        /// The conflicting options.
        message: String,
    },
    /// The requested API is not compiled for this platform.
    #[error("Graphics API {api:?} is unavailable in this build")]
    UnsupportedGraphicsApi {
        /// The requested API.
        api: WgpuBackend,
    },
    /// No adapter matches the requested selector.
    #[error("No adapter available for {device:?}: {message}")]
    AdapterUnavailable {
        /// The requested selector, including its graphics API.
        device: WgpuDevice,
        /// Details from adapter selection.
        message: String,
    },
    /// The adapter refused to create a device.
    #[error("Unable to request a wgpu device: {message}")]
    RequestDevice {
        /// Details from the graphics API.
        message: String,
    },
    /// An existing setup cannot be used by the runtime.
    #[error("Incompatible wgpu setup: {message}")]
    IncompatibleSetup {
        /// The unmet requirement.
        message: String,
    },
    /// Another initializer owns this selector; await it before using the device.
    #[error("Initialization is already in progress for {device:?}")]
    InitializationInProgress {
        /// The device being initialized.
        device: WgpuDevice,
    },
    /// The selector already has a runtime. Clone the original device to reuse it.
    #[error("{device:?} is already initialized; reuse its device instead of configuring it again")]
    AlreadyInitialized {
        /// The initialized selector.
        device: WgpuDevice,
    },
    /// These upstream device handles already belong to a runtime.
    #[error(
        "This wgpu device is already registered; clone the original runtime device to reuse it"
    )]
    SetupAlreadyRegistered,
    /// A runtime could not be registered.
    #[error("Unable to register wgpu runtime: {message}")]
    Registration {
        /// Registration failure details.
        message: String,
    },
}

type Key = (TypeId, DeviceId);

struct Entry {
    ready: bool,
    device: Option<(wgpu::Instance, wgpu::Device)>,
}

// Shared by lazy, explicit, and legacy initialization. Never locked while acquiring a GPU
// or inserting a server into the service registry (which can call lazy initialization).
static REGISTRATIONS: Mutex<Option<HashMap<Key, Entry>>> = Mutex::new(None);

pub(crate) struct Reservation {
    key: Key,
    committed: bool,
}

impl Reservation {
    pub(crate) fn acquire<C: WgpuCompiler>(device: &WgpuDevice) -> Result<Self, WgpuInitError> {
        use cubecl_common::device::Device;
        Self::reserve((TypeId::of::<WgpuServer<C>>(), device.to_id()), device)
    }

    fn reserve(key: Key, device: &WgpuDevice) -> Result<Self, WgpuInitError> {
        let mut guard = REGISTRATIONS.lock().unwrap();
        let entries = guard.get_or_insert_with(HashMap::new);
        if let Some(entry) = entries.get(&key) {
            return Err(if entry.ready {
                WgpuInitError::AlreadyInitialized {
                    device: device.clone(),
                }
            } else {
                WgpuInitError::InitializationInProgress {
                    device: device.clone(),
                }
            });
        }
        entries.insert(
            key,
            Entry {
                ready: false,
                device: None,
            },
        );
        Ok(Self {
            key,
            committed: false,
        })
    }

    pub(crate) fn attach(&self, setup: &crate::WgpuSetup) -> Result<(), WgpuInitError> {
        let mut guard = REGISTRATIONS.lock().unwrap();
        let entries = guard.as_mut().unwrap();
        if entries.values().any(|entry| {
            entry.device.as_ref().is_some_and(|(instance, device)| {
                instance == &setup.instance && device == &setup.device
            })
        }) {
            return Err(WgpuInitError::SetupAlreadyRegistered);
        }
        // wgpu-core device IDs are scoped to an instance: comparing devices alone can
        // mistake the first devices of two independent instances for the same device.
        entries.get_mut(&self.key).unwrap().device =
            Some((setup.instance.clone(), setup.device.clone()));
        Ok(())
    }

    pub(crate) fn commit(mut self) {
        REGISTRATIONS
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .get_mut(&self.key)
            .unwrap()
            .ready = true;
        self.committed = true;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.committed {
            REGISTRATIONS
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simultaneous_initializers_have_one_winner() {
        use std::sync::{Arc, Barrier};
        struct ConcurrentService;
        let barrier = Arc::new(Barrier::new(2));
        let key = (TypeId::of::<ConcurrentService>(), DeviceId::new(0, 0));
        let workers = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let result = Reservation::reserve(key, &WgpuDevice::default());
                    // Keep the winner's reservation alive while the other thread attempts it.
                    barrier.wait();
                    result.is_ok()
                })
            })
            .collect::<Vec<_>>();
        let winners = workers
            .into_iter()
            .map(|worker| usize::from(worker.join().unwrap()))
            .sum::<usize>();
        assert_eq!(winners, 1);
        // Neither worker committed, so the selector can be retried.
        assert!(Reservation::reserve(key, &WgpuDevice::default()).is_ok());
    }

    #[test]
    fn cancellation_releases_a_reservation_and_commit_prevents_reconfiguration() {
        struct TestService;
        let device = WgpuDevice::default();
        let key = (TypeId::of::<TestService>(), DeviceId::new(0, 0));
        let pending = Reservation::reserve(key, &device).unwrap();
        assert!(matches!(
            Reservation::reserve(key, &device),
            Err(WgpuInitError::InitializationInProgress { .. })
        ));
        drop(pending);
        Reservation::reserve(key, &device).unwrap().commit();
        assert!(matches!(
            Reservation::reserve(key, &device),
            Err(WgpuInitError::AlreadyInitialized { .. })
        ));
    }
}

use crate::compute::uninit_vec;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::config::memory::CudaAllocator;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;
use std::collections::HashMap;

/// Records the actual allocation path, which may differ from the configured
/// [`CudaAllocator`] when asynchronous allocation falls back to synchronous.
#[derive(Debug, Clone, Copy)]
enum AllocationKind {
    Async,
    Sync,
    /// A zero-size buffer has no backing allocation.
    Empty,
}

/// Buffer storage for NVIDIA GPUs.
///
/// This struct manages memory resources for CUDA kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, (cudarc::driver::sys::CUdeviceptr, AllocationKind, u64)>,
    deallocations: Vec<StorageId>,
    /// The bytes `memory` holds.
    allocated: u64,
    ptr_bindings: PtrBindings,
    stream: cudarc::driver::sys::CUstream,
    mem_alignment: usize,
    allocator: CudaAllocator,
}

/// A GPU memory resource allocated for CUDA using [`GpuStorage`].
#[derive(Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: u64,
    /// The CUDA binding pointer.
    pub binding: *mut std::ffi::c_void,
    /// The size of the resource.
    pub size: u64,
}

impl GpuResource {
    /// Creates a new [`GpuResource`].
    pub fn new(ptr: u64, binding: *mut std::ffi::c_void, size: u64) -> Self {
        Self { ptr, binding, size }
    }
}

impl GpuStorage {
    /// Creates storage using the selected allocator and CUDA stream.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    /// * `stream` - The stream used for asynchronous allocation and deallocation.
    /// * `allocator` - How CUDA obtains backing memory for `CubeCL`'s pools.
    pub fn new(
        mem_alignment: usize,
        stream: cudarc::driver::sys::CUstream,
        allocator: CudaAllocator,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            allocated: 0,
            ptr_bindings: PtrBindings::new(),
            mem_alignment,
            stream,
            allocator,
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    fn perform_deallocations(&mut self) {
        self.deallocations
            .drain(..)
            .filter_map(|id| self.memory.remove(&id))
            // SAFETY: Nonempty pointers are owned by this storage and have not
            // been freed. Empty entries make no driver call.
            .for_each(|(ptr, kind, size)| unsafe {
                self.allocated -= size;
                let result = match kind {
                    AllocationKind::Empty => Ok(()),
                    AllocationKind::Sync => cudarc::driver::result::free_sync(ptr),
                    AllocationKind::Async => cudarc::driver::result::free_async(ptr, self.stream),
                };
                if let Err(e) = result {
                    eprintln!("CUDA free error: {}", e);
                }
            });
    }
}

// SAFETY: `GpuResource` contains CUDA device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}
// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw CUDA pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStorage").finish()
    }
}

/// Manages active CUDA buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            // SAFETY: `CUdeviceptr` is a `u64`, valid for any bit pattern. All slots are
            // written via `register()` before being read, so uninitialized values are never observed.
            slots: unsafe { uninit_vec(crate::device::CUDA_MAX_BINDINGS as usize) },
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The CUDA device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    /// The resource for the foreign allocation `[ptr, ptr + size)`: the
    /// caller's address carried as-is — no allocation, no copy, no
    /// ownership. The pointer takes a slot in the bindings ring exactly as
    /// [`ComputeStorage::get`] gives one to a pooled address, so the launch
    /// path's param packing (a pointer INTO the ring) is unchanged.
    ///
    /// # Safety
    ///
    /// As the trait method: `ptr` must be a live device allocation on this
    /// storage's device, valid while the returned resource can reach work.
    unsafe fn get_external(&mut self, ptr: u64, size: u64) -> Result<Self::Resource, IoError> {
        let binding = self.ptr_bindings.register(ptr);
        Ok(GpuResource::new(
            *binding,
            binding as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        ))
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        let (ptr, _, _) =
            self.memory
                .get(&handle.id)
                .ok_or_else(|| IoError::StorageHandleNotFound {
                    reason: format!("{} in the CUDA gpu storage", handle.id).into(),
                    backtrace: BackTrace::capture(),
                })?;

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        Ok(GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        ))
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // CubeCL pools these allocations itself. Sync avoids CUDA's additional
        // pool; async preserves stream ordering and the existing sync fallback.
        let allocation = if size == 0 {
            // cuMemAlloc rejects zero bytes. Empty buffers still need a storage
            // entry for binding resolution, but no CUDA allocation or free.
            Ok((0, AllocationKind::Empty))
        } else {
            // SAFETY: The context and stream are valid. Successful allocations
            // remain owned by `self.memory` and are freed by their actual kind.
            unsafe {
                match self.allocator {
                    CudaAllocator::Sync => cudarc::driver::result::malloc_sync(size as usize)
                        .map(|ptr| (ptr, AllocationKind::Sync)),
                    CudaAllocator::Async => {
                        cudarc::driver::result::malloc_async(self.stream, size as usize)
                            .map(|ptr| (ptr, AllocationKind::Async))
                            .or_else(|_| {
                                cudarc::driver::result::malloc_sync(size as usize)
                                    .map(|ptr| (ptr, AllocationKind::Sync))
                            })
                    }
                }
            }
        };
        let (ptr, kind) = match allocation {
            Ok(allocation) => allocation,
            // Not `BufferTooBig`: that variant means the allocation can
            // never fit, and `Command::reserve` skips its reclaim-and-retry
            // when it sees it. A full device is a moment, not a verdict.
            Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                return Err(IoError::OutOfMemory {
                    size,
                    backtrace: BackTrace::capture(),
                });
            }
            Err(other) => {
                return Err(IoError::Unknown {
                    description: format!("CUDA allocation error: {other}"),
                    backtrace: BackTrace::capture(),
                });
            }
        };

        self.memory.insert(id, (ptr, kind, size));
        self.allocated += size;
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn flush(&mut self) {
        self.perform_deallocations();
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::DevicePtr as _;

    /// A foreign allocation rides `get_external` with its exact address and
    /// extent, and the storage's own accounting never sees it: the wrap is
    /// a dereference, not an allocation.
    #[test_log::test]
    fn a_foreign_allocation_resolves_untouched_and_unaccounted() {
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();

        // The foreign allocation: owned by cudarc, not by this storage.
        let foreign: cudarc::driver::CudaSlice<u32> = stream.alloc_zeros(64).unwrap();
        let foreign_ptr = foreign.device_ptr(&stream).0 as u64;

        let mut storage = GpuStorage::new(512, stream.cu_stream(), CudaAllocator::Sync);
        let before = storage.bytes_allocated();

        // SAFETY: `foreign_ptr` is the live allocation `foreign` owns, kept
        // alive past this call.
        let resource =
            unsafe { storage.get_external(foreign_ptr, 64 * 4) }.expect("the wrap resolves");
        assert_eq!(resource.ptr, foreign_ptr, "the address is the caller's, as-is");
        assert_eq!(resource.size, 64 * 4);
        assert!(
            !resource.binding.is_null(),
            "the launch path's param slot exists"
        );
        assert_eq!(
            storage.bytes_allocated(),
            before,
            "the pools' accounting never saw the foreign memory"
        );

        // A second wrap of a different address gets its own ring slot.
        let foreign2: cudarc::driver::CudaSlice<u32> = stream.alloc_zeros(8).unwrap();
        let foreign2_ptr = foreign2.device_ptr(&stream).0 as u64;
        // SAFETY: as above.
        let resource2 = unsafe { storage.get_external(foreign2_ptr, 8 * 4) }.unwrap();
        assert_eq!(resource2.ptr, foreign2_ptr);
        assert_ne!(resource.binding, resource2.binding, "distinct param slots");
    }

    #[test]
    fn zero_size_allocations() {
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();

        for allocator in [CudaAllocator::Sync, CudaAllocator::Async] {
            let mut storage = GpuStorage::new(512, stream.cu_stream(), allocator);
            for size in [0, 16] {
                let handle = storage.alloc(size).unwrap();
                let resource = storage.get(&handle).unwrap();
                assert_eq!(handle.size(), size);
                assert_eq!(resource.size, size);
                assert_eq!(resource.ptr == 0, size == 0);
                assert_eq!(storage.bytes_allocated(), size);

                storage.dealloc(handle.id);
                storage.flush();
                stream.synchronize().unwrap();
                assert_eq!(storage.bytes_allocated(), 0);
            }
        }
    }
}

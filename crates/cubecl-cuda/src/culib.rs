//! Whether the CUDA driver (and NCCL) libraries are loadable — the question
//! cudarc's `sys::is_culib_present` answers, asked here through `libloading`
//! directly instead of through cudarc.
//!
//! Why not cudarc's own function: it exists only under cudarc's
//! `dynamic-loading` mode, and that mode is not part of the feature union
//! this crate can rely on. cudarc's `fallback-dynamic-loading` (the fork's
//! own spec) asks build.rs to enable it — but ONLY when no other linking
//! mode is active, and a consumer graph that also pulls cudarc with
//! `dynamic-linking` (candle does, under its `cuda` feature) suppresses the
//! fallback: the cfg flips off, the function disappears, and this crate
//! stops compiling. Linking modes are a property of the whole dependency
//! graph, so the guard must not depend on which one won.
//!
//! Under any non-dynamic mode the linker settled the question at build time
//! and the answer is trivially yes; asking it again here stays correct in
//! every mode and costs one `dlopen` attempt on the rare paths that ask.

/// The CUDA driver library is loadable (or the platform links it statically,
/// in which case it is present by construction).
pub(crate) fn cuda_present() -> bool {
    #[cfg(target_os = "windows")]
    let names = ["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    let names = ["libcuda.so.1", "libcuda.so"];

    names.iter().copied().any(|name| {
        // SAFETY: opening a library handle; it is dropped at the end of this
        // call, which unloads nothing the process is not otherwise using.
        unsafe { libloading::Library::new(name).is_ok() }
    })
}

/// The NCCL library is loadable (or linked statically).
#[cfg(feature = "nccl")]
pub(crate) fn nccl_present() -> bool {
    #[cfg(target_os = "windows")]
    let names = ["nccl.dll"];
    #[cfg(not(target_os = "windows"))]
    let names = ["libnccl.so.2", "libnccl.so"];

    names.iter().copied().any(|name| {
        // SAFETY: as `cuda_present`.
        unsafe { libloading::Library::new(name).is_ok() }
    })
}

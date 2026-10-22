//! Inline-asm ordering semantics: side-effecting asm between two fence sites
//! must carry the `"memory"` clobber in the emitted C++, so the host compiler
//! cannot move it across a fence, and the ordering must hold on the device.

#[cfg(test)]
mod tests {
    use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler};
    use crate::CudaRuntime;
    use cubecl_core::ir::nvidia::SmArch;
    use cubecl_core::prelude::*;
    // The expansion refers to `::cubecl`.
    use cubecl_core as cubecl;
    use cubecl_server::compiler::{ArtifactId, CompilationRecording};
    use cubecl_server::kernel::CompiledKernel;
    use cubecl_server::runtime::Runtime;

    /// The value the probe's asm writes.
    const MARKER: u32 = 0x1357_9bdf;

    /// Side-effecting inline asm between two storage fences, with the default
    /// memory semantics: the asm writes memory the constraints do not name,
    /// so the emitted C++ must clobber `"memory"` and the write must reach a
    /// reader that acquires after the fence.
    #[cube(launch)]
    fn asm_ordering_probe(buf: &mut [u32], flag: &mut [u32]) {
        if CUBE_POS == 0usize && UNIT_POS == 0u32 {
            let ptr = unsafe { buf.as_mut_ptr_unchecked() };
            let v = MARKER;
            gpu_asm!(
                "st.global.b32 [{}], {v};",
                addr = mem_out(_) ptr, v = in(_) v,
                options(explicit_mem)
            );
        }
        // Release: what the asm wrote is published to the device scope.
        sync_storage();
        if CUBE_POS == 1usize && UNIT_POS == 0u32 {
            let got = buf[0];
            flag[0] = select(got == MARKER, 1u32, 0u32);
        }
        // Acquire on the same scope, so the reader's fence site is real.
        sync_storage();
    }

    /// The same asm opted out of the memory semantics: `nomem` promises the
    /// asm touches no memory the compiler tracks, and no clobber may be
    /// emitted for it.
    #[cube(launch)]
    fn asm_nomem_probe(dst: &mut [u32], src: &[u32]) {
        let v = src[0];
        let out: u32;
        gpu_asm!(
            "mov.b32 {r}, {v};",
            r = out(_) out, v = in(_) v,
            options(nomem, pure)
        );
        dst[1] = out;
    }

    fn probe_kernel_source(backend: CudaBackend) -> String {
        let client = CudaRuntime::client(&Default::default());
        let (properties, target_properties) =
            (client.properties_shared(), client.target_properties_shared());
        drop(client);
        let settings = KernelSettings::new(
            *CubeDim::new_1d(32),
            ExecutionMode::Checked,
            AddressType::U32,
        )
        .kernel_name("asm_ordering_probe");
        let kernel = asm_ordering_probe::AsmOrderingProbe::new(
            settings,
            properties,
            target_properties,
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let mut recording = CompilationRecording::new(&id.kernel);
        let (arch_major, arch_minor) = {
            use cudarc::driver::sys::CUdevice_attribute;
            // SAFETY: a valid device handle from `cuDeviceGet`.
            unsafe {
                let device = cudarc::driver::result::device::get(0).unwrap();
                let major = cudarc::driver::result::device::get_attribute(
                    device,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .unwrap();
                let minor = cudarc::driver::result::device::get_attribute(
                    device,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
                .unwrap();
                (major as u32, minor as u32)
            }
        };
        let version = arch_major * 10 + arch_minor;
        let options = CudaCompilationOptions {
            cpp: Default::default(),
            arch: Some(SmArch::new(version, false)),
            ptx_version: None,
        };
        let compiled = CompiledKernel::lower(&kernel, &mut recording, |definition| {
            CompiledKernel::compile(
                &kernel,
                definition,
                &mut CudaCompiler::new(backend),
                &options,
            )
        })
        .expect("the probe kernel compiles");
        compiled.source
    }

    /// The side-effecting asm between the two fence sites carries the memory
    /// clobber in the emitted C++: this is what makes it an ordering barrier
    /// for the host compiler.
    #[test]
    fn side_effecting_asm_clobbers_memory_in_the_emitted_cpp() {
        let source = probe_kernel_source(CudaBackend::Cpp);
        assert!(
            source.contains("st.global.b32"),
            "the asm is in the kernel:\n{source}"
        );
        assert!(
            source.contains("asm volatile("),
            "the asm compiles to a volatile asm:\n{source}"
        );
        assert!(
            source.contains(": \"memory\""),
            "the asm's clobber list carries memory:\n{source}"
        );
        assert!(
            source.matches("__threadfence()").count() >= 2,
            "the two fence sites are real:\n{source}"
        );
    }

    /// `nomem` opts out: no clobber the asm did not ask for.
    #[test]
    fn nomem_asm_carries_no_memory_clobber() {
        let client = CudaRuntime::client(&Default::default());
        let (properties, target_properties) =
            (client.properties_shared(), client.target_properties_shared());
        drop(client);
        let settings = KernelSettings::new(
            *CubeDim::new_1d(32),
            ExecutionMode::Checked,
            AddressType::U32,
        )
        .kernel_name("asm_nomem_probe");
        let kernel = asm_nomem_probe::AsmNomemProbe::new(
            settings,
            properties,
            target_properties,
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
        );
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let mut recording = CompilationRecording::new(&id.kernel);
        let options = CudaCompilationOptions {
            cpp: Default::default(),
            arch: None,
            ptx_version: None,
        };
        let compiled = CompiledKernel::lower(&kernel, &mut recording, |definition| {
            CompiledKernel::compile(
                &kernel,
                definition,
                &mut CudaCompiler::new(CudaBackend::Cpp),
                &options,
            )
        })
        .expect("the nomem probe kernel compiles");
        assert!(
            compiled.source.contains("mov.b32"),
            "the asm is in the kernel:\n{}",
            compiled.source
        );
        assert!(
            !compiled.source.contains(": \"memory\""),
            "a nomem asm must not clobber memory:\n{}",
            compiled.source
        );
    }

    /// The ordering holds on the device: the asm write before the fence is
    /// observed after it, run after run.
    #[test]
    fn the_asm_write_crosses_the_fence_on_the_device() {
        let client = CudaRuntime::client(&Default::default());
        for run in 0..1000 {
            let buf = client.empty(4 * size_of::<u32>());
            let flag = client.empty(4 * size_of::<u32>());
            asm_ordering_probe::launch(
                &client,
                CubeCount::Static(2, 1, 1),
                CubeDim::new_1d(32),
                unsafe { BufferArg::from_raw_parts(buf.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(flag.clone(), 4) },
            );
            let flags = client.read_one_unchecked(flag.clone());
            let flags = u32::from_bytes(&flags);
            assert_eq!(flags[0], 1, "run {run}: the asm write missed the fence");
        }
    }
}

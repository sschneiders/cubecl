//! The register budget, end to end on the device: a kernel asked to fit a
//! number of registers compiles into at most that many, on both backends and
//! on both load paths.

#[cfg(test)]
mod tests {
    use crate::CudaRuntime;
    use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler};
    use crate::compute::artifact::{CudaArtifact, CudaArtifactCompiler};
    use crate::compute::modules::{CudaModules, load_with_register_budget};
    use cubecl_core as cubecl;
    use cubecl_core::ir::nvidia::SmArch;
    use cubecl_core::prelude::*;
    use cubecl_cpp::cuda::arch::CudaArchitecture;
    use cubecl_server::compiler::{
        ArtifactCompiler, ArtifactId, CompilationRecording, CompilationTarget,
    };
    use cubecl_server::kernel::CompiledKernel;
    use cubecl_server::runtime::Runtime;
    use cudarc::driver::sys::{self, CUctx_st, CUfunction_attribute};
    use std::ffi::{CStr, CString};
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    /// The cap the budget tests run under: the number the deployed megakernel
    /// family asks for, two cubes of 256 units per scheduler.
    const CAP: u32 = 128;
    /// Wide enough that the compiler cannot fit the body under [`CAP`]
    /// without spilling: 192 accumulators live across an unrolled loop nest.
    const WIDTH: usize = 192;

    /// A body the compiler cannot fit under the cap without spilling: every
    /// accumulator is read again each round, so all of them stay live.
    #[cube(launch)]
    fn reg_hog(input: &[f32], output: &mut [f32], #[comptime] width: usize) {
        let mut acc = Array::<f32>::new(width);
        #[unroll]
        for i in 0..width {
            acc[i] = input[i] + i as f32;
        }
        #[unroll]
        for r in 0..8 {
            for i in 0..width {
                acc[i] = acc[i] * 1.0000001 + input[r];
            }
        }
        #[unroll]
        for i in 1..width {
            acc[0] = acc[0] + acc[i];
        }
        output[ABSOLUTE_POS as usize] = acc[0];
    }

    fn hog_kernel(
        cap: Option<u32>,
        properties: Arc<DeviceProperties>,
        target_properties: Arc<TargetProperties>,
    ) -> impl CubeKernel {
        let mut settings = KernelSettings::new(
            *CubeDim::new_1d(256),
            ExecutionMode::Checked,
            AddressType::U32,
        )
        .kernel_name("reg_hog");
        if let Some(cap) = cap {
            settings = settings.max_registers(cap);
        }
        reg_hog::RegHog::new(
            settings,
            properties,
            target_properties,
            BufferCompilationArg { inplace: None },
            BufferCompilationArg { inplace: None },
            WIDTH,
        )
    }

    /// The primary context of the device the client runs on, current in the thread.
    fn current_primary_context() -> *mut CUctx_st {
        let client = CudaRuntime::client(&Default::default());
        drop(client);
        // SAFETY: the driver is initialized (the client did it), and the primary
        // context is retained for the test's life — the same context the server
        // uses, so the module the test loads is valid where the server launches.
        unsafe {
            let device = cudarc::driver::result::device::get(0).unwrap();
            let mut context = MaybeUninit::uninit();
            sys::cuDevicePrimaryCtxRetain(context.as_mut_ptr(), device)
                .result()
                .expect("the primary context is retained");
            let context = context.assume_init();
            cudarc::driver::result::ctx::set_current(context)
                .expect("the primary context becomes current");
            context
        }
    }

    fn arch_of() -> (CudaArchitecture, SmArch) {
        // SAFETY: `device` is a valid CUDA device handle from `cuDeviceGet`.
        let (major, minor) = unsafe {
            let device = cudarc::driver::result::device::get(0).unwrap();
            let major = cudarc::driver::result::device::get_attribute(
                device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .unwrap();
            let minor = cudarc::driver::result::device::get_attribute(
                device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap();
            (major as u32, minor as u32)
        };
        let version = major * 10 + minor;
        let tensor_cores = CudaArchitecture::has_tensor_cores(version, "");
        let arch = CudaArchitecture {
            version,
            tensor_cores,
        };
        (arch, SmArch::new(version, tensor_cores))
    }

    fn options_of(arch: SmArch) -> CudaCompilationOptions {
        CudaCompilationOptions {
            cpp: Default::default(),
            arch: Some(arch),
            // The PTX version the runtime itself picks for this driver, which
            // is what a capped kernel is compiled with in production.
            ptx_version: cubecl_llvm::nvptx::ptx_version::PtxVersion::for_driver(13000),
        }
    }

    fn ptx_of(artifact: &CudaArtifact) -> String {
        // SAFETY: the PTX handed to the driver is a null-terminated C string.
        unsafe { CStr::from_ptr(artifact.ptx.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// The registers the loaded function runs with.
    fn num_regs(func: *mut sys::CUfunc_st) -> u32 {
        // SAFETY: `func` is a loaded function handle, live for the process.
        unsafe {
            cudarc::driver::result::function::get_function_attribute(
                func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
            )
            .expect("the register count is read") as u32
        }
    }

    /// Compiles `kernel` for `backend` and loads it the way the runtime does,
    /// returning the artifact (with its PTX) and the loaded function.
    fn compile_and_load(
        backend: CudaBackend,
        cap: Option<u32>,
        context: *mut CUctx_st,
    ) -> (CudaArtifact, *mut sys::CUfunc_st) {
        let (arch, sm_arch) = arch_of();
        let client = CudaRuntime::client(&Default::default());
        let properties = client.properties_shared();
        drop(client);
        let compiler =
            CudaArtifactCompiler::new((*properties).clone(), options_of(sm_arch), arch, context);
        let client = CudaRuntime::client(&Default::default());
        let (properties, target_properties) = (
            client.properties_shared(),
            client.target_properties_shared(),
        );
        drop(client);
        let kernel = hog_kernel(cap, properties, target_properties);
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let mut recording = CompilationRecording::new(&id.kernel);
        let lowered = CompiledKernel::lower(&kernel, &mut recording, |definition| {
            CompiledKernel::compile(
                &kernel,
                definition,
                &mut CudaCompiler::new(backend),
                &compiler.options(),
            )
        })
        .expect("the probe kernel compiles");
        let artifact = compiler
            .finalize(&id, lowered)
            .expect("the artifact finalizes");
        let mut modules = CudaModules::new(compiler, backend);
        let loaded = modules.load(&id, &artifact).expect("the artifact loads");
        (artifact, loaded.func)
    }

    /// Uncapped, the body wants far more registers than the cap: the probe
    /// must be asking for something the cap actually changes.
    #[test]
    fn uncapped_the_body_exceeds_the_cap() {
        let context = current_primary_context();
        let (_, func) = compile_and_load(CudaBackend::Cpp, None, context);
        let regs = num_regs(func);
        assert!(
            regs > CAP,
            "{regs} registers without a cap, the cap is {CAP}"
        );
    }

    /// The C++ backend capped: the module load JITs the PTX with the budget,
    /// and the function comes back within it.
    #[test]
    fn the_cpp_backend_fits_the_budget_on_the_module_path() {
        let context = current_primary_context();
        let (artifact, func) = compile_and_load(CudaBackend::Cpp, Some(CAP), context);
        // The route switch: no machine code from the link step, which would
        // ignore the budget.
        assert!(
            artifact.cubin.is_empty(),
            "a capped kernel carries no link-step machine code"
        );
        assert!(num_regs(func) <= CAP, "the cap is {}", CAP);
    }

    /// The LLVM backend capped: the PTX declares the budget itself and the
    /// module load enforces it again.
    #[test]
    fn the_llvm_backend_declares_and_fits_the_budget() {
        let context = current_primary_context();
        let (artifact, func) = compile_and_load(CudaBackend::Llvm, Some(CAP), context);
        let ptx = ptx_of(&artifact);
        assert!(
            ptx.contains(&format!(".maxnreg {CAP}")),
            "the PTX declares the cap:\n{ptx}"
        );
        assert!(num_regs(func) <= CAP, "the cap is {}", CAP);
    }

    /// The budget declared in the PTX binds the link step too: the same
    /// assembly the link JIT produces stays within the cap, so a capped kernel
    /// works on both load paths.
    #[test]
    fn the_budget_binds_the_link_path() {
        let context = current_primary_context();
        let (arch, _) = arch_of();
        let (artifact, _) = compile_and_load(CudaBackend::Llvm, Some(CAP), context);
        let client = CudaRuntime::client(&Default::default());
        let properties = client.properties_shared();
        drop(client);
        let compiler = CudaArtifactCompiler::new(
            (*properties).clone(),
            options_of(SmArch::new(arch.version, arch.tensor_cores)),
            arch,
            context,
        );
        // `assemble` makes the device's context current itself.
        let cubin = compiler
            .assemble(&artifact.ptx)
            .expect("the capped PTX assembles through the link step");
        // SAFETY: machine code this driver assembled for this device.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(cubin.as_ptr() as *const _)
                .expect("the link-step machine code loads");
            cudarc::driver::result::module::get_function(
                module,
                CString::new(artifact.entrypoint_name.clone()).unwrap(),
            )
            .expect("the entry point is found")
        };
        assert!(num_regs(func) <= CAP, "the cap is {}", CAP);
    }

    /// The budget held on the module path, in isolation of everything else:
    /// the same PTX, loaded with and without the option, must come back
    /// different — this is the load-path guarantee the link step cannot give.
    #[test]
    fn the_module_jit_honors_the_budget_option() {
        let context = current_primary_context();
        let (_, uncapped) = compile_and_load(CudaBackend::Cpp, None, context);
        let regs_uncapped = num_regs(uncapped);

        let (artifact, _) = compile_and_load(CudaBackend::Cpp, Some(CAP), context);
        // SAFETY: the context is current, and `ptx` the null-terminated PTX the
        // artifact carries.
        let module = unsafe { load_with_register_budget(&artifact.ptx, CAP) }
            .expect("the PTX loads with the budget");
        // SAFETY: `module` is a loaded module.
        let func = unsafe {
            cudarc::driver::result::module::get_function(
                module,
                CString::new(artifact.entrypoint_name.clone()).unwrap(),
            )
            .expect("the entry point is found")
        };
        let regs_capped = num_regs(func);
        assert!(regs_capped <= CAP, "the cap is {}", CAP);
        assert!(
            regs_capped < regs_uncapped,
            "the option changed the allocation: {regs_capped} capped against {regs_uncapped} free"
        );
    }
}

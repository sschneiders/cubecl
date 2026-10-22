//! A mutable local initialized from a comptime value gets a real memory slot:
//! the regression for the verifier failure "Expected operand type cube.ptr,
//! but found cube.index".

#[cfg(test)]
mod tests {
    use crate::CudaRuntime;
    use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler};
    use cubecl_core::ir::nvidia::SmArch;
    use cubecl_core::prelude::*;
    // The expansion refers to `::cubecl`.
    use cubecl_core as cubecl;
    use cubecl_server::compiler::{ArtifactId, CompilationRecording};
    use cubecl_server::kernel::CompiledKernel;
    use cubecl_server::runtime::Runtime;

    /// The shape that broke: a mutable local whose initial value is a comptime
    /// parameter of native type, then mutated inside a `while`. The expansion
    /// used to keep the local as a plain Rust value, so the first assignment
    /// emitted `memory.store` with the constant itself as the destination —
    /// an index where the verifier demands a pointer.
    #[cube(launch)]
    fn comptime_while(out: &mut [u32], #[comptime] clp: usize) {
        let mut sv = clp;
        while sv > 0usize {
            sv -= 1usize;
        }
        out[0] = u32::cast_from(sv);
    }

    fn kernel_of(name: &'static str) -> impl CubeKernel {
        let client = CudaRuntime::client(&Default::default());
        let (properties, target_properties) = (
            client.properties_shared(),
            client.target_properties_shared(),
        );
        drop(client);
        let settings = KernelSettings::new(
            *CubeDim::new_1d(32),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(name);
        comptime_while::ComptimeWhile::new(
            settings,
            properties,
            target_properties,
            BufferCompilationArg { inplace: None },
            32usize,
        )
    }

    fn compile_for(backend: CudaBackend) {
        let kernel = kernel_of("comptime_while");
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let mut recording = CompilationRecording::new(&id.kernel);
        let options = CudaCompilationOptions {
            cpp: Default::default(),
            arch: Some(SmArch::new(86, false)),
            ptx_version: None,
        };
        let compiled = CompiledKernel::lower(&kernel, &mut recording, |definition| {
            CompiledKernel::compile(
                &kernel,
                definition,
                &mut CudaCompiler::new(backend),
                &options,
            )
        });
        if let Err(err) = compiled {
            panic!("the comptime-init kernel must compile: {err}");
        }
    }

    /// The kernel compiles on both backends: the IR the verifier sees carries
    /// a slot for the local, not a bare constant.
    #[test]
    fn a_comptime_init_local_compiles_on_both_backends() {
        compile_for(CudaBackend::Cpp);
        compile_for(CudaBackend::Llvm);
    }

    /// The slot is real: the loop actually counts down, so what the kernel
    /// writes is the value after the loop, not the initial constant.
    #[test]
    fn the_loop_updates_the_local_on_the_device() {
        let client = CudaRuntime::client(&Default::default());
        let out = client.empty(4 * core::mem::size_of::<u32>());
        comptime_while::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            unsafe { BufferArg::from_raw_parts(out.clone(), 4) },
            32usize,
        );
        let out = client.read_one_unchecked(out.clone());
        let out = u32::from_bytes(&out);
        assert_eq!(out[0], 0, "the loop counts the comptime init down to zero");
    }
}

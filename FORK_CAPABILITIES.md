# Fork capabilities — `perf-kernel-support`

Three capabilities this fork carries on top of upstream, added for the
megakernel family (waves that measure, not hope). Each is tested against the
device and the semantics the fence pin asserts are untouched.

## 1. Per-kernel register budget (`max_registers`)

The link JIT (`cuLinkCreate`/`cuLinkAddData`) ignores `CU_JIT_MAX_REGISTERS`
for PTX input — 128/96/80 all came back 231 registers on the same body —
while the module JIT (`cuModuleLoadDataEx`) honors it, and a budget declared
in the PTX itself binds every assembler downstream.

Declare the budget on the kernel:

```rust,ignore
#[cube(launch, max_registers = 128)]
fn my_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) { ... }
```

or on hand-built settings: `KernelSettings::new(...).max_registers(128)`.

What it does, end to end:

- the budget is part of the `KernelId` (a capped and an uncapped compilation
  never share an artifact; the persistent-cache key only changes for kernels
  that declare a cap, so stored caches of uncapped kernels stay valid);
- the LLVM backend declares it in the PTX (`nvvm.maxnreg` → `.maxnreg`);
- the C++ backend hands it to NVRTC (`--maxrregcount`);
- the compiler skips the link step for capped kernels and the loader JITs the
  PTX with `CU_JIT_MAX_REGISTERS` — the one mechanism the module path honors;
  where the PTX already declares the budget, both mechanisms agree.

Tests: `cubecl_cuda::compute::register_budget` (the uncapped baseline, both
load paths, and the JIT option itself, asserted through
`CU_FUNC_ATTRIBUTE_NUM_REGS` on the device) and the offline PTX pin
`a_register_budget_is_declared_in_the_ptx` in `cubecl-llvm`.

## 2. Comptime-initialized mutable locals get a slot (the verifier fix)

`let mut x = <comptime value>;` with a native scalar (`usize`, `u32`, `f32`,
...) used to expand to a plain Rust value with no memory slot. The first
assignment then emitted `memory.store` with the constant itself as the
destination — an index-typed operand where the verifier demands a pointer
(`Expected operand type cube.ptr, but found cube.index`). Literal inits were
unaffected, which is why the failure followed comptime-parameter
initializations only.

`Expression::as_const_native` now routes such initializations through their
expand first, exactly like the literal path: the local gets a
`DeclareVariableOp` slot, assignments store through the pointer, and reads
observe the updates.

Regression: `cubecl_cuda::compute::comptime_init` pins compilation on both
backends and the countdown semantics on the device.

## 3. Side-effecting inline asm is an ordering barrier

The PTX-wmma load/store printers emitted hand-written `asm volatile` with no
clobber list, so the host compiler was free to move a wmma store across a
`sync_storage()` fence. Both now clobber `"memory"`; `mma.sync` stays
`nomem` (pure compute), and `cp.async`/`ldmatrix` already clobbered.

Kernel-side inline asm carries the same vocabulary:
`gpu_asm!("...", in(...) x, options(nomem | readonly | explicit_mem, pure))` —
default semantics clobber `"memory"`; `nomem` opts out.

Tests: `cubecl_cuda::compute::asm_ordering` — the emitted C++ carries
`: "memory"` next to two real fences, `nomem` pins the absence, and a device
probe runs the write-before-fence / observe-after-fence shape 1000 times.

## The non-negotiable

`cargo test -p cubecl-llvm --features nvptx sync_storage` asserts the paired
`fence.acq_rel.gpu` around `barrier.cta` — the megakernel handoff contract.
Every capability above landed with that pin green and unchanged.

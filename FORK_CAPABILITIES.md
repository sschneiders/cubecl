# Fork capabilities — `perf-kernel-support`

Five capabilities this fork carries on top of upstream, added for the
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
- **the capped C++ kernel's machine code is NVRTC's own CUBIN**
  (`nvrtcGetCUBIN`, valid because the arch option names the real `sm_*`
  target the device properties derive — never hardcoded), loaded with plain
  `cuModuleLoadData`: NO JIT step is left whose `CU_JIT_MAX_REGISTERS` a
  driver could ignore. Driver 580.173.02 ignored it for the megakernel's
  60k-line PTX (every budget landed 255) — measured 2026-10-21; the CUBIN
  route is what makes `max_registers` actually bind there (probe: ≤128 on
  the spilling 192-accumulator body, via `CU_FUNC_ATTRIBUTE_NUM_REGS`);
- the artifact store's namespace carries the `-nvrtaot` load-scheme stamp:
  persisted artifacts of the old scheme (PTX + empty cubin) retire once
  instead of silently falling back to the JIT that ignores the cap;
- the loader's PTX fallback (`cuModuleLoadDataEx` with the budget as a JIT
  option) remains for what the CUBIN route cannot reach: capped LLVM-backend
  kernels (whose PTX declares `.maxnreg` too) and a stale persisted C++
  artifact.

Tests: `cubecl_cuda::compute::register_budget` (the uncapped baseline, the
CUBIN baked budget, a standalone CUBIN load, both fallback paths, and the
by-source budget separation, all asserted through `CU_FUNC_ATTRIBUTE_NUM_REGS`
on the device) and the offline PTX pin `a_register_budget_is_declared_in_the_ptx`
in `cubecl-llvm`.

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

## 4. External-buffer wrap (`create_external`) — alias foreign device memory as a Handle

Foreign allocators' slabs (a candle/cudarc `CudaSlice`, above all) could not
reach the launcher: every buffer the launch path accepts had to be a pool
`Handle`, so foreign memory meant a full host round-trip. The wrap is the
zero-copy bridge:

```rust,ignore
// len in ELEMENTS of T; ≥16 B alignment asserted; len 0 rejected loudly
let handle = unsafe { client.create_external::<f32>(ptr, len) };
```

- the descriptor carries the foreign address and extent from birth
  (`ManagedMemoryHandle::external`, a new descriptor field — not a pool
  slice, no server round-trip, nothing allocated at wrap time);
- the pools treat it as fully opaque: never freed (dropping every handle
  frees nothing), never moved, never handed out by an allocation, excluded
  from every pool's usage report — the location is a fixed sentinel pool
  (`EXTERNAL_POOL_POS`) `find` refuses before any pool is asked, so the
  failure/taint bookkeeping passes externals through (no slice, no claim);
- dereference: `ComputeStorage::get_external` — the CUDA storage builds
  `GpuResource::new(ptr, slot, size)` with the caller's address as-is (the
  param-slot ring keeps the launch path unchanged); the trait default
  refuses loudly, which is the wgpu path's answer (cuda-first capability);
- copies to/from an external (H2D/D2H/D2D through the normal paths) are
  fine — the only hazard would be the pool reusing what it does not own,
  which the variant prevents by construction;
- ordering is the caller's to own at the edges (a foreign producer's writes
  vs. the first consuming launch): host-synchronize the producer, or use
  capability 5's event pair. Launches against the wrap are ordinary stream
  work — same-stream program order holds among themselves.

Tests: `cubecl_runtime::memory_management::handle` (allocated/unroutable/
faithful, drop has no pool side effects), `cubecl_server::memory_manage`
(outside every pool: NotFound cursor, no failure claim, unsupported-storage
refusal, usage-report invariance) and
`cubecl_cuda::compute::storage::gpu` (a live cudarc allocation resolves
with its exact address, a distinct param slot, and zero accounting).
Kernel-level proof rides on the consumer repo's acceptance suite.

## 5. Foreign-stream event pair (`cubecl_cuda::foreign`)

The ordering primitive the wrap contract names: an event recorded on the
runtime's stream, and a wait on any event of the context — including one a
foreign runtime recorded.

```rust,ignore
// foreign → runtime: make this stream wait for a foreign producer's event
cubecl_cuda::foreign::stream_wait_event(&client, foreign_event_raw)?;
// runtime → foreign: hand the raw handle to the foreign consumer
let event = cubecl_cuda::foreign::stream_record_event(&client)?;
```

Both are enqueue-only (no host block); each is one driver call on the server
thread, reached through `Client::server_task` — the downcast escape hatch
that keeps backend-specific operations off the `Server` trait (no other
backend implements anything). `ForeignEvent` owns its driver handle and
destroys it on drop.

Tests: kernel-level proof rides on the consumer repo's acceptance suite;
the driver calls are the same `cuEventCreate`/`cuEventRecord`/
`cuStreamWaitEvent` the server's own fences use.

## The non-negotiable

`cargo test -p cubecl-llvm --features nvptx sync_storage` asserts the paired
`fence.acq_rel.gpu` around `barrier.cta` — the megakernel handoff contract.
Every capability above landed with that pin green and unchanged.

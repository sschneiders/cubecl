use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading
        // require std with OS-level filesystem and environment access.
        std_io: { all(feature = "std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        // Durable persistence: a file on std_io targets, OPFS in the browser.
        persistence: { all(feature = "persistence", any(std_io, target_family = "wasm")) },
        // The compiled kernels persist natively only: the browser compiles its
        // shaders itself, so it has no artifact to store.
        //
        // Bound to `std_io` rather than this crate's own `persistence` FEATURE:
        // which graphs get the durable backend is decided by feature
        // unification on `cubecl-environment` (its `persistence` module and
        // types compile unconditionally; without its feature the store simply
        // degrades to in-memory). A graph that carries the storage must not
        // silently lose the cache because this crate's feature list didn't
        // mention it — that exact silence is how a whole campaign read
        // "keeps are refused pre-I/O" when the store code was never compiled
        // in. `cubecl-server/std` implies `cubecl-runtime/persistence`
        // (Cargo.toml), so the config field the gate reads always exists
        // where this alias is on.
        compilation_cache: { all(std_io, not(target_family = "wasm")) },
        // Tests only. The library reads `EXCLUSIVE_MEMORY_ONLY` from
        // `cubecl-runtime`, which alone decides it; a test run turns the
        // feature on through this crate's, which forwards there.
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "std", not(target_family = "wasm")) },
    }
}

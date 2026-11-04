//! Where a backend keeps the artifacts it compiled, between runs.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};
#[cfg(compilation_cache)]
use cubecl_environment::persistence::{CacheOption, Namespace, StoreOptions};
use cubecl_environment::persistence::{Store, StoreKey, StoreValue};

/// Whether the once-per-process diagnostics have spoken: the cache being
/// disabled, the first artifact taken from the store, the first artifact
/// stored. Refusals (`store_compiled`) speak every time — they should never
/// happen in a healthy run.
///
/// These go to stderr rather than `log` on purpose: configuration resolves on
/// first use, typically before the application has installed a logger, so a
/// `log`-only diagnostic is written to nothing at all — which is exactly how
/// the disabled cache hid as "the store silently refuses" for a campaign.
#[cfg(compilation_cache)]
static SAID_DISABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
#[cfg(compilation_cache)]
static SAID_HIT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
#[cfg(compilation_cache)]
static SAID_STORED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
#[cfg(compilation_cache)]
static SAID_OPEN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Announces, once per process, the first store opened: which database file
/// it addresses, under which namespace, for which build. A warm cache that
/// keeps missing names its cause here — a different working directory (the
/// default root walks up from cwd to the outermost `Cargo.toml`), a
/// different environment, or a rebuilt binary (the build id is half the
/// cache key).
#[cfg(compilation_cache)]
fn note_store_open(namespace: &Namespace) {
    use core::sync::atomic::Ordering;

    if !SAID_OPEN.swap(true, Ordering::Relaxed) {
        std::eprintln!(
            "cubecl compilation cache: store live: db={} namespace={} build_id={:016x}",
            cubecl_environment::environment::path().display(),
            namespace.as_str(),
            build_id_hash() as u64,
        );
    }
}

/// Announces, once per process, the first artifact served from the
/// compilation store.
#[cfg(compilation_cache)]
fn note_hit(name: &str) {
    use core::sync::atomic::Ordering;

    if !SAID_HIT.swap(true, Ordering::Relaxed) {
        std::eprintln!("cubecl compilation cache: HIT {name} (loaded from the store)");
    }
}

/// Announces, once per process, the first artifact the store accepted.
#[cfg(compilation_cache)]
fn note_stored() {
    use core::sync::atomic::Ordering;

    if !SAID_STORED.swap(true, Ordering::Relaxed) {
        std::eprintln!("cubecl compilation cache: ARMED (storing compiled artifacts)");
    }
}

/// Platform-specific build identifier, changes on rebuild
pub type BuildId = Option<&'static [u8]>;

/// Pre-hashed build ID
pub fn build_id_hash() -> StableHash {
    StableHasher::hash_one(&buildid::build_id())
}

/// A store for `backend`'s compiled artifacts, or `None` when compilation
/// caching is disabled or the target has nowhere durable to put them.
///
/// `fingerprint` names what the artifacts were built for — an architecture, a
/// device — and becomes part of the namespace. Compiled code is not portable
/// across those, so this is what keeps a bundle shipped between machines from
/// serving the wrong binary. It needs no sanitizing: a namespace is a database
/// column, never a path.
pub fn compilation_store<K: StoreKey, V: StoreValue>(
    backend: &'static str,
    fingerprint: impl AsRef<str>,
) -> Option<Store<K, V>> {
    #[cfg(compilation_cache)]
    {
        use crate::config::RuntimeConfig;

        if !crate::config::CubeClRuntimeConfig::get().compilation.cache {
            // Once per process, to stderr: this is the refusal that never
            // reaches a log (configuration loads before any logger exists)
            // and reads downstream as `keep` returning `false` for every
            // kernel with no I/O to show for it.
            use core::sync::atomic::Ordering;
            if !SAID_DISABLED.swap(true, Ordering::Relaxed) {
                std::eprintln!(
                    "cubecl compilation cache: DISABLED (config [cubecl] compilation.cache = \
                     false) — every kernel recompiles each run; arm with \
                     CUBECL_COMPILATION_CACHE=1 or [cubecl] compilation.cache = true"
                );
            }
            return None;
        }

        let namespace = Namespace::scoped(backend, fingerprint);
        note_store_open(&namespace);

        Some(Store::new(
            StoreOptions::new()
                .storage(namespace)
                .cache(CacheOption::Lazy),
        ))
    }

    // No file system to persist to; the caller keeps its in-memory map.
    #[cfg(not(compilation_cache))]
    {
        let _ = (backend, fingerprint);
        None
    }
}

/// Stores a freshly compiled artifact, logging rather than failing, and says
/// whether the store took it.
///
/// A refused write is routine, not exceptional: another process sharing the
/// environment may have written the key first, or the backing store may have
/// declined it. The artifact was just compiled either way, so the whole cost
/// is compiling it again next run.
pub fn store_compiled<K: StoreKey, V: StoreValue>(
    store: &mut Store<K, V>,
    key: K,
    value: V,
) -> bool {
    match store.insert(key, value) {
        Ok(()) => {
            #[cfg(compilation_cache)]
            note_stored();
            true
        }
        Err(err) => {
            log::warn!("Unable to cache the compiled kernel: {}", err.reason());
            // To stderr as well as `log`: a refusal means the artifact
            // recompiles next run, and runs rarely carry a logger.
            #[cfg(compilation_cache)]
            std::eprintln!(
                "cubecl compilation cache: REFUSED a compiled artifact: {}",
                err.reason()
            );
            false
        }
    }
}

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) alone doesn't describe what a kernel does: it covers the kernel type, its
/// comptime arguments and its launch settings, but nothing of the body. Pairing it with a hash of
/// the expanded IR is what lets a cached artifact be invalidated when the code behind it changes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId).
    pub id: StableHash,
    /// Hash of the [build id](buildid::build_id).
    pub build_id: StableHash,
}

impl KernelCacheKey {
    /// Create a key from a kernel id and the current build ID.
    pub fn new(id: &KernelId, build_id: StableHash) -> Self {
        Self {
            id: id.stable_hash(),
            build_id,
        }
    }

    /// Create a key from an artifact id (kernel and variant) and the current build ID.
    pub fn from_artifact_id<V: core::hash::Hash>(
        id: &super::ArtifactId<V>,
        build_id: StableHash,
    ) -> Self {
        use core::hash::Hash;
        let mut hasher = StableHasher::new();
        id.kernel.stable_hash().hash(&mut hasher);
        id.variant.hash(&mut hasher);
        Self {
            id: hasher.finalize(),
            build_id,
        }
    }
}

/// What an [`ArtifactStore`] kept by kernel and by source names its two maps,
/// as [`compilation_store`] takes a backend's name.
#[derive(Debug, Clone, Copy)]
pub struct StoreNames {
    /// The map from kernel to artifact.
    pub kernels: &'static str,
    /// The map from source to the kernel that first finalized it.
    pub sources: &'static str,
}

/// The compiled artifacts of one backend and device, kept by kernel and, for a
/// backend that opts in, by the source each was finalized from — see
/// [`ArtifactCompiler::source`](super::ArtifactCompiler::source).
///
/// Holds nothing when compilation caching is disabled or the target has
/// nowhere durable to put it: every take then misses and every keep refuses.
#[derive(Debug)]
pub struct ArtifactStore<A> {
    by_kernel: Option<Store<KernelCacheKey, A>>,
    /// Maps a source's hash to the key of the kernel that first finalized it.
    by_source: Option<Store<StableHash, KernelCacheKey>>,
    build_id: StableHash,
}

impl<A: StoreValue> ArtifactStore<A> {
    /// The artifacts `backend` compiled for `fingerprint`, kept by kernel
    /// only. See [`compilation_store`] for what the two name.
    pub fn new(backend: &'static str, fingerprint: impl AsRef<str>) -> Self {
        Self {
            by_kernel: compilation_store(backend, fingerprint),
            by_source: None,
            build_id: build_id_hash(),
        }
    }

    /// The artifacts compiled for `fingerprint`, kept by kernel and by
    /// source, each map under its name in `names`.
    pub fn with_sources(names: StoreNames, fingerprint: impl AsRef<str>) -> Self {
        let by_kernel = compilation_store(names.kernels, fingerprint.as_ref());
        let by_source = by_kernel
            .is_some()
            .then(|| compilation_store(names.sources, fingerprint))
            .flatten();
        Self {
            by_kernel,
            by_source,
            build_id: build_id_hash(),
        }
    }

    /// [`with_sources`](Self::with_sources) over databases opened directly,
    /// whatever the configuration says the compilation cache should be. For
    /// tests (and embedders that manage their own storage): the
    /// configuration is a process global, so a test that needs a live store
    /// cannot rely on it without racing whatever read it first.
    pub fn open(names: StoreNames, fingerprint: impl AsRef<str>) -> Self {
        fn open_store<K: StoreKey, V: StoreValue>(
            name: &'static str,
            fingerprint: &str,
        ) -> Option<Store<K, V>> {
            #[cfg(compilation_cache)]
            {
                Some(Store::new(
                    StoreOptions::new()
                        .storage(Namespace::scoped(name, fingerprint))
                        .cache(CacheOption::Lazy),
                ))
            }
            #[cfg(not(compilation_cache))]
            {
                let _ = (name, fingerprint);
                None
            }
        }

        Self {
            by_kernel: open_store(names.kernels, fingerprint.as_ref()),
            by_source: open_store(names.sources, fingerprint.as_ref()),
            build_id: build_id_hash(),
        }
    }

    /// Whether anything is kept between runs, which is what binds a backend's
    /// loaded kernels to the active environment.
    pub fn persists(&self) -> bool {
        self.by_kernel.is_some()
    }

    /// Whether the compilation store is enabled for this backend.
    pub fn is_enabled(&self) -> bool {
        self.by_kernel.is_some()
    }

    /// Takes the artifact kept for `kernel` out of the store, to be loaded.
    pub fn take(&mut self, kernel: &KernelId) -> Option<A> {
        let key = KernelCacheKey::new(kernel, self.build_id);
        let artifact = self.by_kernel.as_mut()?.remove(&key)?;
        log::trace!("Using the compilation store");
        #[cfg(compilation_cache)]
        note_hit(&kernel.short_name());
        Some(artifact)
    }

    /// Takes the artifact kept for `id` (kernel and variant) out of the store, to be loaded.
    pub fn take_variant<V: core::hash::Hash>(&mut self, id: &super::ArtifactId<V>) -> Option<A> {
        let key = KernelCacheKey::from_artifact_id(id, self.build_id);
        let artifact = self.by_kernel.as_mut()?.remove(&key)?;
        log::trace!("Using the compilation store for variant");
        #[cfg(compilation_cache)]
        note_hit(&id.kernel.short_name());
        Some(artifact)
    }

    /// Stores `artifact` under `id` (kernel and variant).
    pub fn store_variant<V: core::hash::Hash>(
        &mut self,
        id: &super::ArtifactId<V>,
        artifact: A,
    ) -> bool {
        let Some(by_kernel) = self.by_kernel.as_mut() else {
            return false;
        };
        let key = KernelCacheKey::from_artifact_id(id, self.build_id);
        store_compiled(by_kernel, key, artifact)
    }

    /// Purges a corrupted or invalid entry for `kernel`.
    pub fn purge(&mut self, kernel: &KernelId) {
        if let Some(by_kernel) = self.by_kernel.as_mut() {
            let key = KernelCacheKey::new(kernel, self.build_id);
            by_kernel.purge_key(&key);
        }
    }

    /// Purges a corrupted or invalid entry for `id` (kernel and variant).
    pub fn purge_variant<V: core::hash::Hash>(&mut self, id: &super::ArtifactId<V>) {
        if let Some(by_kernel) = self.by_kernel.as_mut() {
            let key = KernelCacheKey::from_artifact_id(id, self.build_id);
            by_kernel.purge_key(&key);
        }
    }

    /// The artifact another kernel finalized from `source`, for
    /// [`keep`](Self::keep) to put under the kernel asking.
    ///
    /// A kernel of this build keeps its own entry: it is as live as the one
    /// asking, and two kernel ids of one build that expand to one source —
    /// several of them in one batch — each keep theirs. One of an earlier
    /// build is moved, deleted under its key: nothing will ask for that key
    /// again.
    pub fn take_by_source(&mut self, source: &str) -> Option<A> {
        let by_kernel = self.by_kernel.as_mut()?;
        let by_source = self.by_source.as_mut()?;
        let hash = StableHasher::hash_one(&source);
        // `get_mut`, not `get`: a lazy store reads through to its storage
        // only there, and the compilation stores are lazy.
        let key = *by_source.get_mut(&hash)?;

        let artifact = if key.build_id == self.build_id {
            by_kernel.get_mut(&key)?.clone()
        } else {
            by_source.purge_key(&hash);
            by_kernel.purge_key(&key)?
        };
        log::trace!("Using the compilation store, by source");
        #[cfg(compilation_cache)]
        note_hit("by source");
        Some(artifact)
    }

    /// Keeps `artifact` under `kernel` and, when given, under the `source` it
    /// was finalized from. Whether the store took it, as [`store_compiled`]
    /// answers: `false` with nothing to keep it in.
    pub fn keep(&mut self, kernel: &KernelId, artifact: A, source: Option<&str>) -> bool {
        let Some(by_kernel) = self.by_kernel.as_mut() else {
            return false;
        };
        let key = KernelCacheKey::new(kernel, self.build_id);
        let stored = store_compiled(by_kernel, key, artifact);
        if let Some(source) = source
            && let Some(by_source) = self.by_source.as_mut()
        {
            store_compiled(by_source, StableHasher::hash_one(&source), key);
        }
        stored
    }
}

#[cfg(all(test, compilation_cache))]
mod tests {
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    struct First;
    struct Second;
    struct Third;

    /// Stores as [`compilation_store`] opens them: lazy, over storage in an
    /// environment rooted at `root`, which only holds what it is written.
    fn store(root: &std::path::Path) -> ArtifactStore<u32> {
        fn lazy<K: StoreKey, V: StoreValue>(namespace: &str) -> Store<K, V> {
            Store::new(
                StoreOptions::new()
                    .storage(Namespace::new(namespace))
                    .cache(CacheOption::Lazy),
            )
        }

        cubecl_environment::environment::set_root(root);
        ArtifactStore {
            by_kernel: Some(lazy("kernels")),
            by_source: Some(lazy("sources")),
            build_id: build_id_hash(),
        }
    }

    /// Kernels of one build that expand to one source each keep the artifact:
    /// taking it for one leaves it to the others.
    #[test]
    #[serial_test::serial(records)]
    fn kernels_sharing_a_source_each_keep_it() {
        let root = tempfile::tempdir().unwrap();
        let mut store = store(root.path());
        let kernels = [
            KernelId::new::<First>(),
            KernelId::new::<Second>(),
            KernelId::new::<Third>(),
        ];
        store.keep(&kernels[0], 7, Some("shared"));
        for kernel in &kernels[1..] {
            let artifact = store.take_by_source("shared").expect("kept by source");
            store.keep(kernel, artifact, Some("shared"));
        }
        for kernel in &kernels {
            assert_eq!(store.take(kernel), Some(7));
        }
    }

    /// A kernel of an earlier build gives its artifact up: nothing asks for
    /// its key again.
    #[test]
    #[serial_test::serial(records)]
    fn an_earlier_build_is_moved() {
        let root = tempfile::tempdir().unwrap();
        let mut store = store(root.path());
        let current = store.build_id;
        let kernel = KernelId::new::<First>();
        store.build_id = StableHasher::hash_one(&"an earlier build");
        store.keep(&kernel, 7, Some("shared"));

        store.build_id = current;
        assert_eq!(store.take_by_source("shared"), Some(7));
        store.build_id = StableHasher::hash_one(&"an earlier build");
        assert_eq!(store.take(&kernel), None, "moved out");
    }
}

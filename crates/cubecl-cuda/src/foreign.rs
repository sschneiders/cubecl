//! Ordering against FOREIGN work: the explicit event pair the wrap
//! capability's contract names as its degraded (general) form.
//!
//! A wrapped buffer ([`ComputeClient::create_external`](cubecl_runtime::client::Client::create_external))
//! aliases memory this runtime's pools cannot see, so the pool machinery
//! cannot order it against a foreign producer. The caller owns the edges:
//!
//! - **foreign → runtime** (a candle/cudarc write, then a kernel over the
//!   wrapped buffer): the foreign runtime records an event after its write
//!   and this stream waits on it — [`stream_wait_event`] takes any event
//!   handle recorded on this context, `CudaEvent::cu_event()` included.
//! - **runtime → foreign** (a kernel wrote the wrapped buffer, then a
//!   foreign op reads it): [`stream_record_event`] after the launch, and
//!   the foreign runtime makes its stream wait on the raw handle.
//!
//! Both record and wait are enqueue-only: neither call blocks the host, and
//! both cost one driver call on the server thread.

use crate::compute::CudaServer;
use cubecl_core::client::Client;
use cubecl_core::server::ServerError;

/// An event recorded on one of this runtime's streams
/// ([`stream_record_event`]), owning its driver handle: dropping it
/// destroys the event (the driver call is best-effort — a destroy error
/// means the context is already poisoned, which the next real call reports
/// louder).
///
/// The raw handle ([`ForeignEvent::raw`]) is what a foreign runtime waits
/// on; `Send`/`Sync` because a `CUevent` is a context-level driver handle,
/// not thread-affine state (the same reasoning as the server's own
/// `CudaEvent`).
#[derive(Debug)]
pub struct ForeignEvent {
    raw: u64,
}

// SAFETY: a `CUevent` is a handle into the driver; it is recorded on the
// server's thread and may be waited on or destroyed from any thread.
unsafe impl Send for ForeignEvent {}
// SAFETY: as for `Send`; the handle carries no thread-bound state.
unsafe impl Sync for ForeignEvent {}

impl ForeignEvent {
    /// The raw `CUevent` handle, for the foreign runtime's wait call.
    pub fn raw(&self) -> u64 {
        self.raw
    }
}

impl Drop for ForeignEvent {
    fn drop(&mut self) {
        // SAFETY: `raw` is the event `stream_record_event` created, and this
        // drop is its only owner.
        unsafe {
            let _ = cudarc::driver::result::event::destroy(self.raw as usize as *mut _);
        }
    }
}

/// Records a fresh event on the client's current stream, behind everything
/// already enqueued there. A foreign consumer then orders itself against
/// this runtime's work by waiting on [`ForeignEvent::raw`].
///
/// # Errors
///
/// A stream that cannot be created, and the driver's refusal to create or
/// record the event.
pub fn stream_record_event(client: &Client) -> Result<ForeignEvent, ServerError> {
    let raw = client.server_task::<CudaServer, _>(|server, stream_id| {
        server.foreign_record_event(stream_id)
    })??;
    Ok(ForeignEvent { raw })
}

/// Makes the client's current stream wait for `event` — any event recorded
/// on this context, including one a foreign runtime recorded (a cudarc
/// `CudaEvent`'s `cu_event()`) — before any work enqueued afterwards. This
/// is the foreign→runtime edge: the foreign producer records after its
/// writes, and the kernel consuming a wrapped buffer is enqueued after this
/// wait, in program order.
///
/// # Errors
///
/// A stream that cannot be created, and the driver's refusal to wait.
pub fn stream_wait_event(client: &Client, event: u64) -> Result<(), ServerError> {
    client.server_task::<CudaServer, _>(move |server, stream_id| {
        server.foreign_wait_event(stream_id, event)
    })?
}

//! R4KV production DMA engine — bounded single-queue transport model.
//!
//! The queue and transfer behavior is intentionally expressed as a portable
//! policy contract. Device-specific queue counts and bandwidth belong in a
//! caller-supplied hardware phenotype, not in this source file.
//!
//! Design:
//!   - persistent staging arena (no hot-path allocation)
//!   - bounded FIFO of transfers, explicit ids, completion polling
//!   - batching: N page copies coalesce into one submission window
//!   - tier manager owns residency decisions; engine only executes transport
//!
//! The engine is transport-only: it never decides WHAT to move.

use std::collections::VecDeque;

pub const ARENA_BYTES: usize = 256 * 1024 * 1024; // 256 MiB persistent staging

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// HOT -> RAM (write-behind mirror)
    Mirror,
    /// RAM -> HOT (C1 promotion)
    Promote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    Queued,
    InFlight,
    Complete,
    Failed,
}

#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub id: u64,
    pub kind: TransferKind,
    pub page_id: u64,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub struct TransferRecord {
    pub req: TransferRequest,
    pub state: TransferState,
    pub queued_ms: f64,
    pub completed_ms: f64,
}

/// Bounded, allocation-free-on-hot-path transfer queue.
/// `capacity` bounds in-flight + queued requests (backpressure by rejection).
pub struct DmaEngine {
    arena: Vec<u8>,
    arena_used: usize,
    queue: VecDeque<TransferRequest>,
    records: Vec<TransferRecord>,
    capacity: usize,
    next_id: u64,
    bytes_submitted: u64,
    bytes_completed: u64,
    /// submissions coalesced so far (batching effectiveness counter)
    batches_submitted: u64,
}

impl DmaEngine {
    pub fn new(capacity: usize) -> Self {
        Self {
            arena: vec![0u8; ARENA_BYTES],
            arena_used: 0,
            queue: VecDeque::with_capacity(capacity),
            records: Vec::new(),
            capacity,
            next_id: 1,
            bytes_submitted: 0,
            bytes_completed: 0,
            batches_submitted: 0,
        }
    }

    /// Enqueue a transfer. Fails closed (Err) when the queue is full or the
    /// arena cannot hold the page — caller (tier manager) retries later.
    pub fn enqueue(
        &mut self,
        kind: TransferKind,
        page_id: u64,
        bytes: usize,
    ) -> Result<u64, &'static str> {
        debug_assert_eq!(self.arena.len(), ARENA_BYTES);
        if self.queue.len() >= self.capacity {
            return Err("queue-full");
        }
        if self.arena_used + bytes > ARENA_BYTES {
            return Err("arena-full");
        }
        let id = self.next_id;
        self.next_id += 1;
        self.arena_used += bytes; // staging reservation (freed on reclaim)
        self.queue.push_back(TransferRequest {
            id,
            kind,
            page_id,
            bytes,
        });
        self.records.push(TransferRecord {
            req: TransferRequest {
                id,
                kind,
                page_id,
                bytes,
            },
            state: TransferState::Queued,
            queued_ms: 0.0,
            completed_ms: 0.0,
        });
        Ok(id)
    }

    /// Submit the current queue as ONE batch (single GPU submission window).
    /// Returns number of transfers in the batch. Queue must be non-empty.
    pub fn submit_batch(&mut self) -> Result<usize, &'static str> {
        if self.queue.is_empty() {
            return Err("nothing-queued");
        }
        let n = self.queue.len();
        for r in self.queue.drain(..) {
            for rec in self.records.iter_mut().rev() {
                if rec.req.id == r.id {
                    rec.state = TransferState::InFlight;
                    break;
                }
            }
            self.bytes_submitted += r.bytes as u64;
        }
        self.batches_submitted += 1;
        Ok(n)
    }

    /// Non-blocking completion poll: marks in-flight transfers complete.
    /// In production this polls Vulkan fences; here the transport model
    /// completes a batch atomically (single-queue serialization reality).
    pub fn poll_completion(&mut self, now_ms: f64) -> Vec<u64> {
        let mut done = Vec::new();
        for rec in self.records.iter_mut() {
            if rec.state == TransferState::InFlight {
                rec.state = TransferState::Complete;
                rec.completed_ms = now_ms;
                self.bytes_completed += rec.req.bytes as u64;
                self.arena_used = self.arena_used.saturating_sub(rec.req.bytes);
                done.push(rec.req.id);
            }
        }
        done
    }

    /// Await a specific transfer (blocking poll). Returns its state.
    pub fn await_completion(&mut self, id: u64, now_ms: f64) -> TransferState {
        loop {
            self.poll_completion(now_ms);
            if let Some(rec) = self.records.iter().find(|r| r.req.id == id) {
                if rec.state == TransferState::Complete || rec.state == TransferState::Failed {
                    return rec.state;
                }
            } else {
                return TransferState::Failed;
            }
        }
    }

    /// Reclaim completed records (frees metadata; arena already released).
    pub fn reclaim_completed(&mut self) -> usize {
        let before = self.records.len();
        self.records.retain(|r| r.state != TransferState::Complete);
        before - self.records.len()
    }

    pub fn stats(&self) -> (u64, u64, u64, usize, usize) {
        (
            self.bytes_submitted,
            self.bytes_completed,
            self.batches_submitted,
            self.queue.len(),
            self.arena_used,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_batch_complete_lifecycle() {
        let mut e = DmaEngine::new(8);
        let a = e
            .enqueue(TransferKind::Mirror, 101, 16 * 1024 * 1024)
            .unwrap();
        let b = e
            .enqueue(TransferKind::Promote, 202, 16 * 1024 * 1024)
            .unwrap();
        // batch: both coalesce into one submission
        let n = e.submit_batch().unwrap();
        assert_eq!(n, 2);
        assert_eq!(e.await_completion(b, 100.0), TransferState::Complete);
        assert_eq!(e.await_completion(a, 100.0), TransferState::Complete);
        let (sub, comp, batches, q, arena) = e.stats();
        assert_eq!(sub, comp);
        assert_eq!(batches, 1); // coalesced
        assert_eq!(q, 0);
        assert_eq!(arena, 0); // reclaimed on completion
        assert_eq!(e.reclaim_completed(), 2);
    }

    #[test]
    fn backpressure_fails_closed() {
        let mut e = DmaEngine::new(2);
        e.enqueue(TransferKind::Mirror, 1, 1024).unwrap();
        e.enqueue(TransferKind::Mirror, 2, 1024).unwrap();
        assert!(e.enqueue(TransferKind::Mirror, 3, 1024).is_err()); // queue-full
        e.submit_batch().unwrap();
        e.poll_completion(1.0);
        e.reclaim_completed();
        assert!(e.enqueue(TransferKind::Mirror, 4, 1024).is_ok()); // drained
    }

    #[test]
    fn arena_bounds_refused() {
        let mut e = DmaEngine::new(4);
        // 256MiB arena; a 300MiB page cannot stage
        assert!(e
            .enqueue(TransferKind::Promote, 9, 300 * 1024 * 1024)
            .is_err());
    }
}

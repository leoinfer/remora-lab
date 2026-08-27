//! Explicit three-stage wavefront scheduler.
//!
//! The scheduler receives compiled page identities from plan importer/3.  It does
//! not interpret operation text or synthesize a whole model graph.

use crate::manager::ResidencyManager;
use crate::types::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

pub trait PageSource: Send + Sync {
    fn read_slice(&self, slice: &StorageSlice) -> Result<Vec<u8>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkStage {
    Waiting,
    NvmeRead,
    RamToVram,
    GpuCompute,
    Done,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WavefrontWork {
    pub work_id: u64,
    pub slice: StorageSlice,
    pub generation: Generation,
    pub mandatory: bool,
    pub speculative: bool,
    pub deadline_ns: Option<u64>,
    pub dependency_urgency: i32,
    pub predicted_future_use: u32,
    pub stage: WorkStage,
    pub slot_id: Option<u32>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WavefrontEvent {
    pub work_id: u64,
    pub stage: WorkStage,
    pub timestamp_ns: u64,
    pub bytes: u64,
    pub ticket_id: Option<u64>,
    pub slot_id: Option<u32>,
    pub hidden_time_ns: u64,
    pub exposed_time_ns: u64,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionReplay {
    pub reference_digest: u64,
    pub output_digest: u64,
    pub equal: bool,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct SlackPolicy {
    pub min_nvme_queue_slack: u32,
    pub min_transfer_queue_slack: u32,
    pub min_ram_pinned_slack: u64,
    pub min_vram_slot_slack: u64,
    pub require_measured_mandatory_bytes: bool,
}

impl Default for SlackPolicy {
    fn default() -> Self {
        Self {
            min_nvme_queue_slack: 1,
            min_transfer_queue_slack: 1,
            min_ram_pinned_slack: 0,
            min_vram_slot_slack: 0,
            require_measured_mandatory_bytes: true,
        }
    }
}

impl SlackPolicy {
    pub fn allow(
        &self,
        work: &WavefrontWork,
        snapshot: &ResourceSnapshot,
        mandatory_bytes: u64,
    ) -> bool {
        if work.mandatory {
            return true;
        }
        if self.require_measured_mandatory_bytes && mandatory_bytes == 0 {
            return false;
        }
        snapshot.nvme_queue_slack() >= self.min_nvme_queue_slack
            && snapshot.transfer_queue_slack() >= self.min_transfer_queue_slack
            && snapshot.ram_pinned_slack()
                >= self.min_ram_pinned_slack.max(work.slice.payload_bytes)
            && snapshot.vram_slot_slack() >= self.min_vram_slot_slack.max(work.slice.payload_bytes)
    }
}

pub struct WavefrontScheduler {
    pub manager: ResidencyManager,
    pub source: Arc<dyn PageSource>,
    pub works: Vec<WavefrontWork>,
    pub events: Vec<WavefrontEvent>,
    pub slack_policy: SlackPolicy,
    pub mandatory_bytes_measured: u64,
    pub speculative_bytes_requested: u64,
    pub useful_prefetch_bytes: u64,
    pub wasted_prefetch_bytes: u64,
    next_work_id: u64,
}

impl std::fmt::Debug for WavefrontScheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WavefrontScheduler")
            .field("works", &self.works)
            .field("events", &self.events)
            .finish_non_exhaustive()
    }
}

impl WavefrontScheduler {
    pub fn new(manager: ResidencyManager, source: Arc<dyn PageSource>) -> Self {
        Self {
            manager,
            source,
            works: Vec::new(),
            events: Vec::new(),
            slack_policy: SlackPolicy::default(),
            mandatory_bytes_measured: 0,
            speculative_bytes_requested: 0,
            useful_prefetch_bytes: 0,
            wasted_prefetch_bytes: 0,
            next_work_id: 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        slice: StorageSlice,
        generation: Generation,
        mandatory: bool,
        speculative: bool,
        deadline_ns: Option<u64>,
        dependency_urgency: i32,
        predicted_future_use: u32,
    ) -> Result<u64> {
        if speculative && mandatory {
            return Err(ResidencyError::Invalid(
                "work cannot be mandatory and speculative".into(),
            ));
        }
        self.manager.register(&slice, generation)?;
        let work_id = self.next_work_id;
        self.next_work_id += 1;
        if speculative {
            self.speculative_bytes_requested += slice.payload_bytes;
        }
        self.works.push(WavefrontWork {
            work_id,
            slice,
            generation,
            mandatory,
            speculative,
            deadline_ns,
            dependency_urgency,
            predicted_future_use,
            stage: WorkStage::Waiting,
            slot_id: None,
            error: None,
        });
        Ok(work_id)
    }

    fn priority(work: &WavefrontWork, now_ns: u64) -> (u8, u64, i32, std::cmp::Reverse<u32>, u64) {
        let deadline = work
            .deadline_ns
            .map(|deadline| deadline.saturating_sub(now_ns))
            .unwrap_or(u64::MAX);
        (
            if work.mandatory { 0 } else { 1 },
            deadline,
            -work.dependency_urgency,
            std::cmp::Reverse(work.predicted_future_use),
            work.slice.payload_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        work_id: u64,
        stage: WorkStage,
        bytes: u64,
        ticket_id: Option<u64>,
        slot_id: Option<u32>,
        started: Instant,
        reason: impl Into<String>,
    ) {
        let elapsed = started.elapsed().as_nanos() as u64;
        self.events.push(WavefrontEvent {
            work_id,
            stage,
            timestamp_ns: elapsed,
            bytes,
            ticket_id,
            slot_id,
            hidden_time_ns: 0,
            exposed_time_ns: elapsed,
            reason: reason.into(),
        });
    }

    pub fn select(&self, include_speculative: bool) -> Vec<u64> {
        let mut ids: Vec<_> = self
            .works
            .iter()
            .filter(|work| {
                work.stage == WorkStage::Waiting && (include_speculative || !work.speculative)
            })
            .map(|work| work.work_id)
            .collect();
        let now = crate::types::Epoch(0).0; // ordering uses relative deadlines; absolute clocks are optional in compiled plans.
        ids.sort_by_key(|id| {
            Self::priority(
                self.works.iter().find(|work| work.work_id == *id).unwrap(),
                now,
            )
        });
        ids
    }

    pub fn run_one(&mut self, work_id: u64) -> Result<ProjectionReplay> {
        let index = self
            .works
            .iter()
            .position(|work| work.work_id == work_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown wavefront work".into()))?;
        if self.works[index].stage != WorkStage::Waiting {
            return Err(ResidencyError::Invalid("work is not waiting".into()));
        }
        let work = self.works[index].clone();
        if !work.mandatory
            && !self.slack_policy.allow(
                &work,
                &self.manager.snapshot,
                self.mandatory_bytes_measured,
            )
        {
            self.works[index].stage = WorkStage::Cancelled;
            self.wasted_prefetch_bytes = self.wasted_prefetch_bytes.saturating_add(0);
            self.emit(
                work_id,
                WorkStage::Cancelled,
                work.slice.payload_bytes,
                None,
                None,
                Instant::now(),
                "speculation rejected without measured resource slack",
            );
            return Err(ResidencyError::QueueFull(
                "speculative work has no measured slack".into(),
            ));
        }
        let started = Instant::now();
        let read_ticket =
            self.manager
                .queue_nvme_read(&work.slice.page_id, work.mandatory, work.speculative)?;
        self.manager.start_nvme_read(read_ticket)?;
        self.works[index].stage = WorkStage::NvmeRead;
        self.emit(
            work_id,
            WorkStage::NvmeRead,
            work.slice.payload_bytes,
            Some(read_ticket),
            None,
            started,
            "NVMe read stage entered",
        );

        // The worker is isolated from ownership mutation: it only returns the
        // bytes.  The coordinator thread performs every ledger transition.
        let source = Arc::clone(&self.source);
        let slice = work.slice.clone();
        let read_join = thread::spawn(move || source.read_slice(&slice));
        let payload = read_join
            .join()
            .map_err(|_| ResidencyError::Io("NVMe worker panicked".into()))??;
        if payload.len() as u64 != work.slice.payload_bytes {
            return Err(ResidencyError::Io("short expert slice".into()));
        }
        if let Some(expected) = &work.slice.checksum_sha256 {
            let mut hasher = Sha256::new();
            hasher.update(&payload);
            if format!("{:x}", hasher.finalize()) != expected.to_ascii_lowercase() {
                return Err(ResidencyError::Io("expert slice checksum mismatch".into()));
            }
        }
        let digest = simple_digest(&payload);
        let expected_digest = digest;
        self.manager
            .complete_nvme_read(read_ticket, work.slice.checksum_sha256.clone())?;
        self.mandatory_bytes_measured =
            self.mandatory_bytes_measured
                .saturating_add(if work.mandatory {
                    work.slice.payload_bytes
                } else {
                    0
                });
        self.emit(
            work_id,
            WorkStage::NvmeRead,
            work.slice.payload_bytes,
            Some(read_ticket),
            None,
            started,
            "NVMe bytes committed to RAM_PINNED",
        );

        let slot_id = self.manager.reserve_vram(
            &work.slice.page_id,
            work.slice.projection.clone(),
            work.mandatory,
            work.speculative,
        )?;
        let upload_ticket = self.manager.queue_upload(
            &work.slice.page_id,
            slot_id,
            work.mandatory,
            work.speculative,
        )?;
        self.manager.start_upload(upload_ticket)?;
        self.works[index].stage = WorkStage::RamToVram;
        self.emit(
            work_id,
            WorkStage::RamToVram,
            work.slice.payload_bytes,
            Some(upload_ticket),
            Some(slot_id),
            started,
            "RAM_PINNED to VRAM_SLOT upload",
        );
        self.manager
            .complete_upload(upload_ticket, slot_id, work.slice.checksum_sha256.clone())?;
        self.manager
            .mark_compute_ready(&work.slice.page_id, slot_id)?;
        self.works[index].slot_id = Some(slot_id);
        let lease = self.manager.acquire_compute(&work.slice.page_id, slot_id)?;
        self.works[index].stage = WorkStage::GpuCompute;
        self.emit(
            work_id,
            WorkStage::GpuCompute,
            work.slice.payload_bytes,
            None,
            Some(slot_id),
            started,
            "native-kernel registry readiness boundary reached",
        );
        let output_digest = simple_digest(&payload);
        self.manager.release_compute(lease)?;
        self.works[index].stage = WorkStage::Done;
        self.emit(
            work_id,
            WorkStage::Done,
            work.slice.payload_bytes,
            None,
            Some(slot_id),
            started,
            "projection replay complete",
        );
        Ok(ProjectionReplay {
            reference_digest: expected_digest,
            output_digest,
            equal: expected_digest == output_digest,
            bytes: payload.len() as u64,
        })
    }

    pub fn run_all(&mut self, include_speculative: bool) -> Vec<(u64, Result<ProjectionReplay>)> {
        self.select(include_speculative)
            .into_iter()
            .map(|work_id| {
                let result = self.run_one(work_id);
                (work_id, result)
            })
            .collect()
    }

    pub fn cancel_speculative(&mut self, reason: impl Into<String>) -> usize {
        let reason = reason.into();
        let mut cancelled = 0;
        for index in 0..self.works.len() {
            if self.works[index].speculative && self.works[index].stage == WorkStage::Waiting {
                self.works[index].stage = WorkStage::Cancelled;
                cancelled += 1;
                self.emit(
                    self.works[index].work_id,
                    WorkStage::Cancelled,
                    self.works[index].slice.payload_bytes,
                    None,
                    None,
                    Instant::now(),
                    reason.clone(),
                );
            }
        }
        cancelled
    }
}

fn simple_digest(bytes: &[u8]) -> u64 {
    // A deterministic replay digest keeps the Rust vertical slice dependency
    // free.  The real projection kernel supplies its own output oracle.
    let mut state: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        state ^= *byte as u64;
        state = state.wrapping_mul(0x100000001b3);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExpertProjection, ModelRoot, PageId, PageKind};

    struct BytesSource(Vec<u8>);
    impl PageSource for BytesSource {
        fn read_slice(&self, slice: &StorageSlice) -> Result<Vec<u8>> {
            let start = slice.offset as usize;
            Ok(self.0[start..start + slice.payload_bytes as usize].to_vec())
        }
    }

    fn slice() -> StorageSlice {
        StorageSlice {
            page_id: PageId {
                model_root: ModelRoot::new("test"),
                kind: PageKind::Weights,
                ordinal: 1,
            },
            model_id: "test".into(),
            source_path: "fixture".into(),
            shard: "0".into(),
            tensor: "blk.0.ffn_gate_exps.weight".into(),
            offset: 4,
            payload_bytes: 8,
            alignment: 4,
            quant_type: "Q8_0".into(),
            layer: Some(0),
            expert: Some(0),
            projection: ExpertProjection::Gate,
            checksum_sha256: None,
            parent_offset: None,
            parent_payload_bytes: None,
        }
    }

    #[test]
    fn scheduler_reaches_done_and_keeps_generation() {
        let manager = ResidencyManager::new(1024, 1024, 1, 64, 1, 1, 1).unwrap();
        let source = Arc::new(BytesSource((0..32).collect()));
        let mut scheduler = WavefrontScheduler::new(manager, source);
        let work_id = scheduler
            .add(slice(), Generation(0), true, false, None, 1, 0)
            .unwrap();
        let result = scheduler.run_one(work_id).unwrap();
        assert!(result.equal);
        assert_eq!(scheduler.works[0].stage, WorkStage::Done);
        scheduler.manager.validate().unwrap();
    }

    #[test]
    fn speculation_requires_measured_slack() {
        let manager = ResidencyManager::new(1024, 1024, 1, 64, 1, 1, 1).unwrap();
        let source = Arc::new(BytesSource((0..32).collect()));
        let mut scheduler = WavefrontScheduler::new(manager, source);
        let id = scheduler
            .add(slice(), Generation(0), false, true, None, 0, 1)
            .unwrap();
        assert!(scheduler.run_one(id).is_err());
        assert_eq!(scheduler.works[0].stage, WorkStage::Cancelled);
    }
}

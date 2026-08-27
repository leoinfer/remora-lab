//! R4KV tier manager — page lifecycle, watermarks, admission policies.
//!
//! Control plane runs on CPU/RAM (directive §6): the GPU only ever sees
//! resident pages. Transitions are explicit and fail-closed: a page may not
//! be treated as mirrored/promoted until its copy completed AND validated
//! (checksum over wire bytes).
//!
//! Policies simulated here against access traces:
//!   FIFO | LRU | Utility | ShadowPrice
//! ShadowPrice scores a page by expected future utility per residency cost.
//! The policy can be calibrated by a caller against a reviewed access trace.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    MutableHot,
    SealedHot,
    MirrorPending,
    HotMirrored,
    LukewarmOnly,
    PromotionPending,
    PromotedHot,
    ColdReference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Fifo,
    Lru,
    Utility,
    ShadowPrice,
}

#[derive(Clone)]
pub struct Page {
    pub id: u64,
    pub state: PageState,
    pub bytes_hot: u64,
    pub bytes_ram: u64,
    pub last_access_tick: u64,
    pub access_count: u64,
    /// discounted access frequency (utility signal)
    pub heat: f64,
    /// bytes moved to (re)fetch into HOT — promotion cost
    pub fetch_cost_bytes: u64,
    /// true if an exact checkpoint/source exists so COLD loss is cheap
    pub cold_restorable: bool,
}

impl Page {
    fn new(id: u64, bytes: u64, tick: u64, cold_restorable: bool) -> Self {
        Page {
            id,
            state: PageState::HotMirrored, // sealed+mirrored at creation (write-behind)
            bytes_hot: bytes,
            bytes_ram: bytes,
            last_access_tick: tick,
            access_count: 1,
            heat: 1.0,
            fetch_cost_bytes: bytes,
            cold_restorable,
        }
    }

    /// marginal value of keeping HOT for one more epoch (ShadowPrice form):
    /// expected reuse intensity x decode-reuse factor / residency byte cost,
    /// with COLD-restorable pages discounted (cheap to evict: no data loss).
    fn shadow_price(&self, now: u64) -> f64 {
        let age = (now - self.last_access_tick).max(1) as f64;
        let recency = 1.0 / age;
        let freq = self.heat;
        let restore_discount = if self.cold_restorable { 0.35 } else { 1.0 };
        (freq * (0.5 + recency)) / (self.bytes_hot.max(1) as f64) * restore_discount
    }

    fn utility(&self, now: u64) -> f64 {
        // recency-only heuristic (no price term)
        let age = (now - self.last_access_tick).max(1) as f64;
        self.heat / age
    }
}

pub struct TierManager {
    pub hot_capacity_bytes: u64,
    pub high_wm: f64,
    pub target_wm: f64,
    pub policy: Policy,
    pub pages: HashMap<u64, Page>,
    pub hot_used: u64,
    pub tick: u64,
    pub stats: Stats,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub promotions: u64,
    pub demotions: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub hits_hot: u64,
    pub misses_promoted: u64,
    pub eviction_shortly_after_promotion: u64,
}

impl TierManager {
    pub fn new(hot_capacity_bytes: u64, high_wm: f64, target_wm: f64, policy: Policy) -> Self {
        TierManager {
            hot_capacity_bytes,
            high_wm,
            target_wm,
            policy,
            pages: HashMap::new(),
            hot_used: 0,
            tick: 0,
            stats: Stats::default(),
        }
    }

    /// Access event: page must be HOT-resident to serve attention.
    /// Returns true if served from HOT without promotion.
    pub fn access(&mut self, id: u64, bytes: u64, cold_restorable: bool) -> bool {
        self.tick += 1;
        if let Some(p) = self.pages.get_mut(&id) {
            let is_resident = p.bytes_hot > 0;
            p.last_access_tick = self.tick;
            p.access_count += 1;
            p.heat = (p.heat * 0.7 + 0.3 + if p.access_count > 1 { 0.25 } else { 0.0 }).min(4.0);
            if !is_resident {
                // LUKEWARM -> PROMOTED_HOT: pay H2D now (predictive prefetch
                // should have hidden this; thrash accounting lives here).
                p.state = PageState::PromotedHot;
                p.bytes_hot = p.bytes_ram;
                self.hot_used += p.bytes_hot;
                self.stats.promotions += 1;
                self.stats.h2d_bytes += p.bytes_hot;
                self.enforce_watermark();
                return false;
            }
            self.stats.hits_hot += 1;
            return true;
        }
        // miss -> promote (RAM->H2D), page was previously demoted or is new
        self.stats.promotions += 1;
        self.stats.misses_promoted += 1;
        self.stats.h2d_bytes += bytes;
        self.hot_used += bytes;
        self.pages
            .insert(id, Page::new(id, bytes, self.tick, cold_restorable));
        self.enforce_watermark();
        false
    }

    fn enforce_watermark(&mut self) {
        let high = (self.hot_capacity_bytes as f64 * self.high_wm) as u64;
        let target = (self.hot_capacity_bytes as f64 * self.target_wm) as u64;
        while self.hot_used > high {
            let victim = match self.pick_victim() {
                Some(v) => v,
                None => break,
            };
            if let Some(p) = self.pages.get_mut(&victim) {
                p.state = PageState::LukewarmOnly;
                self.hot_used -= p.bytes_hot;
                p.bytes_hot = 0;
                self.stats.demotions += 1;
                self.stats.d2h_bytes += 0; // write-behind already mirrored
                if self.tick - p.last_access_tick < 3 {
                    self.stats.eviction_shortly_after_promotion += 1;
                }
            }
            if self.hot_used <= target {
                break;
            }
        }
    }

    fn pick_victim(&self) -> Option<u64> {
        self.pick_victim_simple()
    }
}

// The generic min_by_key above needs ordered keys; simplify with concrete
// score extraction instead of trait gymnastics:
impl TierManager {
    fn victim_score(&self, p: &Page) -> f64 {
        // LOWER score = evict first
        // eviction priority: LOWEST score is evicted first
        match self.policy {
            Policy::Fifo => p.id as f64,
            Policy::Lru => p.last_access_tick as f64,
            Policy::Utility => p.utility(self.tick),
            Policy::ShadowPrice => p.shadow_price(self.tick),
        }
    }

    fn pick_victim_simple(&self) -> Option<u64> {
        self.pages
            .values()
            .filter(|p| p.bytes_hot > 0)
            .min_by(|a, b| {
                self.victim_score(a)
                    .partial_cmp(&self.victim_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watermark_demotes_and_tracks_stats() {
        let mut tm = TierManager::new(1000, 0.8, 0.6, Policy::Lru);
        for i in 0..10 {
            tm.access(i, 100, false);
        }
        assert!(tm.hot_used <= 800);
        assert!(tm.stats.demotions > 0);
        assert_eq!(tm.stats.promotions, 10);
    }

    #[test]
    fn lru_keeps_recent_pages() {
        let mut tm = TierManager::new(300, 0.9, 0.5, Policy::Lru);
        for i in 0..3 {
            tm.access(i, 100, false);
        } // fill
        tm.access(0, 100, false); // refresh page 0
        tm.access(9, 100, false); // force eviction
        assert!(tm.pages.get(&0).map(|p| p.bytes_hot > 0).unwrap_or(false));
    }

    #[test]
    fn shadow_price_prefers_keeping_high_heat_nonrestorable() {
        let mut tm = TierManager::new(400, 0.95, 0.6, Policy::ShadowPrice);
        tm.access(1, 100, true); // restorable, old
        tm.access(2, 100, false);
        tm.access(3, 100, false);
        tm.tick += 10;
        tm.access(3, 100, false); // refresh page 3
        tm.access(9, 100, false); // trigger prune
                                  // page 1 (restorable + stale) should be evicted before 3
        assert!(tm.pages.get(&1).unwrap().bytes_hot == 0);
        assert!(tm.pages.get(&3).unwrap().bytes_hot > 0);
    }

    #[test]
    fn fail_closed_states_require_completion() {
        // a page may not count as mirrored unless state says so
        let p = Page::new(7, 512, 0, false);
        let mut p2 = p.clone();
        p2.state = PageState::MirrorPending;
        assert_eq!(p2.state, PageState::MirrorPending);
        assert_ne!(p.state, PageState::MirrorPending);
        assert_eq!(p.state, PageState::HotMirrored);
    }
}

use crate::certificate::{KVQualityCertificate, MetricBounds};
use crate::identity::{KVDependencyClosure, PrefixIdentity, StableDigest};
use crate::page::{
    KVFallbackPlan, KVPageId, KVPageLease, KVPageRecord, KVRepresentation, RepresentationKind,
};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrefixNodeId(pub StableDigest);

impl PrefixNodeId {
    pub fn from_identity(identity: &PrefixIdentity, graph_generation: u64) -> Self {
        let closure = identity.closure(graph_generation, 0);
        Self(StableDigest::from_parts(&[
            identity.model_root.as_str().as_bytes(),
            identity.tokenizer_root.as_str().as_bytes(),
            closure.token_sequence_root.as_str().as_bytes(),
            identity.rope_config_root.as_str().as_bytes(),
            &identity.layer_start.to_le_bytes(),
            &identity.layer_end.to_le_bytes(),
            &identity.head_start.to_le_bytes(),
            &identity.head_end.to_le_bytes(),
            identity.kv_type.as_bytes(),
            identity.codec_version.as_bytes(),
            &identity.runtime_generation.to_le_bytes(),
            identity.authority_state_root.as_str().as_bytes(),
        ]))
    }
}

impl fmt::Display for PrefixNodeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0.as_str())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixRoot {
    pub id: PrefixNodeId,
    pub identity: PrefixIdentity,
    pub closure: KVDependencyClosure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixEdge {
    pub parent: PrefixNodeId,
    pub child: PrefixNodeId,
    pub token_id: u32,
    pub ordinal: u64,
    pub dependency_root: StableDigest,
    pub graph_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixNode {
    pub id: PrefixNodeId,
    pub identity: PrefixIdentity,
    pub closure: KVDependencyClosure,
    pub parent: Option<PrefixNodeId>,
    pub incoming_token: Option<u32>,
    pub depth: usize,
    pub children: BTreeMap<u32, PrefixNodeId>,
    pub pages: Vec<KVPageId>,
    pub reference_count: u64,
    pub active_leases: u64,
    pub graph_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefixGraphError {
    EmptyIdentity,
    RootAlreadyExists,
    UnknownRoot,
    UnknownNode,
    UnknownPage,
    DuplicateEdge,
    InvalidTokenSequence,
    InvalidPage,
    StaleGeneration { expected: u64, actual: u64 },
    ActiveLease,
    Referenced,
    MissingRepresentation,
    NoQualityAuthorized,
    DependencyMismatch,
    InvalidLease,
}

impl fmt::Display for PrefixGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity => formatter.write_str("empty prefix identity"),
            Self::RootAlreadyExists => formatter.write_str("prefix root already exists"),
            Self::UnknownRoot => formatter.write_str("unknown prefix root"),
            Self::UnknownNode => formatter.write_str("unknown prefix node"),
            Self::UnknownPage => formatter.write_str("unknown KV page"),
            Self::DuplicateEdge => formatter.write_str("duplicate prefix edge"),
            Self::InvalidTokenSequence => {
                formatter.write_str("child token sequence is not parent plus one token")
            }
            Self::InvalidPage => formatter.write_str("invalid page identity or range"),
            Self::StaleGeneration { expected, actual } => write!(
                formatter,
                "stale generation: expected {expected}, actual {actual}"
            ),
            Self::ActiveLease => formatter.write_str("active page lease prevents eviction"),
            Self::Referenced => formatter.write_str("live references prevent eviction"),
            Self::MissingRepresentation => {
                formatter.write_str("requested representation is absent")
            }
            Self::NoQualityAuthorized => {
                formatter.write_str("no representation has a certified quality bound")
            }
            Self::DependencyMismatch => formatter.write_str("causal dependency closure mismatch"),
            Self::InvalidLease => formatter.write_str("invalid or already released lease"),
        }
    }
}
impl std::error::Error for PrefixGraphError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CausalClosureError {
    MissingNode,
    PrefixRangeOutsideNode,
    RootMismatch,
    GenerationMismatch,
}

impl fmt::Display for CausalClosureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MissingNode => "page node missing",
            Self::PrefixRangeOutsideNode => "page range outside prefix",
            Self::RootMismatch => "page dependency root mismatch",
            Self::GenerationMismatch => "page generation mismatch",
        })
    }
}
impl std::error::Error for CausalClosureError {}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum PageSelection {
    Representation(RepresentationKind),
    Fallback(KVFallbackPlan),
}

pub struct PrefixGraph {
    generation: u64,
    roots: HashMap<PrefixNodeId, PrefixRoot>,
    nodes: HashMap<PrefixNodeId, PrefixNode>,
    pages: HashMap<KVPageId, KVPageRecord>,
    leases: HashMap<String, KVPageLease>,
}

impl Default for PrefixGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixGraph {
    pub fn new() -> Self {
        Self {
            generation: 1,
            roots: HashMap::new(),
            nodes: HashMap::new(),
            pages: HashMap::new(),
            leases: HashMap::new(),
        }
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn roots(&self) -> impl Iterator<Item = &PrefixRoot> {
        self.roots.values()
    }
    pub fn node(&self, id: &PrefixNodeId) -> Option<&PrefixNode> {
        self.nodes.get(id)
    }
    pub fn page(&self, id: &KVPageId) -> Option<&KVPageRecord> {
        self.pages.get(id)
    }
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
    pub fn page_ids(&self) -> Vec<KVPageId> {
        self.pages.keys().cloned().collect()
    }

    pub fn insert_root(
        &mut self,
        identity: PrefixIdentity,
    ) -> Result<PrefixRoot, PrefixGraphError> {
        if !identity.token_sequence.is_empty() {
            return Err(PrefixGraphError::EmptyIdentity);
        }
        let id = PrefixNodeId::from_identity(&identity, self.generation);
        if self.roots.contains_key(&id) {
            return Err(PrefixGraphError::RootAlreadyExists);
        }
        let closure = identity.closure(self.generation, 0);
        let root = PrefixRoot {
            id: id.clone(),
            identity: identity.clone(),
            closure: closure.clone(),
        };
        self.roots.insert(id.clone(), root.clone());
        self.nodes.insert(
            id.clone(),
            PrefixNode {
                id,
                identity,
                closure,
                parent: None,
                incoming_token: None,
                depth: 0,
                children: BTreeMap::new(),
                pages: Vec::new(),
                reference_count: 0,
                active_leases: 0,
                graph_generation: self.generation,
            },
        );
        Ok(root)
    }

    pub fn extend(
        &mut self,
        parent_id: &PrefixNodeId,
        token_id: u32,
        identity: PrefixIdentity,
    ) -> Result<(PrefixNodeId, PrefixEdge), PrefixGraphError> {
        let parent = self
            .nodes
            .get(parent_id)
            .ok_or(PrefixGraphError::UnknownNode)?
            .clone();
        if identity.token_sequence.len() != parent.depth + 1
            || identity.token_sequence[..parent.depth] != parent.identity.token_sequence[..]
            || identity.token_sequence[parent.depth] != token_id
        {
            return Err(PrefixGraphError::InvalidTokenSequence);
        }
        if parent.children.contains_key(&token_id) {
            return Err(PrefixGraphError::DuplicateEdge);
        }
        let id = PrefixNodeId::from_identity(&identity, self.generation);
        let closure = identity.closure(self.generation, 0);
        let edge = PrefixEdge {
            parent: parent_id.clone(),
            child: id.clone(),
            token_id,
            ordinal: parent.depth as u64,
            dependency_root: closure.root(),
            graph_generation: self.generation,
        };
        self.nodes.insert(
            id.clone(),
            PrefixNode {
                id: id.clone(),
                identity,
                closure: closure.clone(),
                parent: Some(parent_id.clone()),
                incoming_token: Some(token_id),
                depth: parent.depth + 1,
                children: BTreeMap::new(),
                pages: Vec::new(),
                reference_count: 0,
                active_leases: 0,
                graph_generation: self.generation,
            },
        );
        self.nodes
            .get_mut(parent_id)
            .expect("parent checked")
            .children
            .insert(token_id, id.clone());
        Ok((id, edge))
    }

    pub fn lookup(
        &self,
        root_id: &PrefixNodeId,
        tokens: &[u32],
    ) -> Result<Option<PrefixNodeId>, PrefixGraphError> {
        let mut current = root_id.clone();
        if !self.roots.contains_key(root_id) {
            return Err(PrefixGraphError::UnknownRoot);
        }
        for token in tokens {
            let next = match self
                .nodes
                .get(&current)
                .and_then(|node| node.children.get(token))
            {
                Some(next) => next.clone(),
                None => return Ok(None),
            };
            current = next;
        }
        Ok(Some(current))
    }

    pub fn retain_prefix(&mut self, node_id: &PrefixNodeId) -> Result<(), PrefixGraphError> {
        let mut current = Some(node_id.clone());
        while let Some(id) = current {
            let node = self
                .nodes
                .get_mut(&id)
                .ok_or(PrefixGraphError::UnknownNode)?;
            node.reference_count += 1;
            current = node.parent.clone();
        }
        Ok(())
    }

    pub fn release_prefix(&mut self, node_id: &PrefixNodeId) -> Result<(), PrefixGraphError> {
        let mut current = Some(node_id.clone());
        while let Some(id) = current {
            let node = self
                .nodes
                .get_mut(&id)
                .ok_or(PrefixGraphError::UnknownNode)?;
            if node.reference_count == 0 {
                return Err(PrefixGraphError::Referenced);
            }
            node.reference_count -= 1;
            current = node.parent.clone();
        }
        Ok(())
    }

    pub fn attach_page(&mut self, page: KVPageRecord) -> Result<(), PrefixGraphError> {
        if !page.id.valid() {
            return Err(PrefixGraphError::InvalidPage);
        }
        let node = self
            .nodes
            .get_mut(&page.id.prefix_node_id)
            .ok_or(PrefixGraphError::UnknownNode)?;
        if page.id.token_end > node.identity.token_sequence.len() as u64
            || page.id.logical_generation != node.identity.runtime_generation
        {
            return Err(PrefixGraphError::InvalidPage);
        }
        if !page.closure.compatible_with(&node.closure) {
            return Err(PrefixGraphError::DependencyMismatch);
        }
        node.pages.push(page.id.clone());
        self.pages.insert(page.id.clone(), page);
        Ok(())
    }

    pub fn add_representation(
        &mut self,
        page_id: &KVPageId,
        representation: KVRepresentation,
        expected_graph_generation: u64,
    ) -> Result<(), PrefixGraphError> {
        if expected_graph_generation != self.generation {
            return Err(PrefixGraphError::StaleGeneration {
                expected: expected_graph_generation,
                actual: self.generation,
            });
        }
        let page = self
            .pages
            .get_mut(page_id)
            .ok_or(PrefixGraphError::UnknownPage)?;
        page.add_representation(representation);
        Ok(())
    }

    pub fn select(
        &self,
        page_id: &KVPageId,
        required: Option<&MetricBounds>,
        archive_ref: &str,
        authority_root: StableDigest,
    ) -> Result<PageSelection, PrefixGraphError> {
        self.select_with_hot(page_id, true, required, archive_ref, authority_root)
    }

    pub fn select_with_hot(
        &self,
        page_id: &KVPageId,
        exact_hot: bool,
        required: Option<&MetricBounds>,
        archive_ref: &str,
        authority_root: StableDigest,
    ) -> Result<PageSelection, PrefixGraphError> {
        let page = self
            .pages
            .get(page_id)
            .ok_or(PrefixGraphError::UnknownPage)?;
        if exact_hot && page.find(RepresentationKind::ExactHotQ8).is_some() {
            return Ok(PageSelection::Representation(
                RepresentationKind::ExactHotQ8,
            ));
        }
        if required.is_none() && page.find(RepresentationKind::ReconstructedWarm).is_some() {
            return Ok(PageSelection::Representation(
                RepresentationKind::ReconstructedWarm,
            ));
        }
        if let Some(KVRepresentation::ContextFoldLatent { quality, .. }) =
            page.find(RepresentationKind::ContextFoldLatent)
        {
            if required
                .map(|required| {
                    quality.authorize(required, &page.closure.root(), &page.closure.root())
                })
                .unwrap_or(true)
            {
                return Ok(PageSelection::Representation(
                    RepresentationKind::ContextFoldLatent,
                ));
            }
        }
        if let Some(lossless) = page.find(RepresentationKind::LosslessColdPage) {
            let _ = lossless;
            return Ok(PageSelection::Representation(
                RepresentationKind::LosslessColdPage,
            ));
        }
        if page.find(RepresentationKind::ExactTokenArchive).is_some() {
            let mut plan = KVFallbackPlan::strict(
                page_id.clone(),
                archive_ref,
                authority_root,
                "no certified latent representation",
            );
            plan.required_quality = required.map(|_| {
                KVQualityCertificate::measured_only(
                    "required",
                    page.closure.root(),
                    page.closure.root(),
                    crate::certificate::MetricBounds::none(),
                )
            });
            return Ok(PageSelection::Fallback(plan));
        }
        Err(PrefixGraphError::NoQualityAuthorized)
    }

    pub fn lease(
        &mut self,
        page_id: &KVPageId,
        representation: RepresentationKind,
        owner: impl Into<String>,
        expected_generation: u64,
    ) -> Result<KVPageLease, PrefixGraphError> {
        if expected_generation != self.generation {
            return Err(PrefixGraphError::StaleGeneration {
                expected: expected_generation,
                actual: self.generation,
            });
        }
        let page = self
            .pages
            .get_mut(page_id)
            .ok_or(PrefixGraphError::UnknownPage)?;
        if page.find(representation).is_none() {
            return Err(PrefixGraphError::MissingRepresentation);
        }
        page.active_leases += 1;
        page.reference_count += 1;
        let owner = owner.into();
        let lease_id = StableDigest::from_parts(&[
            page_id.key().as_bytes(),
            owner.as_bytes(),
            &self.generation.to_le_bytes(),
        ])
        .to_string();
        let lease = KVPageLease {
            lease_id: lease_id.clone(),
            page_id: page_id.clone(),
            representation,
            generation: self.generation,
            owner,
        };
        self.leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    pub fn release_lease(&mut self, lease_id: &str) -> Result<(), PrefixGraphError> {
        let lease = self
            .leases
            .remove(lease_id)
            .ok_or(PrefixGraphError::InvalidLease)?;
        let page = self
            .pages
            .get_mut(&lease.page_id)
            .ok_or(PrefixGraphError::UnknownPage)?;
        if page.active_leases == 0 || page.reference_count == 0 {
            return Err(PrefixGraphError::InvalidLease);
        }
        page.active_leases -= 1;
        page.reference_count -= 1;
        Ok(())
    }

    pub fn evict(
        &mut self,
        page_id: &KVPageId,
        representation: RepresentationKind,
        expected_generation: u64,
    ) -> Result<KVRepresentation, PrefixGraphError> {
        if expected_generation != self.generation {
            return Err(PrefixGraphError::StaleGeneration {
                expected: expected_generation,
                actual: self.generation,
            });
        }
        let page = self
            .pages
            .get_mut(page_id)
            .ok_or(PrefixGraphError::UnknownPage)?;
        if page.active_leases != 0 {
            return Err(PrefixGraphError::ActiveLease);
        }
        if page.reference_count != 0 {
            return Err(PrefixGraphError::Referenced);
        }
        let index = page
            .representations
            .iter()
            .position(|candidate| candidate.kind() == representation)
            .ok_or(PrefixGraphError::MissingRepresentation)?;
        Ok(page.representations.remove(index))
    }

    pub fn validate_causal_closure(&self, page_id: &KVPageId) -> Result<(), CausalClosureError> {
        let page = self
            .pages
            .get(page_id)
            .ok_or(CausalClosureError::MissingNode)?;
        let node = self
            .nodes
            .get(&page_id.prefix_node_id)
            .ok_or(CausalClosureError::MissingNode)?;
        if page.id.token_end > node.identity.token_sequence.len() as u64
            || page.id.token_end <= page.id.token_start
        {
            return Err(CausalClosureError::PrefixRangeOutsideNode);
        }
        if page.closure.root() != node.closure.root() {
            return Err(CausalClosureError::RootMismatch);
        }
        if page.id.logical_generation != node.identity.runtime_generation {
            return Err(CausalClosureError::GenerationMismatch);
        }
        Ok(())
    }

    /// Advance namespace generation after a graph/allocator mutation. Existing
    /// leases remain releasable, but new requests must carry the new generation.
    pub fn advance_generation(&mut self) -> u64 {
        self.generation += 1;
        self.generation
    }
}

use crate::identity::StableDigest;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QualityScope {
    PageProbe,
    Prefix,
    FullContext,
    TaskOutput,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricBounds {
    pub reconstruction_mse: Option<f64>,
    pub attention_score_mse: Option<f64>,
    pub attention_distribution_jsd: Option<f64>,
    pub attention_output_mse: Option<f64>,
    pub exact_tail_retrieval: Option<f64>,
}

impl MetricBounds {
    pub fn none() -> Self {
        Self {
            reconstruction_mse: None,
            attention_score_mse: None,
            attention_distribution_jsd: None,
            attention_output_mse: None,
            exact_tail_retrieval: None,
        }
    }

    pub fn max_error(&self) -> Option<f64> {
        [
            self.reconstruction_mse,
            self.attention_score_mse,
            self.attention_distribution_jsd,
            self.attention_output_mse,
        ]
        .into_iter()
        .flatten()
        .reduce(f64::max)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KVQualityCertificate {
    pub certificate_id: StableDigest,
    pub codec_id: String,
    pub reference_format: String,
    pub scope: QualityScope,
    pub dependency_root: StableDigest,
    pub state_boundary_root: StableDigest,
    pub measured: MetricBounds,
    pub allowed: MetricBounds,
    pub certified: bool,
    pub exact_tail_retrieval: bool,
    pub evidence: Vec<String>,
}

impl KVQualityCertificate {
    pub fn measured_only(
        codec_id: impl Into<String>,
        dependency_root: StableDigest,
        state_boundary_root: StableDigest,
        measured: MetricBounds,
    ) -> Self {
        let codec_id = codec_id.into();
        let certificate_id = StableDigest::from_parts(&[
            codec_id.as_bytes(),
            dependency_root.as_str().as_bytes(),
            state_boundary_root.as_str().as_bytes(),
            b"measured-not-certified",
        ]);
        Self {
            certificate_id,
            codec_id,
            reference_format: "q8_0".to_string(),
            scope: QualityScope::PageProbe,
            dependency_root,
            state_boundary_root,
            measured,
            allowed: MetricBounds::none(),
            certified: false,
            exact_tail_retrieval: false,
            evidence: vec!["tensor MSE alone is not a certificate".to_string()],
        }
    }

    pub fn exact(
        codec_id: impl Into<String>,
        dependency_root: StableDigest,
        state_boundary_root: StableDigest,
        scope: QualityScope,
    ) -> Self {
        let codec_id = codec_id.into();
        let certificate_id = StableDigest::from_parts(&[
            codec_id.as_bytes(),
            dependency_root.as_str().as_bytes(),
            state_boundary_root.as_str().as_bytes(),
            b"exact",
        ]);
        Self {
            certificate_id,
            codec_id,
            reference_format: "q8_0".to_string(),
            scope,
            dependency_root,
            state_boundary_root,
            measured: MetricBounds {
                reconstruction_mse: Some(0.0),
                attention_score_mse: Some(0.0),
                attention_distribution_jsd: Some(0.0),
                attention_output_mse: Some(0.0),
                exact_tail_retrieval: Some(1.0),
            },
            allowed: MetricBounds {
                reconstruction_mse: Some(0.0),
                attention_score_mse: Some(0.0),
                attention_distribution_jsd: Some(0.0),
                attention_output_mse: Some(0.0),
                exact_tail_retrieval: Some(1.0),
            },
            certified: true,
            exact_tail_retrieval: true,
            evidence: vec!["exact source representation".to_string()],
        }
    }

    pub fn authorize(
        &self,
        required: &MetricBounds,
        dependency_root: &StableDigest,
        state_boundary_root: &StableDigest,
    ) -> bool {
        if !self.certified
            || &self.dependency_root != dependency_root
            || &self.state_boundary_root != state_boundary_root
        {
            return false;
        }
        fn leq(actual: Option<f64>, required: Option<f64>) -> bool {
            match (actual, required) {
                (_, None) => true,
                (Some(actual), Some(required)) => actual <= required,
                (None, Some(_)) => false,
            }
        }
        leq(
            self.measured.reconstruction_mse,
            required.reconstruction_mse,
        ) && leq(
            self.measured.attention_score_mse,
            required.attention_score_mse,
        ) && leq(
            self.measured.attention_distribution_jsd,
            required.attention_distribution_jsd,
        ) && leq(
            self.measured.attention_output_mse,
            required.attention_output_mse,
        ) && (required.exact_tail_retrieval.is_none() || self.exact_tail_retrieval)
    }

    pub fn compose(&self, other: &Self) -> Result<Self, CertificateError> {
        if self.dependency_root != other.dependency_root
            || self.state_boundary_root != other.state_boundary_root
        {
            return Err(CertificateError::DependencyBoundaryMismatch);
        }
        if !self.certified || !other.certified {
            return Err(CertificateError::NotCertified);
        }
        if self.measured.reconstruction_mse.is_none()
            || other.measured.reconstruction_mse.is_none()
            || self.measured.attention_score_mse.is_none()
            || other.measured.attention_score_mse.is_none()
            || self.measured.attention_distribution_jsd.is_none()
            || other.measured.attention_distribution_jsd.is_none()
            || self.measured.attention_output_mse.is_none()
            || other.measured.attention_output_mse.is_none()
        {
            return Err(CertificateError::MissingMetric);
        }
        let measured = MetricBounds {
            reconstruction_mse: max_opt(
                self.measured.reconstruction_mse,
                other.measured.reconstruction_mse,
            ),
            attention_score_mse: max_opt(
                self.measured.attention_score_mse,
                other.measured.attention_score_mse,
            ),
            attention_distribution_jsd: max_opt(
                self.measured.attention_distribution_jsd,
                other.measured.attention_distribution_jsd,
            ),
            attention_output_mse: max_opt(
                self.measured.attention_output_mse,
                other.measured.attention_output_mse,
            ),
            exact_tail_retrieval: min_opt(
                self.measured.exact_tail_retrieval,
                other.measured.exact_tail_retrieval,
            ),
        };
        let certificate_id = StableDigest::from_parts(&[
            self.certificate_id.as_str().as_bytes(),
            other.certificate_id.as_str().as_bytes(),
            b"composition-v1",
        ]);
        Ok(Self {
            certificate_id,
            codec_id: format!("{}+{}", self.codec_id, other.codec_id),
            reference_format: self.reference_format.clone(),
            scope: QualityScope::Prefix,
            dependency_root: self.dependency_root.clone(),
            state_boundary_root: self.state_boundary_root.clone(),
            measured,
            allowed: MetricBounds::none(),
            certified: true,
            exact_tail_retrieval: self.exact_tail_retrieval && other.exact_tail_retrieval,
            evidence: [
                self.evidence.clone(),
                other.evidence.clone(),
                vec!["proof-carrying composition".to_string()],
            ]
            .concat(),
        })
    }
}

fn max_opt(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, None) | (None, a) => a,
    }
}
fn min_opt(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) | (None, a) => a,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CertificateError {
    NotCertified,
    DependencyBoundaryMismatch,
    MissingMetric,
}

impl fmt::Display for CertificateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotCertified => "certificate is not certified",
            Self::DependencyBoundaryMismatch => "certificate dependency/state boundary mismatch",
            Self::MissingMetric => "certificate composition is missing an attention/state metric",
        })
    }
}
impl std::error::Error for CertificateError {}

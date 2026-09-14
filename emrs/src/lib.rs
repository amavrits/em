mod logsumexp;
pub use logsumexp::row_logsumexp;

mod models;
pub use models::normal_logpdf;
pub use models::normal_mle;

mod em;
pub use crate::em::Em;

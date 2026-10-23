//! HTTP shard fetching shared by the dataset downloaders under `scripts/`.

use std::time::Duration;

use bytes::Bytes;

/// Output-file buffer capacity: 1 MB amortizes per-row writes over syscall boundaries.
pub const OUTPUT_BUFFER_BYTES: usize = 1 << 20;

/// Time limit for each HTTP request, shard downloads included.
pub const HTTP_REQUEST_TIME_LIMIT: Duration = Duration::from_secs(600);

/// Attempts per shard before a failure ends the run.
const SHARD_ATTEMPTS: u32 = 5;

/// Delay before the first retry, doubled for each one after it.
const SHARD_RETRY_DELAY: Duration = Duration::from_secs(2);

/// An HTTP client whose requests give up after [`HTTP_REQUEST_TIME_LIMIT`].
pub fn http_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder().timeout(HTTP_REQUEST_TIME_LIMIT).build()
}

/// Fetches one shard, retrying failures with doubling delays. Returns `None` only when every
/// attempt answered 403 or 404, which a bucket's ACL uses past its last shard.
pub async fn download_shard(client: &reqwest::Client, url: &str) -> Result<Option<Bytes>, String> {
    let mut missing_answers = 0;
    let mut last_failure = String::new();
    for attempt in 0..SHARD_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(SHARD_RETRY_DELAY * (1 << (attempt - 1))).await;
        }
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => match response.bytes().await {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(error) => last_failure = format!("GET {url}: {error}"),
            },
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::FORBIDDEN {
                    missing_answers += 1;
                }
                last_failure = format!("GET {url} -> {status}");
            }
            Err(error) => last_failure = format!("GET {url}: {error}"),
        }
    }
    match missing_answers == SHARD_ATTEMPTS {
        true => Ok(None),
        false => Err(last_failure),
    }
}

//! Docker container lifecycle management for Tier 2 backends.

use std::{collections::HashMap, time::Duration};

use bollard::{
    models::{ContainerCreateBody, HostConfig, PortBinding},
    Docker,
};
use futures_util::StreamExt;

pub struct ContainerHandle {
    docker: Docker,
    container_id: String,
    name: String,
}

impl ContainerHandle {
    pub async fn start(
        image: &str,
        container_name: &str,
        ports: &[(u16, u16)],
        env: &[String],
        command: &[String],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let docker = Docker::connect_with_local_defaults()?;

        eprintln!("  Pulling image {image}...");
        let pull_opts = bollard::query_parameters::CreateImageOptionsBuilder::default()
            .from_image(image)
            .build();
        let mut pull_stream = docker.create_image(Some(pull_opts), None, None);
        while let Some(result) = pull_stream.next().await {
            if let Err(error) = result {
                eprintln!("  Pull warning: {error}");
            }
        }

        let remove_opts = bollard::query_parameters::RemoveContainerOptionsBuilder::default()
            .force(true)
            .build();
        let _ = docker.remove_container(container_name, Some(remove_opts)).await;

        let mut port_bindings: HashMap<String, Option<Vec<PortBinding>>> = HashMap::new();
        for &(host_port, container_port) in ports {
            port_bindings.insert(
                format!("{container_port}/tcp"),
                Some(vec![PortBinding {
                    host_ip: Some("127.0.0.1".to_string()),
                    host_port: Some(host_port.to_string()),
                }]),
            );
        }

        let budget = crate::output::budget();
        let memory = i64::try_from(budget.memory_limit).unwrap_or(i64::MAX);
        let config = ContainerCreateBody {
            image: Some(image.to_string()),
            host_config: Some(HostConfig {
                port_bindings: Some(port_bindings),
                cpuset_cpus: Some(budget.cpus.clone()),
                memory: Some(memory),
                memory_swap: Some(memory),
                ..Default::default()
            }),
            env: Some(env.to_vec()),
            cmd: (!command.is_empty()).then(|| command.to_vec()),
            ..Default::default()
        };

        let create_opts = bollard::query_parameters::CreateContainerOptionsBuilder::default()
            .name(container_name)
            .build();

        eprintln!("  Creating container {container_name}...");
        let response = docker.create_container(Some(create_opts), config).await?;
        let container_id = response.id;

        eprintln!("  Starting container {container_name}...");
        docker.start_container(&container_id, None).await?;

        Ok(Self {
            docker,
            container_id,
            name: container_name.to_string(),
        })
    }

    /// Poll `check` every 500ms until it returns `true` or `timeout` expires.
    async fn wait_until<F: std::future::Future<Output = bool>>(
        &self,
        label: &str,
        timeout: Duration,
        check: impl Fn() -> F,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let deadline = tokio::time::Instant::now() + timeout;
        eprintln!("  Waiting for {label}...");
        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(format!("timeout waiting for {label}").into());
            }
            if check().await {
                eprintln!("  {} is ready.", self.name);
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    pub async fn wait_for_http(&self, url: &str, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        self.wait_until(url, timeout, || {
            let client = client.clone();
            let url = url.to_string();
            async move {
                client
                    .get(&url)
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success())
            }
        })
        .await
    }

    /// The container's anonymous resident bytes from its cgroup's `memory.stat`, leaving out the page
    /// cache that Docker's `usage` counts.
    pub async fn memory_usage_bytes(&self) -> u64 {
        let opts = bollard::query_parameters::StatsOptionsBuilder::default()
            .stream(false)
            .one_shot(true)
            .build();
        let mut stream = self.docker.stats(&self.container_id, Some(opts));
        let Some(Ok(stats)) = stream.next().await else {
            return 0;
        };
        let stats = stats.memory_stats.and_then(|memory| memory.stats).unwrap_or_default();
        stats.get("anon").or_else(|| stats.get("rss")).copied().unwrap_or(0)
    }

    /// The memory, swap and cpuset limits Docker applied, read back from the container's `HostConfig`.
    pub async fn limits(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let inspected = self.docker.inspect_container(&self.container_id, None).await?;
        let host = inspected.host_config.unwrap_or_default();
        Ok(serde_json::json!({
            "memory": host.memory,
            "memory_swap": host.memory_swap,
            "cpuset_cpus": host.cpuset_cpus,
        }))
    }

    pub async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("  Stopping container {}...", self.name);
        let _ = self.docker.stop_container(&self.container_id, None).await;
        let remove_opts = bollard::query_parameters::RemoveContainerOptionsBuilder::default()
            .force(true)
            .build();
        self.docker
            .remove_container(&self.container_id, Some(remove_opts))
            .await?;
        eprintln!("  Container {} removed.", self.name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ContainerHandle;

    /// Marks the run inside the scope, so the test re-executes itself only once.
    const INSIDE_SCOPE: &str = "RETRIEVAL_BUDGET_SCOPE";

    /// Inside a two-core 512 MiB scope, the budget reads the scope back and a server container is
    /// created with that cpuset and memory, as its `HostConfig` reports them.
    #[test]
    #[ignore = "spawns systemd-run --user and pulls the Qdrant image"]
    fn containers_take_the_scope_budget() {
        if std::env::var_os(INSIDE_SCOPE).is_none() {
            let status = std::process::Command::new("systemd-run")
                .args(["--user", "--scope", "--quiet", "--collect"])
                .arg(format!("--unit=retrieval-budget-{}", std::process::id()))
                .args(["-p", "MemoryMax=512M", "-p", "MemorySwapMax=0", "taskset", "-c", "0-1"])
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "docker::tests::containers_take_the_scope_budget",
                    "--include-ignored",
                ])
                .env(INSIDE_SCOPE, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let budget = crate::output::budget();
        assert_eq!(
            (budget.cores, budget.cpus.as_str(), budget.memory_limit),
            (2, "0-1", 512 << 20)
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let container = runtime
            .block_on(ContainerHandle::start(
                "qdrant/qdrant:v1.19.1",
                "retrieval-budget-test",
                &[],
                &[],
                &[],
            ))
            .unwrap();
        let limits = runtime.block_on(container.limits()).map_err(|error| error.to_string());
        runtime.block_on(container.stop()).unwrap();
        let expected = serde_json::json!({"memory": 512 << 20, "memory_swap": 512 << 20, "cpuset_cpus": "0-1"});
        assert_eq!(limits.unwrap(), expected);
    }
}

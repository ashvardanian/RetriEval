//! Docker container lifecycle management for Tier 2 backends.

use std::collections::HashMap;
use std::time::Duration;

use bollard::models::*;
use bollard::Docker;
use futures_util::StreamExt;

pub struct ContainerHandle {
    docker: Docker,
    container_id: String,
    name: String,
}

pub type PortMap = Vec<(u16, u16)>;

impl ContainerHandle {
    pub async fn start(
        image: &str,
        container_name: &str,
        ports: &PortMap,
        env: &[String],
        _timeout: Duration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let docker = Docker::connect_with_local_defaults()?;

        eprintln!("  Pulling image {image}...");
        let pull_opts = bollard::query_parameters::CreateImageOptionsBuilder::default()
            .from_image(image)
            .build();
        let mut pull_stream = docker.create_image(Some(pull_opts), None, None);
        while let Some(result) = pull_stream.next().await {
            if let Err(e) = result {
                eprintln!("  Pull warning: {e}");
            }
        }

        let remove_opts = bollard::query_parameters::RemoveContainerOptionsBuilder::default()
            .force(true)
            .build();
        let _ = docker
            .remove_container(container_name, Some(remove_opts))
            .await;

        let mut port_bindings: HashMap<String, Option<Vec<PortBinding>>> = HashMap::new();
        for &(host_port, container_port) in ports {
            port_bindings.insert(
                format!("{container_port}/tcp"),
                Some(vec![PortBinding {
                    host_ip: Some("0.0.0.0".to_string()),
                    host_port: Some(host_port.to_string()),
                }]),
            );
        }

        let config = ContainerCreateBody {
            image: Some(image.to_string()),
            host_config: Some(HostConfig {
                port_bindings: Some(port_bindings),
                ..Default::default()
            }),
            env: Some(env.to_vec()),
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
    async fn wait_until(
        &self,
        label: &str,
        timeout: Duration,
        check: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = bool>>>,
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

    pub async fn wait_for_http(
        &self,
        url: &str,
        timeout: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        self.wait_until(url, timeout, || {
            let client = client.clone();
            let url = url.to_string();
            Box::pin(async move {
                client
                    .get(&url)
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success())
            })
        })
        .await
    }

    pub async fn wait_for_tcp(
        &self,
        host: &str,
        port: u16,
        timeout: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let address = format!("{host}:{port}");
        self.wait_until(&address, timeout, || {
            let address = address.clone();
            Box::pin(async move { tokio::net::TcpStream::connect(&address).await.is_ok() })
        })
        .await
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

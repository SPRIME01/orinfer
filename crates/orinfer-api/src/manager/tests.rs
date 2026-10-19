use super::*;
// Lifecycle scenarios adapted from Ollama; provenance and MIT notice are in fixtures/ollama.
mod lifecycle;
mod protocol;

async fn eventually(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("test condition did not become true");
}

#[test]
fn profile_identity_ignores_alias_and_sampling_defaults() {
    let mut record = ModelRecord {
        name: "one".into(),
        path: "/model".into(),
        identity: "weights".into(),
        architecture: "test".into(),
        execution_package: "package".into(),
        weight_bytes: 1,
        load_bytes: 2,
        max_context: 8192,
        sequence_bytes: 0,
        kv_bytes_per_token: 0,
        options: WorkerOptions::default(),
    };
    let key = runner_key(&record);
    record.name = "two".into();
    record
        .options
        .default_request_params
        .insert("temperature".into(), json!(0));
    assert_eq!(key, runner_key(&record));
    record.options.mtp_drafts = "0".into();
    assert_ne!(key, runner_key(&record));
}

#[test]
fn unified_memory_is_read_without_initializing_cuda() {
    assert!(available_bytes().unwrap() > 0);
}

struct Fixture {
    root: PathBuf,
    manager: Arc<Manager>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Fixture {
    fn new() -> Self {
        Self::with_config(ManagerConfig::default())
    }
    fn with_config(mut config: ManagerConfig) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "orinfer-manager-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        private_directory(&root).unwrap();
        private_directory(&root.join("cache")).unwrap();
        fs::write(root.join("cache/model.json"), b"fixture").unwrap();
        let executable = root.join("worker.py");
        fs::write(&executable, include_str!("tests/worker.py")).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        let record = ModelRecord {
            name: "fixture:2b".into(),
            path: root.clone(),
            identity: orinfer_engine::artifact::sha256(b"fixture"),
            architecture: "test".into(),
            execution_package: "package".into(),
            weight_bytes: 1,
            load_bytes: 1,
            max_context: 8192,
            sequence_bytes: 0,
            kv_bytes_per_token: 0,
            options: WorkerOptions::default(),
        };
        config.data_dir = root.clone();
        config.gpu_lock = root.join("shared-gpu.lock");
        config.load_timeout_s = 5;
        let manager = Arc::new(Manager {
            config,
            directory: root.clone(),
            executable,
            records: Mutex::new(BTreeMap::from([(record.name.clone(), record)])),
            runners: Mutex::new(BTreeMap::new()),
            failures: Mutex::new(BTreeMap::new()),
            transition: Semaphore::new(1),
            requests: Arc::new(Semaphore::new(128)),
            body_memory: Arc::new(Semaphore::new(512)),
            request_budget: Arc::new(SyncMutex::new(0)),
            gpu_owner: SyncMutex::new(None),
            stopping: AtomicBool::new(false),
        });
        Self { root, manager }
    }
    async fn alias(&self, name: &str, change: impl FnOnce(&mut ModelRecord)) {
        let mut records = self.manager.records.lock().await;
        let mut record = records["fixture:2b"].clone();
        record.name = name.into();
        change(&mut record);
        records.insert(name.into(), record);
    }
    fn starts(&self) -> usize {
        fs::read_to_string(self.root.join("starts"))
            .unwrap_or_default()
            .lines()
            .count()
    }
    async fn unload(&self) {
        let _transition = self.manager.transition.acquire().await.unwrap();
        let runners: Vec<_> = self
            .manager
            .runners
            .lock()
            .await
            .values()
            .cloned()
            .collect();
        for runner in runners {
            self.manager.remove_runner(&runner).await.unwrap();
        }
    }
    async fn assert_gpu_released(&self) {
        assert!(self.manager.gpu_owner.lock().unwrap().is_none());
        // Parallel tests may fork with this CLOEXEC descriptor just before its
        // owner drops it. Allow that unrelated child to complete exec/close.
        eventually(|| locked_file(&self.manager.config.gpu_lock).is_ok()).await;
    }
}

#[tokio::test]
async fn concurrent_loads_share_worker_and_idle_cleanup_releases_ownership() {
    let fixture = Fixture::new();
    let loads = (0..8).map(|_| {
        let manager = Arc::clone(&fixture.manager);
        async move { manager.acquire("fixture:2b").await.unwrap().0 }
    });
    let leases = futures_util::future::join_all(loads).await;
    let pid = leases[0].0.pid;
    assert!(leases.iter().all(|lease| lease.0.pid == pid));
    assert_eq!(
        fs::read_to_string(fixture.root.join("starts"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(locked_file(&fixture.manager.config.gpu_lock).is_err());
    leases[0].0.keep_alive_s.store(0, Ordering::Relaxed);
    fixture.manager.sweep().await;
    assert_eq!(fixture.manager.runners.lock().await.len(), 1);
    drop(leases);
    fixture.manager.sweep().await;
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn dead_worker_is_restarted_and_failed_startup_does_not_hold_gpu_lock() {
    let fixture = Fixture::new();
    let (lease, _) = fixture.manager.acquire("fixture:2b").await.unwrap();
    let old = lease.0.pid;
    lease.0.child.lock().await.kill().await.unwrap();
    let (replacement, _) = fixture.manager.acquire("fixture:2b").await.unwrap();
    assert_ne!(old, replacement.0.pid);
    fixture.manager.remove_runner(&lease.0).await.unwrap();
    assert_eq!(fixture.manager.runners.lock().await.len(), 1);
    drop(lease);
    drop(replacement);
    fixture.manager.stop("fixture:2b", false).await.unwrap();
    fs::write(&fixture.manager.executable, "#!/bin/sh\nexit 2\n").unwrap();
    assert!(fixture.manager.acquire("fixture:2b").await.is_err());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn cancellation_during_loading_does_not_orphan_a_worker() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("hold_start"), "").unwrap();
    let manager = Arc::clone(&fixture.manager);
    let caller = tokio::spawn(async move { manager.acquire("fixture:2b").await });
    eventually(|| fixture.root.join("starts").exists()).await;
    caller.abort();
    assert!(matches!(caller.await, Err(error) if error.is_cancelled()));
    fs::remove_file(fixture.root.join("hold_start")).unwrap();
    let deadline = Instant::now();
    loop {
        if !fixture.manager.runners.lock().await.is_empty() {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(5));
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    fixture.manager.stop("fixture:2b", false).await.unwrap();
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[test]
fn registry_write_is_atomic_and_roundtrips_existing_profiles() {
    let fixture = Fixture::new();
    let records = fixture.manager.records.blocking_lock().clone();
    registry::write(&fixture.root, &records).unwrap();
    assert_eq!(
        registry::read(&fixture.root).unwrap()["fixture:2b"].identity,
        records["fixture:2b"].identity
    );
    assert!(!fixture.root.join("models.json.next").exists());
}

#[test]
fn request_budgets_are_global_and_released_on_cancellation() {
    let fixture = Fixture::new();
    let mut record = fixture.manager.records.blocking_lock()["fixture:2b"].clone();
    record.sequence_bytes = 1024;
    record.kv_bytes_per_token = 128;
    let budget = fixture
        .manager
        .reserve_request(&record, &json!({"max_tokens":64}), 32)
        .unwrap();
    assert_eq!(
        *fixture.manager.request_budget.lock().unwrap(),
        1024 + 96 * 128
    );
    drop(budget);
    assert_eq!(*fixture.manager.request_budget.lock().unwrap(), 0);
    record.sequence_bytes = usize::MAX;
    assert!(
        fixture
            .manager
            .reserve_request(&record, &json!({}), 32)
            .is_err()
    );
}

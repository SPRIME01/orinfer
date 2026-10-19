use super::*;

#[tokio::test]
async fn cancellation_while_waiting_for_startup_serialization_never_spawns_a_worker() {
    let fixture = Fixture::new();
    let transition = fixture.manager.transition.acquire().await.unwrap();
    let manager = Arc::clone(&fixture.manager);
    let task = tokio::spawn(async move { manager.acquire("fixture:2b").await });
    tokio::task::yield_now().await;
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    drop(transition);
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(fixture.starts(), 0);
    assert!(fixture.manager.runners.lock().await.is_empty());
}

#[tokio::test]
async fn aliases_share_loaded_workers_but_execution_profiles_are_isolated() {
    let fixture = Fixture::new();
    fixture
        .alias("alias", |record| {
            record
                .options
                .default_request_params
                .insert("temperature".into(), json!(0));
        })
        .await;
    fixture
        .alias("other-profile", |record| {
            record.options.mtp_drafts = "0".into()
        })
        .await;
    let first = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let alias = fixture.manager.acquire("alias").await.unwrap().0;
    let other = fixture.manager.acquire("other-profile").await.unwrap().0;
    assert_eq!(first.0.pid, alias.0.pid);
    assert_ne!(first.0.pid, other.0.pid);
    assert_eq!(fixture.starts(), 2);
    drop((first, alias, other));
    fixture.unload().await;
}

#[tokio::test]
async fn expired_idle_timestamp_cannot_unload_a_new_active_request() {
    let fixture = Fixture::new();
    let first = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let runner = Arc::clone(&first.0);
    drop(first);
    runner.keep_alive_s.store(1, Ordering::Release);
    *runner.last_used.lock().unwrap() = Instant::now() - Duration::from_secs(60);
    let fresh = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    fixture.manager.sweep().await;
    assert_eq!(fixture.manager.runners.lock().await.len(), 1);
    assert_eq!(fresh.0.pid, runner.pid);
    drop(fresh);
    assert!(runner.last_used.lock().unwrap().elapsed() < Duration::from_secs(1));
    fixture.manager.sweep().await;
    assert_eq!(fixture.manager.runners.lock().await.len(), 1);
    *runner.last_used.lock().unwrap() = Instant::now() - Duration::from_secs(60);
    fixture.manager.sweep().await;
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn cancelling_capacity_waiter_does_not_load_it_after_space_becomes_available() {
    let fixture = Fixture::with_config(ManagerConfig {
        max_loaded_models: Some(1),
        ..Default::default()
    });
    fixture
        .alias("waiting", |record| record.options.mtp_drafts = "0".into())
        .await;
    let active = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let manager = Arc::clone(&fixture.manager);
    let waiting = tokio::spawn(async move { manager.acquire("waiting").await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!waiting.is_finished());
    waiting.abort();
    let _ = waiting.await;
    drop(active);
    // Let the capacity retry run if it wrongly survived its cancelled caller.
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(fixture.starts(), 1);
    fixture.unload().await;
}

#[tokio::test]
async fn concurrent_startup_failure_is_shared_and_later_retry_recovers() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("fail_start"), "").unwrap();
    let attempts = (0..8).map(|_| fixture.manager.acquire("fixture:2b"));
    let errors: Vec<_> = futures_util::future::join_all(attempts)
        .await
        .into_iter()
        .map(|result| result.err().unwrap())
        .collect();
    assert!(errors.iter().all(|message| message == &errors[0]));
    assert_eq!(fixture.starts(), 1);
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
    fs::remove_file(fixture.root.join("fail_start")).unwrap();
    for (at, _) in fixture.manager.failures.lock().await.values_mut() {
        *at = Instant::now() - Duration::from_secs(11);
    }
    let loaded = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    assert_eq!(fixture.starts(), 2);
    drop(loaded);
    fixture.unload().await;
}

#[tokio::test]
async fn memory_pressure_evicts_oldest_idle_worker_and_preserves_active_worker() {
    let fixture = Fixture::with_config(ManagerConfig {
        max_loaded_models: Some(2),
        ..Default::default()
    });
    fixture
        .alias("second", |record| record.options.mtp_drafts = "0".into())
        .await;
    fixture
        .alias("third", |record| record.options.cuda_graph = "off".into())
        .await;
    let first = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let second = fixture.manager.acquire("second").await.unwrap().0;
    let key = second.0.key.clone();
    drop(second);
    let third = fixture.manager.acquire("third").await.unwrap().0;
    assert!(!fixture.manager.runners.lock().await.contains_key(&key));
    assert!(
        fixture
            .manager
            .runners
            .lock()
            .await
            .contains_key(&first.0.key)
    );
    drop(first);
    let second = fixture.manager.acquire("second").await.unwrap().0;
    assert!(
        fixture
            .manager
            .runners
            .lock()
            .await
            .contains_key(&third.0.key)
    );
    drop((second, third));
    fixture.unload().await;
}

#[tokio::test]
async fn capacity_eviction_uses_idle_lru_order() {
    let fixture = Fixture::with_config(ManagerConfig {
        max_loaded_models: Some(2),
        ..Default::default()
    });
    fixture
        .alias("second", |record| record.options.mtp_drafts = "0".into())
        .await;
    fixture
        .alias("third", |record| record.options.cuda_graph = "off".into())
        .await;
    let first = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let second = fixture.manager.acquire("second").await.unwrap().0;
    let oldest = Arc::clone(&first.0);
    let newest = Arc::clone(&second.0);
    drop((first, second));
    *oldest.last_used.lock().unwrap() = Instant::now() - Duration::from_secs(60);
    let third = fixture.manager.acquire("third").await.unwrap().0;
    assert!(
        !fixture
            .manager
            .runners
            .lock()
            .await
            .contains_key(&oldest.key)
    );
    assert!(
        fixture
            .manager
            .runners
            .lock()
            .await
            .contains_key(&newest.key)
    );
    assert!(oldest.child.lock().await.try_wait().unwrap().is_some());
    drop(third);
    fixture.unload().await;
}

#[tokio::test]
async fn capacity_waiter_proceeds_after_active_request_finishes() {
    let fixture = Fixture::with_config(ManagerConfig {
        max_loaded_models: Some(1),
        ..Default::default()
    });
    fixture
        .alias("waiting", |record| record.options.mtp_drafts = "0".into())
        .await;
    let active = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let manager = Arc::clone(&fixture.manager);
    let waiting = tokio::spawn(async move { manager.acquire("waiting").await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!waiting.is_finished());
    assert_eq!(fixture.starts(), 1);
    drop(active);
    let (next, _) = tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(next.0.model, "waiting");
    assert_eq!(fixture.starts(), 2);
    drop(next);
    fixture.unload().await;
}

#[tokio::test]
async fn graceful_stop_drains_active_requests_before_unloading() {
    let fixture = Fixture::new();
    let lease = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    let runner = Arc::clone(&lease.0);
    let manager = Arc::clone(&fixture.manager);
    let stop = tokio::spawn(async move { manager.stop("fixture:2b", false).await });
    eventually(|| runner.draining.load(Ordering::Acquire)).await;
    assert!(!stop.is_finished());
    assert!(runner.child.lock().await.try_wait().unwrap().is_none());
    drop(lease);
    tokio::time::timeout(Duration::from_secs(5), stop)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn enough_memory_allows_parallel_cold_loads_and_unblocked_ready_models() {
    let fixture = Fixture::new();
    fixture
        .alias("second", |r| r.options.mtp_drafts = "0".into())
        .await;
    fixture
        .alias("third", |r| r.options.cuda_graph = "off".into())
        .await;
    let first = fixture.manager.acquire("fixture:2b").await.unwrap().0;
    fs::write(fixture.root.join("hold_start"), "").unwrap();
    let manager = Arc::clone(&fixture.manager);
    let second = tokio::spawn(async move { manager.acquire("second").await });
    eventually(|| fixture.starts() == 2).await;
    let manager = Arc::clone(&fixture.manager);
    let third = tokio::spawn(async move { manager.acquire("third").await });
    eventually(|| fixture.starts() == 3).await;
    assert!(!second.is_finished() && !third.is_finished());
    assert_eq!(fixture.manager.budgets.lock().unwrap().loading, 2);
    let ready = tokio::time::timeout(
        Duration::from_millis(250),
        fixture.manager.acquire("fixture:2b"),
    )
    .await
    .expect("A ready model waited for another model's cold load")
    .unwrap()
    .0;
    assert_eq!(first.0.pid, ready.0.pid);
    fs::remove_file(fixture.root.join("hold_start")).unwrap();
    let second = second.await.unwrap().unwrap().0;
    let third = third.await.unwrap().unwrap().0;
    assert_eq!(fixture.manager.budgets.lock().unwrap().loading, 0);
    drop((first, ready, second, third));
    fixture.unload().await;
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn fitting_http_requests_bypass_an_unrelated_lifecycle_transition() {
    let fixture = Fixture::new();
    drop(fixture.manager.acquire("fixture:2b").await.unwrap().0);
    let http = super::protocol::Http::new(Arc::clone(&fixture.manager)).await;
    let transition = fixture.manager.transition.acquire().await.unwrap();
    let response = tokio::time::timeout(
        Duration::from_millis(250),
        http.post(&json!({
            "model":"fixture:2b", "messages":[{"role":"user","content":"hello"}], "max_tokens":8
        })),
    )
    .await
    .expect("A fitting request waited for the lifecycle permit");
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.json::<Value>().await.unwrap()["model"],
        "fixture:2b"
    );
    drop(transition);
    fixture.unload().await;
}

#[tokio::test]
async fn request_memory_pressure_evicts_idle_workers_even_with_forever_keep_alive() {
    let fixture = Fixture::new();
    fixture
        .alias("idle", |r| r.options.mtp_drafts = "0".into())
        .await;
    let (active, mut record) = fixture.manager.acquire("fixture:2b").await.unwrap();
    let idle = fixture.manager.acquire("idle").await.unwrap().0;
    let victim = Arc::clone(&idle.0);
    victim.keep_alive_s.store(u64::MAX, Ordering::Release);
    drop(idle);
    // Impossible to satisfy without allocating physical RAM: exercise eviction
    // and the remaining active-work wait boundary deterministically on CPU.
    record.sequence_bytes = usize::MAX / 2;
    let result = fixture
        .manager
        .reserve_request_with_eviction(&record, &json!({}), 32)
        .await;
    assert!(
        result
            .err()
            .unwrap()
            .starts_with("Insufficient unified memory")
    );
    assert!(victim.child.lock().await.try_wait().unwrap().is_some());
    assert!(active.0.child.lock().await.try_wait().unwrap().is_none());
    assert_eq!(fixture.manager.runners.lock().await.len(), 1);
    assert_eq!(fixture.manager.budgets.lock().unwrap().requests, 0);
    drop(active);
    fixture.unload().await;
}

#[tokio::test]
async fn pending_load_reservations_prevent_double_promising_memory_and_release_on_drop() {
    let fixture = Fixture::new();
    let (_, mut record) = fixture.manager.acquire("fixture:2b").await.unwrap();
    record.sequence_bytes = 1024;
    let reserved = usize::MAX / 2;
    fixture.manager.budgets.lock().unwrap().loading = reserved;
    let budget = MemoryBudget {
        bytes: reserved,
        kind: BudgetKind::Loading,
        total: Arc::clone(&fixture.manager.budgets),
    };
    assert!(
        fixture
            .manager
            .reserve_request(&record, &json!({}), 32)
            .is_err()
    );
    drop(budget);
    let request = fixture
        .manager
        .reserve_request(&record, &json!({}), 32)
        .unwrap();
    drop(request);
    fixture.unload().await;
}

#[tokio::test]
async fn claimed_idle_victim_cannot_accept_a_racing_request() {
    let fixture = Fixture::new();
    drop(fixture.manager.acquire("fixture:2b").await.unwrap().0);
    let transition = fixture.manager.transition.acquire().await.unwrap();
    let victim = fixture.manager.claim_idle().await.unwrap();
    let manager = Arc::clone(&fixture.manager);
    let arriving = tokio::spawn(async move { manager.acquire("fixture:2b").await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!arriving.is_finished());
    assert_eq!(victim.active.load(Ordering::Acquire), 0);
    fixture.manager.remove_runner(&victim).await.unwrap();
    drop(transition);
    let replacement = arriving.await.unwrap().unwrap().0;
    assert_ne!(victim.pid, replacement.0.pid);
    drop(replacement);
    fixture.unload().await;
}

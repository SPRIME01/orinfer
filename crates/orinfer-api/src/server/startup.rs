//! Shared model loading and GPU worker lifecycle for HTTP and offline jobs.
use super::*;

pub(super) struct Worker {
    pub state: Service,
    pub shutdown: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}
impl Worker {
    pub async fn stop(self, timeout_ms: u64) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        // Closing the last ingress wakes an idle blocking receiver.
        drop(self.state);
        if !wait_worker(&self.handle, timeout_ms).await {
            return Err("GPU worker did not exit before drain deadline".into());
        }
        self.handle
            .join()
            .map_err(|_| "GPU worker panicked".to_owned())
    }
}

pub(super) async fn start(settings: &Settings) -> Result<Worker> {
    let settings = settings.clone();
    let codec = Arc::new(ChatCodec::load(&settings.model_dir)?);
    if !settings.default_request_params.is_empty() {
        codec
            .prepare(
                defaults::validation_request(&settings.default_request_params)?,
                "defaults-validation",
                usize::MAX,
                None,
                &mut preparation::Context::new(
                    Arc::new(AtomicBool::new(false)),
                    settings.limits.memory_mib * 1024 * 1024,
                ),
            )
            .map_err(|e| format!("default_request_params: {e}"))?;
    }
    let default_request_params = Arc::new(settings.default_request_params.clone());
    let (sender, receiver) = mpsc::channel(128);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let model_id: Arc<str> = settings.model.clone().into();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_codec = Arc::clone(&codec);
    let worker_model = Arc::clone(&model_id);
    let worker_shutdown = Arc::clone(&shutdown);
    let scheduler = settings.scheduler;
    let verify_weights = settings.verify_weights;
    let resident_experts = settings.resident_experts;
    let load_workers = settings.load_workers;
    let prompt_lookup = settings.prompt_lookup;
    let activity = Arc::new(continuous::Activity::default());
    let lifecycle = Arc::clone(&activity.lifecycle);
    let worker_activity = Arc::clone(&activity);
    let worker_lifecycle = Arc::clone(&lifecycle);
    let limits = settings.limits;
    let worker = std::thread::Builder::new()
        .name("orinfer-gpu".into())
        .spawn(move || {
            let _ = worker_lifecycle.supervise(|| {
                let initialize = || -> Result<_> {
                    use std::os::fd::AsRawFd;
                    if let Some(parent) = settings
                        .gpu_lock
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                    {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    let lock = std::fs::OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .read(true)
                        .write(true)
                        .open(&settings.gpu_lock)
                        .map_err(|e| e.to_string())?;
                    // SAFETY: flock borrows this live file descriptor. The file remains
                    // owned by this worker for the complete CUDA model lifetime.
                    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0
                    {
                        return Err("GPU experiment lock is busy".into());
                    }
                    let model = Model::load_with_options(
                        &settings.model_dir,
                        LoadOptions {
                            cuda_graph: settings.cuda_graph,
                            verify_weights,
                            load_workers,
                            resident_experts: settings.resident_experts,
                            prefix_cache_bytes: settings.prefix_cache_bytes,
                            mtp_drafts: settings.mtp_drafts,
                            prompt_lookup: settings.prompt_lookup,
                        },
                    )?;
                    if &worker_codec.asset_hashes != model.frontend_assets() {
                        return Err("Frontend assets changed while loading model".into());
                    }
                    if worker_codec.tokenizer.get_vocab_size(true) > model.vocab() {
                        return Err("Tokenizer exceeds model vocabulary".into());
                    }
                    let context = model.max_context();
                    let vision = model.vision().cloned();
                    if !model.batching_supported() && scheduler.max_active != 1 {
                        return Err("This execution package schedules one active request; set --max-active-requests 1 (additional requests queue)".into());
                    }
                    Ok((lock, model, context, vision))
                };
                match initialize() {
                    Ok((_lock, mut model, context, vision)) => {
                        worker_lifecycle.ready();
                        let mtp_drafts = model.mtp_drafts();
                        if ready_sender.send(Ok((context, vision, mtp_drafts))).is_ok() {
                            continuous::worker(
                                &mut model,
                                receiver,
                                &worker_codec,
                                &worker_model,
                                &worker_shutdown,
                                scheduler,
                                &worker_activity,
                            );
                        }
                    }
                    Err(error) => {
                        worker_lifecycle
                            .fail(orinfer_engine::error::EngineError::take(error.clone(), true));
                        let _ = ready_sender.send(Err(error));
                    }
                }
            });
            if !worker_shutdown.load(Ordering::Relaxed) && worker_lifecycle.is_ready() {
                worker_lifecycle.fail(orinfer_engine::error::EngineError::take(
                    "GPU worker exited unexpectedly".into(),
                    true,
                ));
            }
        })
        .map_err(|e| e.to_string())?;
    let ready = ready_receiver
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r);
    let (context, vision, mtp_drafts) = match ready {
        Ok(ready) => ready,
        Err(error) => {
            shutdown.store(true, Ordering::Relaxed);
            drop(sender);
            if wait_worker(&worker, limits.drain_ms).await {
                let _ = worker.join();
            }
            return Err(error);
        }
    };
    let state = Service {
        jobs: sender,
        slots: Arc::new(tokio::sync::Semaphore::new(128 + scheduler.max_active)),
        codec,
        model: model_id,
        context,
        mtp_drafts,
        prompt_lookup,
        verify_weights,
        resident_experts,
        vision,
        api_key: std::env::var("ORINFER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(Arc::from),
        ids: Arc::new(AtomicU64::new(0)),
        scheduler,
        activity,
        preparation: Arc::new(preparation::Pool::new(limits)),
        limits,
        default_request_params,
    };
    Ok(Worker {
        state,
        shutdown,
        handle: worker,
    })
}

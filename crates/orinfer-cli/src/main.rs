use clap::{Args, Parser, Subcommand};
use orinfer_api::{BatchConfig, PreprocessingLimits, ServerConfig};
use orinfer_engine::execution::{
    CudaGraphMode, LoadOptions, ResidentExperts, parse_cache_mib, parse_load_workers,
    parse_mtp_drafts,
};
use std::{path::PathBuf, process::ExitCode};

#[derive(Clone, Debug)]
struct MtpDrafts(Option<usize>);
impl std::str::FromStr for MtpDrafts {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        parse_mtp_drafts(value).map(Self)
    }
}

#[derive(Parser)]
#[command(name = "orinfer", version, about = "LLM inference on Jetson AGX Orin")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Manage precompiled model execution packages without CUDA or Python.
    Package {
        #[command(subcommand)]
        command: PackageCommand,
    },
    /// Show target and implementation status (default command).
    Info,
    /// Print the benchmark specification as JSON.
    Plan,
    /// Check AOT fixture and file hashes without CUDA.
    ValidateArtifact { manifest: PathBuf },
    /// Execute an SM87 AOT fixture and validate graph replay.
    RunArtifact { manifest: PathBuf },
    /// Inspect a native execution library identity and capabilities without CUDA.
    InspectLibrary { library: PathBuf },
    /// Inspect the native model package plan without CUDA.
    PlanModel { model_dir: PathBuf },
    /// Verify model, execution package and tensor hashes without CUDA.
    ValidateModel { model_dir: PathBuf },
    /// Run requests through the native model package.
    RunModel(ModelArgs),
    /// Score frozen token histories through real prefill/decode kernels.
    ScoreModel(ModelArgs),
    /// Serve the OpenAI Chat Completions API.
    Serve(Box<ServeArgs>),
    /// Process Chat requests from JSONL without an HTTP server.
    Batch(Box<BatchArgs>),
}
#[derive(Subcommand)]
enum PackageCommand {
    /// Verify and atomically install a .tar.xz or .tar.gz execution package.
    Install {
        archive: PathBuf,
        /// Installation cache; defaults to ORINFER_EXECUTION_CACHE or the user cache.
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}
#[derive(Args)]
struct LoadArgs {
    /// Routed experts per layer equivalent, shared globally; full keeps all. MTP stays resident.
    #[arg(long, value_name = "full|N", default_value_t = ResidentExperts::default())]
    resident_experts: ResidentExperts,
    /// CUDA graph capture: decode_only, full or off.
    #[arg(long, value_name = "MODE", default_value_t = CudaGraphMode::default())]
    cuda_graph: CudaGraphMode,
    /// Hash loaded GPU weights and CPU row tables at startup (slower).
    #[arg(long)]
    verify_weights: bool,
    /// Parallel weight readers; default selects a bounded number of CPU cores.
    #[arg(long, value_name = "N", value_parser = parse_load_workers)]
    load_workers: Option<usize>,
    /// Opt in to copied greedy drafts from this request's token history.
    #[arg(long)]
    prompt_lookup: bool,
}
#[derive(Args)]
struct ModelArgs {
    model_dir: PathBuf,
    requests: PathBuf,
    #[command(flatten)]
    load: LoadArgs,
}
#[derive(Args)]
struct InferenceArgs {
    /// Prepared model directory including tokenizer and execution package.
    model_dir: PathBuf,
    #[arg(long, default_value_t = ServerConfig::default().model)]
    model: String,
    /// latency (serve default) or throughput (batch default).
    #[arg(long, value_name = "POLICY")]
    scheduler_policy: Option<orinfer_engine::scheduler::Policy>,
    /// JSON object of Chat request defaults; explicit request fields take priority.
    #[arg(long, value_name = "JSON", value_parser = parse_request_defaults)]
    default_request_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[arg(long, default_value_os_t = ServerConfig::default().gpu_lock)]
    gpu_lock: PathBuf,
    #[command(flatten)]
    load: LoadArgs,
    /// auto uses the package default; 0 disables MTP; 1..7 sets draft count.
    #[arg(long, value_name = "N", default_value = "auto")]
    mtp_drafts: MtpDrafts,
    /// Snapshot cache budget in MiB; 0 disables reuse.
    #[arg(long = "prefix-cache-mib", value_name = "MIB", default_value_t = ServerConfig::default().prefix_cache_bytes >> 20, value_parser = parse_cache_mib)]
    prefix_cache_bytes: usize,
    #[arg(long, default_value_t = ServerConfig::default().scheduler.max_active)]
    max_active_requests: usize,
    /// Decode/verification row budget; mixed prefill may use the package capacity.
    #[arg(long, default_value_t = ServerConfig::default().scheduler.max_batch_tokens)]
    max_batch_tokens: usize,
    /// Predicted mixed prefill budget in milliseconds (latency policy).
    #[arg(long, default_value_t = ServerConfig::default().scheduler.prefill_budget_ms)]
    prefill_budget_ms: f64,
    /// Soft decode token interval target in milliseconds (latency policy).
    #[arg(long, default_value_t = ServerConfig::default().scheduler.target_tpot_ms)]
    target_tpot_ms: f64,
    #[arg(long = "memory-reserve-mib", value_name = "MIB", default_value_t = ServerConfig::default().scheduler.memory_reserve_bytes >> 20, value_parser = parse_cache_mib)]
    memory_reserve_bytes: usize,
    #[arg(long, default_value_t = PreprocessingLimits::default().preprocess_workers)]
    preprocess_workers: usize,
    #[arg(long, default_value_t = PreprocessingLimits::default().memory_mib)]
    preprocess_memory_mib: usize,
}
#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long, default_value_t = ServerConfig::default().listen)]
    listen: String,
    /// Queue deadline in milliseconds; 0 means unlimited.
    #[arg(long, default_value_t = PreprocessingLimits::default().queue_ms)]
    queue_timeout_ms: u64,
    #[arg(long, default_value_t = PreprocessingLimits::default().output_ms)]
    output_timeout_ms: u64,
    #[arg(long, default_value_t = PreprocessingLimits::default().drain_ms)]
    drain_timeout_ms: u64,
}
#[derive(Args)]
struct BatchArgs {
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Skip committed responses and retry failed rows in the same job.
    #[arg(long)]
    resume: bool,
}
impl From<InferenceArgs> for ServerConfig {
    fn from(args: InferenceArgs) -> Self {
        Self {
            model_dir: args.model_dir,
            model: args.model,
            default_request_params: args.default_request_params.unwrap_or_default(),
            listen: ServerConfig::default().listen,
            gpu_lock: args.gpu_lock,
            cuda_graph: args.load.cuda_graph,
            verify_weights: args.load.verify_weights,
            load_workers: args.load.load_workers,
            resident_experts: args.load.resident_experts,
            mtp_drafts: args.mtp_drafts.0,
            prompt_lookup: args.load.prompt_lookup,
            prefix_cache_bytes: args.prefix_cache_bytes,
            scheduler: orinfer_engine::scheduler::Options {
                policy: args.scheduler_policy.unwrap_or_default(),
                max_active: args.max_active_requests,
                max_batch_tokens: args.max_batch_tokens,
                prefill_budget_ms: args.prefill_budget_ms,
                target_tpot_ms: args.target_tpot_ms,
                memory_reserve_bytes: args.memory_reserve_bytes,
            },
            limits: PreprocessingLimits {
                preprocess_workers: args.preprocess_workers,
                memory_mib: args.preprocess_memory_mib,
                ..Default::default()
            },
        }
    }
}
impl From<ServeArgs> for ServerConfig {
    fn from(args: ServeArgs) -> Self {
        let mut config: Self = args.inference.into();
        config.listen = args.listen;
        config.limits.queue_ms = args.queue_timeout_ms;
        config.limits.output_ms = args.output_timeout_ms;
        config.limits.drain_ms = args.drain_timeout_ms;
        config
    }
}
impl From<BatchArgs> for BatchConfig {
    fn from(args: BatchArgs) -> Self {
        let policy = args
            .inference
            .scheduler_policy
            .unwrap_or(orinfer_engine::scheduler::Policy::Throughput);
        let mut config: ServerConfig = args.inference.into();
        config.scheduler.policy = policy;
        Self {
            input: args.input,
            output: args.output,
            resume: args.resume,
            model_dir: config.model_dir,
            model: config.model,
            gpu_lock: config.gpu_lock,
            scheduler: config.scheduler,
            limits: config.limits,
            default_request_params: config.default_request_params,
            load: LoadOptions {
                cuda_graph: config.cuda_graph,
                verify_weights: config.verify_weights,
                load_workers: config.load_workers,
                resident_experts: config.resident_experts,
                mtp_drafts: config.mtp_drafts,
                prompt_lookup: config.prompt_lookup,
                prefix_cache_bytes: config.prefix_cache_bytes,
            },
        }
    }
}
fn parse_request_defaults(
    value: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    serde_json::from_str(value).map_err(|e| format!("Expected a JSON object: {e}"))
}
fn print_json(value: impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn execute(command: Command) -> Result<(), String> {
    match command {
        Command::Package {
            command: PackageCommand::Install { archive, cache },
        } => print_json(orinfer_engine::operators::install::install(
            &archive,
            &cache.unwrap_or_else(orinfer_engine::operators::default_cache),
        )?),
        Command::Info => {
            println!("Orinfer {}", env!("CARGO_PKG_VERSION"));
            println!(
                "Target: {} / {} / {}",
                orinfer_engine::TARGET_DEVICE,
                orinfer_engine::TARGET_ARCH,
                orinfer_engine::CUDA_ARCH
            );
            println!("Model: {}", orinfer_engine::FIRST_MODEL);
            println!("Status: {}", orinfer_engine::STATUS);
            Ok(())
        }
        Command::Plan => {
            println!("{}", orinfer_engine::BENCHMARK_PLAN);
            Ok(())
        }
        Command::Serve(args) => orinfer_api::run((*args).into()),
        Command::Batch(args) => {
            let report = orinfer_api::run_batch((*args).into())?;
            let failed = report.failed != 0 || report.truncated != 0 || report.interrupted;
            print_json(report)?;
            if failed {
                Err("Batch has errors, truncated responses or was interrupted; inspect the output JSONL".into())
            } else {
                Ok(())
            }
        }
        Command::ValidateArtifact { manifest } => {
            print_json(orinfer_engine::artifact::validate_artifact(&manifest)?)
        }
        Command::RunArtifact { manifest } => {
            print_json(orinfer_engine::artifact::run_artifact(&manifest)?)
        }
        Command::InspectLibrary { library } => {
            print_json(orinfer_engine::model::inspect_execution_library(&library)?)
        }
        Command::PlanModel { model_dir } => {
            print_json(orinfer_engine::model::inspect_plan(&model_dir)?)
        }
        Command::ValidateModel { model_dir } => {
            print_json(orinfer_engine::model::validate_model(&model_dir)?)
        }
        Command::RunModel(args) => print_json(orinfer_engine::model::run_with_options(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                prompt_lookup: args.load.prompt_lookup,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                resident_experts: args.load.resident_experts,
                ..Default::default()
            },
        )?),
        Command::ScoreModel(args) => print_json(orinfer_engine::model::score_with_options(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                prompt_lookup: args.load.prompt_lookup,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                resident_experts: args.load.resident_experts,
                ..Default::default()
            },
        )?),
    }
}
fn main() -> ExitCode {
    match execute(Cli::parse().command.unwrap_or(Command::Info)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn serve(options: &[&str]) -> Result<ServerConfig, clap::Error> {
        let mut args = vec!["orinfer", "serve", "/tmp"];
        args.extend_from_slice(options);
        let Some(Command::Serve(args)) = Cli::try_parse_from(args)?.command else {
            unreachable!()
        };
        Ok((*args).into())
    }
    #[test]
    fn defaults_and_server_options_are_preserved() {
        let config = serve(&[]).unwrap();
        config.validate().unwrap();
        assert_eq!(config.listen, "0.0.0.0:8088");
        assert_eq!(config.cuda_graph, CudaGraphMode::DecodeOnly);
        assert_eq!(config.prefix_cache_bytes, 12usize << 30);
        assert_eq!(config.scheduler.max_active, 32);
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Latency
        );
        assert_eq!(config.mtp_drafts, None);
        assert_eq!(config.resident_experts, ResidentExperts::Full);
        assert!(!config.prompt_lookup);
        assert!(!config.verify_weights);
        assert_eq!(config.load_workers, None);
        assert!(config.default_request_params.is_empty());
        let config = serve(&[
            "--verify-weights",
            "--prompt-lookup",
            "--mtp-drafts=7",
            "--resident-experts=256",
            "--cuda-graph",
            "off",
            "--max-active-requests",
            "8",
            "--max-batch-tokens",
            "64",
            "--prefill-budget-ms",
            "150",
            "--target-tpot-ms",
            "350",
            "--memory-reserve-mib",
            "2048",
            "--prefix-cache-mib",
            "0",
        ])
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.cuda_graph, CudaGraphMode::Off);
        assert_eq!(config.mtp_drafts, Some(7));
        assert_eq!(config.resident_experts, ResidentExperts::Count(256));
        assert!(config.prompt_lookup);
        assert!(config.verify_weights);
        assert_eq!(config.prefix_cache_bytes, 0);
        assert_eq!(config.scheduler.memory_reserve_bytes, 2usize << 30);
    }
    #[test]
    fn batch_reuses_inference_options_without_http_settings() {
        let Some(Command::Batch(args)) = Cli::try_parse_from([
            "orinfer",
            "batch",
            "/tmp",
            "--input",
            "requests.jsonl",
            "--output",
            "results.jsonl",
            "--resume",
            "--model",
            "translation",
            "--max-active-requests",
            "64",
            "--mtp-drafts",
            "0",
            "--default-request-params",
            "{\"seed\":20261002,\"temperature\":0}",
        ])
        .unwrap()
        .command
        else {
            panic!("Expected batch command")
        };
        let config: BatchConfig = (*args).into();
        assert_eq!(config.input, PathBuf::from("requests.jsonl"));
        assert_eq!(config.output, PathBuf::from("results.jsonl"));
        assert!(config.resume);
        assert_eq!(config.model, "translation");
        assert_eq!(config.scheduler.max_active, 64);
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Throughput
        );
        assert_eq!(config.load.mtp_drafts, Some(0));
        assert_eq!(config.default_request_params["seed"], 20261002);
        assert!(
            Cli::try_parse_from(["orinfer", "batch", "/tmp", "--input", "requests.jsonl"]).is_err()
        );
    }
    #[test]
    fn scheduler_policy_is_explicit_and_batch_defaults_do_not_change_serve() {
        let config = serve(&["--scheduler-policy", "throughput"]).unwrap();
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Throughput
        );
        assert!(serve(&["--scheduler-policy", "unknown"]).is_err());
        let Some(Command::Batch(args)) = Cli::try_parse_from([
            "orinfer",
            "batch",
            "/tmp",
            "--input",
            "input.jsonl",
            "--output",
            "output.jsonl",
            "--scheduler-policy",
            "latency",
        ])
        .unwrap()
        .command
        else {
            unreachable!()
        };
        let config: BatchConfig = (*args).into();
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Latency
        );
    }
    #[test]
    fn syntax_help_and_limits_are_checked_before_loading() {
        for args in [
            vec!["orinfer", "--help"],
            vec!["orinfer", "--version"],
            vec!["orinfer", "serve", "--help"],
            vec!["orinfer", "batch", "--help"],
            vec!["orinfer", "package", "--help"],
            vec!["orinfer", "package", "install", "--help"],
        ] {
            assert_eq!(Cli::try_parse_from(args).err().unwrap().exit_code(), 0);
        }
        for options in [
            &["--cuda-graph"][..],
            &["--cuda-graph", "on"],
            &["--verify-weights=false"],
            &["--mtp-drafts", "8"],
            &["--unknown", "off"],
            &["--prefix-cache-mib", "1.5"],
            &["--listen", "a", "--listen", "b"],
            &["--default-request-params", "[]"],
            &["--default-request-params", "{broken"],
        ] {
            assert!(serve(options).is_err());
        }
        for (name, value) in [
            ("--max-active-requests", "129"),
            ("--max-batch-tokens", "0"),
            ("--target-tpot-ms", "NaN"),
            ("--output-timeout-ms", "0"),
        ] {
            assert!(serve(&[name, value]).unwrap().validate().is_err());
        }
    }
    #[test]
    fn package_install_accepts_an_archive_and_optional_cache() {
        for cache in [None, Some("/tmp/packages")] {
            let mut args = vec!["orinfer", "package", "install", "model.tar.xz"];
            if let Some(path) = cache {
                args.extend(["--cache", path]);
            }
            let Some(Command::Package {
                command:
                    PackageCommand::Install {
                        archive,
                        cache: parsed,
                    },
            }) = Cli::try_parse_from(args).unwrap().command
            else {
                unreachable!()
            };
            assert_eq!(archive, PathBuf::from("model.tar.xz"));
            assert_eq!(parsed, cache.map(PathBuf::from));
        }
        for args in [
            vec!["orinfer", "package"],
            vec!["orinfer", "package", "install"],
            vec!["orinfer", "package", "install", "model.tar.xz", "extra"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
    }
    #[test]
    fn deployment_defaults_are_typed_and_preserved() {
        let config = serve(&[
            "--default-request-params",
            r#"{"enable_thinking":true,"temperature":0.7}"#,
        ])
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.default_request_params["enable_thinking"], true);
        assert_eq!(config.default_request_params["temperature"], 0.7);
        for defaults in [
            r#"{"enable_thinking":"true"}"#,
            r#"{"unknown":1}"#,
            r#"{"model":"x"}"#,
            r#"{"messages":[]}"#,
        ] {
            assert!(
                serve(&["--default-request-params", defaults])
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
    }
    #[test]
    fn model_subcommands_support_equals_and_reject_extra_positionals() {
        let cli = Cli::try_parse_from([
            "orinfer",
            "run-model",
            "model",
            "requests.json",
            "--cuda-graph=full",
            "--verify-weights",
        ])
        .unwrap();
        let Some(Command::RunModel(args)) = cli.command else {
            unreachable!()
        };
        assert_eq!(args.load.cuda_graph, CudaGraphMode::Full);
        assert!(args.load.verify_weights);
        assert_eq!(
            serve(&["--load-workers", "12"]).unwrap().load_workers,
            Some(12)
        );
        assert!(serve(&["--load-workers", "0"]).is_err());
        assert!(serve(&["--load-workers", "33"]).is_err());
        for verify in [false, true] {
            let mut argv = vec!["orinfer", "score-model", "model", "requests.json"];
            if verify {
                argv.push("--verify-weights");
            }
            let Some(Command::ScoreModel(args)) = Cli::try_parse_from(argv).unwrap().command else {
                unreachable!()
            };
            assert_eq!(args.load.verify_weights, verify);
        }
        assert!(Cli::try_parse_from(["orinfer", "run-model", "model"]).is_err());
        assert!(Cli::try_parse_from(["orinfer", "plan-model", "model", "extra"]).is_err());
    }
}

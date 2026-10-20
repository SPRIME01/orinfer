# Build, checks, and release

[简体中文](../../../../tools/build/README.md) | **English**

The online service needs only a Rust binary, CUDA driver, prepared model directory, and execution package. Python, PyTorch, and TileLang belong to the **offline** build environment.

## Compiler environment

```bash
make compiler-image
make check-offline
```

`compiler.Dockerfile` pins a public NVIDIA Jetson CUDA 12.6 / PyTorch 2.9.1 base-image digest. Pinned `uv` manages `/opt/venv` in the image and installs TileLang 0.1.15, Transformers 5.20.1, Tokenizers 0.23.3, and Hub 2.2.0. Models and compile caches are not copied into the image. `compiler-requirements.txt` pins all Python and build dependencies; `uv` synchronization removes unrelated inherited service dependencies. The original Jetson PyTorch 2.9.1 and torchvision 0.24.1 remain unchanged, and the build verifies that Torch still links to CUDA 12.6. System packages are pinned by the base-image digest. The GPU compiler keeps Python 3.10 and NumPy 2.2.6; TVM FFI and Z3 follow TileLang's constraints.

The default image is `orinfer-compiler:0.1.2`, overridable with `ORINFER_OPERATOR_IMAGE`. NVRTC uses CUDA Python 12.6.2.post1; some offline native-operator references use Triton 3.5.1. Both are pinned alongside TileLang in the compiler image. Deployment loads only precompiled native execution packages.

GPU compilation example:

```bash
ORINFER_CHECKPOINT_DIR=/path/to/converted \
  bash tools/operators/run.sh tools/model/qwen35/build.py artifacts/build/qwen35 \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
```

GPU entry points run under the invoking UID/GID and supplementary device groups to avoid root-owned outputs. The repository and output directories are mounted at their current absolute paths; `ORINFER_CHECKPOINT_DIR` mounts external checkpoints read-only. See [Model builds](../model/README.md) for input requirements. The 27B builder reads fully quantized custom-codebook weights; the Flash Next builder reads a current native model with its execution contract. Neither imports arbitrary bare model weights.

A new compiler environment can be smoke-tested without a checkpoint:

```bash
bash tools/operators/run.sh tools/build/smoke.py artifacts/build/compiler-smoke
```

This compiles actual production norm and dynamic-row greedy reduction from an empty cache, checking unaligned tails, cubin sharing across 3/7 rows, scalar ABI for affine shapes, output guards, CUDA Graph replay after input changes, and AOT export.

## Automated checks

`make check` runs Rust/Ruff formatting, Python static checks, compilation, Clippy, Rust/CPU Python tests, and acceptance-config validation. GitHub Actions runs the same entry point on x86_64 and aarch64; native C ABI loading is tested on aarch64, using `rust-toolchain.toml`. `make fmt` formats; `make lint` runs formatting and static analysis. Rust builds reject dead code, and Clippy warnings are errors.

`make python-env` uses `uv` to create a Python 3.12 `.venv` with pinned dependencies. Checks default to `.venv/bin/python` and do not modify system Python; override with `PYTHON`. CI and the compiler image pin uv 0.13.1. Rust is pinned by `rust-toolchain.toml`. Python sources remain syntax-compatible with Python 3.10 for GPU builds.

`make check-offline` checks imports across offline modules in the compiler image and runs Torch-dependent CPU reference math and dynamic-row ABI tests, without a GPU.

```bash
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

This builds native Chat-template fixtures for text, image, and multi-image cases using seed 20261002. It checks dynamic batching in `off`/`decode_only`/`full` modes, cancellation/reordering/reuse isolation, request cleanup, and matched-history probabilities. The old serial plan exists only as a testing reference; production full-request generation and serving share the same state machine. This does **not** replace independent BF16 quality acceptance.

GPU jobs acquire `artifacts/gpu-experiment.lock` and exit if it is held. They do not stop unrelated services, change power/frequency settings, or clear global caches. Release only idle services started by your own test.

## Releases

Commit sources, update the workspace version, and create an annotated tag pointing to that commit. Then run:

```bash
python3 tools/release/package.py --version v0.1.4 \
  --operators /path/to/qwen3_5-execution.tar.xz \
  --operators /path/to/flash-next-execution.tar.xz \
  --output artifacts/releases/v0.1.4
```

By default this builds and verifies artifacts without publishing. The tool requires a clean working tree, builds from `Cargo.lock`, verifies aarch64 and version identity, and packages the complete sources, license, dependency notices, model packaging tools, and architecture contract. Each native execution package and library identity is verified. Engine, execution, and bundled source archives use XZ compression (`.tar.xz`, preset 6). Legacy `.tar.gz` execution-package inputs are verified and *repacked* to XZ, not merely renamed. Asset names derive from architecture, compute policy, and GPU target; `RELEASE.json` lists each package digest. Full offline tools and Flash Next documentation accompany the release. Weights are excluded.

Install an execution package with `orinfer package install`; Python is not required. See [Installing execution packages](../../model-packages.md#installing-execution-packages).

Release 0.1.2 accepts only current schema-1 native execution packages containing `.so` and cubins; old operator assets from v0.1.1 are incompatible. Models in the old format must be rebuilt or reassembled with current tools. Current native models can use the [execution-library update workflow](../../model-packages.md), preserving compatible weights and kernel assets.

To publish, add `--publish --notes /path/to/release-notes.md`. The tool checks that the tag points to the source revision, pushes it, creates a GitHub Release, then downloads and SHA256-verifies all uploaded assets. Existing releases are never overwritten.

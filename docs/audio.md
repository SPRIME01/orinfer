# Speech recognition

Orinfer runs Qwen3-ASR-1.7B on Jetson AGX Orin 64GB with the native Rust runtime and TileLang execution package. The model accepts one recording per request and provides transcription with automatic language identification or an explicit language hint.

## Deploy and transcribe

Register the prepared native model with the running model manager:

```bash
orinfer import qwen3-asr-1.7b ./model
orinfer transcribe qwen3-asr-1.7b ./sample.wav --language zh
```

The manager loads the model on demand and applies its configured idle-unload policy. `--prompt` supplies context and names to help recognition.

The OpenAI-compatible file endpoint accepts multipart form data:

```bash
curl http://localhost:8088/v1/audio/transcriptions \
  -F model=qwen3-asr-1.7b \
  -F file=@sample.wav \
  -F language=zh \
  -F response_format=json
```

`response_format=json` returns `{"text":"…"}`; `text` returns UTF-8 text. Optional fields are `language`, `prompt`, and `temperature`. Decoding supports WAV, FLAC, MP3, AAC/MP4, Ogg/Vorbis and the other enabled Symphonia formats. Input preprocessing downmixes channels, resamples to 16 kHz and computes the checkpoint's 128-bin log-mel features. The file limit is 24 MiB and the recording limit is 1,200 seconds.

The CPU frontend uses Symphonia for decoding, Rubato's fixed-ratio FFT resampler for files, and NEON-enabled RealFFT for the 400-point real-input transform. Rayon processes independent frame groups through its shared worker pool. The checkpoint-specific window, Slaney filters, normalization and padding retain their reference semantics.

Chat Completions accepts `input_audio` content parts with base64 file data and a format hint. Its `stream` option streams generated output tokens. Chat responses preserve the checkpoint’s `language …<asr_text>…` output protocol; the file endpoint returns transcription text. JSONL inference uses the same request body and continuous scheduler. Each recording retains its own KV, feature buffers and positions; prefix matching includes the audio feature identity.

Word timestamps use the separately released Qwen3-ForcedAligner model. This ASR package provides transcription text. Live microphone sessions involve audio-input buffering and transcript revision in addition to generated-token streaming.

## Model and precision

The source is the pinned official `Qwen/Qwen3-ASR-1.7B-hf` checkpoint. Its 1.72B-parameter Qwen3 decoder uses 28 full-attention layers with 16 query heads, 8 KV heads and head dimension 128. The audio encoder and projector contribute approximately 0.32B parameters.

Text matrices and the shared embedding/output-head bank use two-stage E8P Q4A8, coefficients `[4,1]`, signed-block128 rotation and beam-8 encoding. Decoder layers 0–3 retain W8 for Q/K/V and FFN matrices in the same rotated input basis. Attention output projections retain W8. The audio encoder and projector use FP16; normalization reductions use FP32 and KV storage uses group-64 INT8 with FP16 scales. The checkpoint context limit is 65,536 tokens.

The native weight payload is **1.665 GB** (decimal), including alignment. The text portion averages about **4.71 bits per parameter**; the FP16 audio encoder brings the complete model to about **6.53 bits per parameter**. Workspace, request state and KV occupy additional memory.

Audio convolution uses NHWC tensors in bounded groups of 32 one-second mel chunks. Recordings selected for the current text packet share encoder work with 104-token-aligned, request-isolated windows. Features are copied directly to each owner's storage on the execution stream. The throughput policy bounds pending cold admission by the model's prefill packet capacity; ready decoders and cached prompts share the active cohort. Arena selection is reused across admission estimates and leasing, with invalidation on state changes. The text decoder losslessly joins Q/K/V and gate/up weight rows and runs the existing integer projection templates. The encoder uses 104-token bidirectional windows; the text decoder reuses the shared dynamic-row Qwen executor and GQA kernels with ordinary RMSNorm, full-dimensional RoPE and ungated attention output.

Ordinary decode uses address-table packet boundaries for token input, embedding, media overlays, position updates, final normalization and output distribution. The shared Qwen implementation also handles the decode prefix of mixed prefill/decode packets. Offline compilation wraps the original one-row arithmetic in a dynamic owner dimension; ASR decode uses 347 kernel nodes across 1–128 active requests. Each owner retains its complete logits and selected token. Validation covers fixed-history distributions, owner reordering, cancellation, prefix restoration and mixed prompt chunks.

## Validation

Quality comparisons use original BF16 weights, fixed seed `20261002`, frozen token histories and real recordings. CER/WER assess transcription quality; target-token NLL and top-three probabilities diagnose quantization changes. Performance measurement includes audio decoding, resampling, feature extraction, encoding, text prefill and generation. Report audio seconds processed per wall-clock second and real-time factor alongside token throughput.

The 2026-10-21 quick evaluation covers three original Chinese/English recordings and derived noisy, short, silent, concatenated and repeated inputs. Nine frozen-history cases (312 target tokens, including a 96-token prefix for the repeated recording) have weighted ΔNLL **0.0396 nats/token** against BF16; each native top-one choice falls within the BF16 top three. Chinese clean/noisy transcripts, short English and the independent LibriSpeech recording match the reference after punctuation/case normalization. The ordinary English recording omits an initial filler. Silence returns empty text, and concatenated-audio output follows the BF16 transcript apart from punctuation.

In an eight-repeat stress recording, BF16 returns seven repetitions and the quantized model six. This test exposes long-recording repetition merging. A full multilingual ASR benchmark remains a separate quality milestone.

### Rust CLI/API performance

The 2026-10-21 optimized package uses the precision policy above and keeps the original quantized weight values. The package stores 648 weight tensors in two safetensors shards, including the joined projection banks. Measurements use `decode_only`, greedy sampling, seed `20261002`, MTP off and prefix caching off. Each scenario has one warmup followed by three timed requests or cohorts; tables report medians.

The vLLM baseline is [vrfai/Qwen3-ASR-1.7B-fp8](https://huggingface.co/vrfai/Qwen3-ASR-1.7B-fp8), revision `201f1359c25a2d2a759c6f5742a747c789fc776f`, with vLLM source `18f658bb3185779ee58999a328246d09886d568b`. On SM87 it uses Marlin FP8 weights with FP16 activations, FP16 audio and FP16 KV. The research loader lowers the ModelOpt capability check from SM89 to SM80 to enable that fallback; its checkpoint payload is unchanged. Audio caches are reset before each vLLM trial, and concurrent recordings have distinct PCM identities.

Time to first content (TTFT) includes file decoding, feature extraction, audio encoding, text prefill and the checkpoint's language prefix. Decode throughput is `(completion_tokens - 1) / (response_end - first_content)`, including streaming overhead. These timings describe the complete speech pipeline.

| Recording | Prompt tokens | vLLM TTFT (ms) | Orinfer TTFT (ms) | vLLM decode (tok/s) | Orinfer decode (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chinese, 4.20 s | 70 | 96.63 | 34.05 | 74.71 | 115.21 |
| English, 15.05 s | 211 | 113.38 | 59.64 | 73.08 | 116.92 |
| Independent English, 7.06 s | 107 | 99.43 | 38.92 | 73.86 | 117.01 |
| Silence, 2.00 s | 41 | 92.33 | 29.12 | 76.22 | 108.91 |
| Concatenated speech, 19.46 s | 268 | 125.70 | 75.82 | 73.00 | 116.80 |
| Repeated speech, 122.01 s | 1602 | 479.72 | 377.14 | 67.99 | 105.18 |

The single-request reference table uses the `latency` policy. The following concurrent measurements use `--scheduler-policy throughput`, 128 active-request slots, a 128-row decode budget, two preprocessing workers and an 8 GiB preprocessing budget. The native before/after packet comparison uses the same Rust CLI and frozen weights, with one warmup and three timed cohorts in the same scenario order; the vLLM FP8 reference uses one warmup and two timed cohorts, with its active-request limit also set to 128.

| Recording mix | Concurrent requests | vLLM output (tok/s) | Orinfer before packet I/O (tok/s) | Orinfer packet I/O (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| Mixed short recordings | 8 | 192.44 | 277.45 | 279.42 |
| Mixed short recordings | 32 | 403.94 | 449.43 | 455.73 |
| Mixed short recordings | 64 | 512.64 | 504.27 | 513.77 |
| Mixed short recordings | 128 | 577.82 | 522.07 | 521.97 |
| English, 15.05 s each | 8 | 303.74 | 390.53 | 395.03 |
| English, 15.05 s each | 32 | 544.03 | 565.80 | 574.16 |
| English, 15.05 s each | 64 | 511.91 | 614.77 | 623.90 |
| English, 15.05 s each | 128 | 527.38 | 646.03 | 623.95 |

Cohort throughput divides all output tokens by the complete cohort duration, including frontend preparation, admission, audio encoding, text prefill and natural-EOS generation. Mixed cohorts contain distinct Chinese, English, short, silent, noisy, concatenated and independent recordings. Ordinary English produces 49 tokens with vLLM and 46 with Orinfer because of an initial filler.

Address-table packet boundaries keep ASR decode at **347 kernel nodes** for 1–128 requests. Fixed-batch GPU replay measures **8.314 → 8.322 ms** at one request and **10.262 → 10.176 ms** at eight requests. An eight-owner, fixed-history comparison against the original quantized package gives maximum KL **6.1e-13** and mean target-token NLL change **−1.1e-9**.

Complete-cohort throughput depends on admission and adaptive scheduling as well as GPU kernels. In the ordered workload above, mixed recordings are approximately unchanged to 2% faster; 128 equal-length recordings change from **646.03 to 623.95 tok/s**. A separate fresh 128-request cohort, with one warmup and five timed repetitions, measures **651.83 → 661.31 tok/s**. The ordered run uses more small decode batches after mixed-length traffic; the fresh run retains the full decode batch. Further optimization targets encoder CNNs, joint-prefill FFNs and scheduling feedback to reach the 2× speech-pipeline target.

Three fresh Rust `serve` processes reach `/health` in median **0.95 s**, emit first content **1.21 s** after launch and complete the first Chinese request in **1.30 s**. Filesystem caches are retained; GPU initialization, model loading and first-request work are included.

### CPU frontend performance

Release-mode measurements on the same Orin use one warmup and seven timed iterations. For a 15.05-second English recording, the fixed-ratio FFT resampler reduces file decoding plus resampling time as follows:

| Input WAV | Previous sinc resampler (ms) | FFT resampler (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 48 kHz, stereo | 48.50 | 18.85 | 2.57× |
| 44.1 kHz, mono | 54.56 | 17.06 | 3.20× |

An alternating in-process comparison of the spectral transforms reduces log-mel time from **2.01 to 1.54 ms** for 15.05 seconds of speech and from **12.49 to 9.93 ms** for 122.01 seconds. These are CPU-stage measurements; complete-request timings also include GPU encoding and generation. Resampler tests cover duration, impulse alignment, passband gain and alias rejection across 8–192 kHz. Real-recording checks include both resampled WAV inputs and fixed-history token probabilities.

### State and capacity checks

Validation covers TensorFlow mel references, FP16 audio-encoder comparisons, changed-input Graph replay, packed versus individual recordings, reversed admission and arena reuse, prefix restoration, cancellation and owner-order state isolation. The joined-weight package preserves the optimized encoder's frozen-history scores. The shared runtime also passes 2B prefix and owner-order GPU regressions and the complete CPU checks.

The optimized package transcribes 1,200 seconds of silent FLAC in **5.44 s** through the file API: 15,615 prompt tokens and an empty transcription. A separate initial capacity check uses 488 seconds of repeated speech to exercise prefill beyond 4,096 tokens with a bounded 128-token output. WAV (including 48 kHz PCM24 and float PCM), FLAC, MP3, AAC/MP4 and Ogg/Vorbis uploads pass end-to-end checks.

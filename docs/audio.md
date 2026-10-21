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

Chat Completions accepts `input_audio` content parts with base64 file data and a format hint. Its `stream` option streams generated output tokens. Chat responses preserve the checkpoint’s `language …<asr_text>…` output protocol; the file endpoint returns transcription text. JSONL inference uses the same request body and continuous scheduler. Each recording retains its own KV, feature buffers and positions; prefix matching includes the audio feature identity.

Word timestamps use the separately released Qwen3-ForcedAligner model. This ASR package provides transcription text. Live microphone sessions involve audio-input buffering and transcript revision in addition to generated-token streaming.

## Model and precision

The source is the pinned official `Qwen/Qwen3-ASR-1.7B-hf` checkpoint. Its 1.72B-parameter Qwen3 decoder uses 28 full-attention layers with 16 query heads, 8 KV heads and head dimension 128. The audio encoder and projector contribute approximately 0.32B parameters.

Text matrices and the shared embedding/output-head bank use two-stage E8P Q4A8, coefficients `[4,1]`, signed-block128 rotation and beam-8 encoding. Decoder layers 0–3 retain W8 for Q/K/V and FFN matrices in the same rotated input basis. Attention output projections retain W8. The audio encoder and projector use FP16; normalization reductions use FP32 and KV storage uses group-64 INT8 with FP16 scales. The checkpoint context limit is 65,536 tokens.

The native weight payload is **1.665 GB** (decimal), including alignment. The text portion averages about **4.71 bits per parameter**; the FP16 audio encoder brings the complete model to about **6.53 bits per parameter**. Workspace, request state and KV occupy additional memory.

Audio convolution runs in bounded groups of 32 one-second mel chunks. The encoder uses 104-token bidirectional windows; the text decoder reuses the shared dynamic-row Qwen executor and GQA kernels with ordinary RMSNorm, full-dimensional RoPE and ungated attention output.

## Validation

Quality comparisons use original BF16 weights, fixed seed `20261002`, frozen token histories and real recordings. CER/WER assess transcription quality; target-token NLL and top-three probabilities diagnose quantization changes. Performance measurement includes audio decoding, resampling, feature extraction, encoding, text prefill and generation. Report audio seconds processed per wall-clock second and real-time factor alongside token throughput.

The 2026-10-21 quick evaluation covers three original Chinese/English recordings and derived noisy, short, silent, concatenated and repeated inputs. Nine frozen-history cases (312 target tokens, including a 96-token prefix for the repeated recording) have weighted ΔNLL **0.0435 nats/token** against BF16; each native top-one choice falls within the BF16 top three. Chinese clean/noisy transcripts, short English and the independent LibriSpeech recording match the reference after punctuation/case normalization. The ordinary English recording omits an initial filler. Silence returns empty text, and concatenated-audio output follows the BF16 transcript apart from punctuation.

In an eight-repeat stress recording, BF16 returns seven repetitions and the quantized model six. This test exposes long-recording repetition merging. A full multilingual ASR benchmark remains a separate quality milestone.

### End-to-end measurements

Measured on Jetson AGX Orin 64GB through the Rust CLI/API with `decode_only`, greedy sampling, MTP disabled and a 1 GiB prefix cache. These are individual completed requests and cohorts, with file caches preserved. Chat time to first content includes the checkpoint's language prefix.

| Recording / request | Audio duration | Complete response | First content | Real-time factor |
| --- | ---: | ---: | ---: | ---: |
| Chinese | 4.20 s | 0.125 s | 0.027 s | 0.030 |
| English | 15.05 s | 0.591 s | 0.170 s | 0.039 |
| Independent English | 7.06 s | 0.313 s | 0.077 s | 0.044 |
| 8 concurrent mixed recordings | 68.17 s total | 1.520 s cohort | — | 0.022 aggregate |

For the Chinese cold request, a fresh lazy manager reaches `/health` in **0.008 s**, emits its first content **1.21 s** after process launch and completes in **1.31 s**. Model loading and first-request work are included; filesystem caches are retained.

Concurrency checks cover 2/4/8 requests with independent recordings and silence. Four JSONL requests complete with four active requests and zero errors or truncations. WAV (including 48 kHz PCM24 and float PCM), FLAC, MP3, AAC/MP4 and Ogg/Vorbis uploads pass end-to-end checks. A 1,200-second silent FLAC produces 15,615 prompt tokens and empty transcription in **8.50 s**. A 488-second repeated-speech test exercises prefill beyond 4,096 tokens with a deliberately bounded 128-token output; its timing measures capacity and prefix generation.

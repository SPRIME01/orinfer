//! Bounded audio decoding and input preparation for native ASR models.
use base64::{Engine, engine::general_purpose::STANDARD};
use orinfer_engine::audio::{AudioInput, AudioSpec};
use rubato::{
    Fft, FixedSync, Resampler, WindowFunction, audioadapter_buffers::direct::InterleavedSlice,
};
use serde_json::{Value, json};
use std::io::Cursor;
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};
type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: usize = 24 * 1024 * 1024;

pub(super) fn decode(
    bytes: Vec<u8>,
    format_hint: &str,
    context: &super::preparation::Context,
) -> Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len() > MAX_BYTES {
        return Err("Audio file must contain 1..24 MiB".into());
    }
    context.checkpoint(bytes.len())?;
    let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(format_hint);
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("Audio container: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("No audio track")?;
    let track_id = track.id;
    let parameters = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("Missing audio codec parameters")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(parameters, &AudioDecoderOptions::default())
        .map_err(|e| format!("Audio decoder: {e}"))?;
    let mut samples = Vec::new();
    let mut rate = None;
    let mut temporary = Vec::<f32>::new();
    while let Some(packet) = format
        .next_packet()
        .map_err(|e| format!("Audio packet: {e}"))?
    {
        context.checkpoint(samples.len() * 4)?;
        if packet.track_id != track_id {
            continue;
        }
        let audio = decoder
            .decode(&packet)
            .map_err(|e| format!("Audio decode: {e}"))?;
        let spec = audio.spec();
        let current = spec.rate();
        let channels = spec.channels().count();
        if !(8000..=192000).contains(&current)
            || channels == 0
            || channels > 8
            || rate.is_some_and(|r| r != current)
        {
            return Err("Unsupported or changing audio sample rate/channel layout".into());
        }
        rate = Some(current);
        temporary.resize(audio.samples_interleaved(), 0.0);
        audio.copy_to_slice_interleaved(&mut temporary);
        if samples.len() + temporary.len() / channels > current as usize * 1200 {
            return Err("ASR supports recordings up to 1200 seconds".into());
        }
        samples.extend(
            temporary
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
    }
    let rate = rate.ok_or("Audio has no decoded samples")?;
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err("Audio contains invalid PCM samples".into());
    }
    if rate != 16000 {
        // Files have one fixed sample-rate ratio. Rubato's FFT resampler
        // preserves anti-aliasing and trims its delay in process_all.
        let mut resampler = Fft::<f32>::new_custom(
            rate as usize,
            16000,
            1024,
            1,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Both,
        )
        .map_err(|e| format!("Audio resampler: {e}"))?;
        let buffer =
            InterleavedSlice::new(&samples, 1, samples.len()).map_err(|e| e.to_string())?;
        samples = resampler
            .process_all(&buffer, samples.len(), None)
            .map_err(|e| format!("Audio resampling: {e}"))?
            .take_data();
    }
    Ok(samples)
}

pub(super) fn messages(
    mut messages: Vec<Value>,
    spec: Option<&AudioSpec>,
    preparation: &mut super::preparation::Context,
) -> Result<(Vec<Value>, Vec<AudioInput>)> {
    let mut inputs = vec![];
    for message in &mut messages {
        let user = message["role"] == "user";
        if let Some(parts) = message["content"].as_array_mut() {
            for part in parts {
                if matches!(part["type"].as_str(), Some("input_audio" | "audio")) {
                    if !user {
                        return Err("Audio parts require user messages".into());
                    }
                    spec.ok_or(
                        "This model accepts text and images; select an ASR model for audio",
                    )?;
                    if !inputs.is_empty() {
                        return Err("ASR accepts one recording per request".into());
                    }
                    let audio = &part["input_audio"];
                    let data = audio["data"]
                        .as_str()
                        .ok_or("input_audio.data requires base64 file bytes")?;
                    if data.len() > MAX_BYTES.div_ceil(3) * 4 {
                        return Err("Audio exceeds the file size limit".into());
                    }
                    let format = audio["format"]
                        .as_str()
                        .ok_or("input_audio.format is required")?;
                    let raw = STANDARD
                        .decode(data)
                        .map_err(|e| format!("Audio base64: {e}"))?;
                    let pcm = decode(raw, format, preparation)?;
                    // Frame groups and final mel coexist during bounded reassembly.
                    preparation.checkpoint(pcm.len() * 4 + 128 * (pcm.len() / 160 + 100) * 8)?;
                    let input =
                        orinfer_engine::audio::log_mel(&pcm, &|| preparation.checkpoint(0))?;
                    input.features()?;
                    preparation.retain(input.mel.len() * 4)?;
                    inputs.push(input);
                    *part = json!({"type":"audio"});
                }
            }
        }
    }
    Ok((messages, inputs))
}

pub(crate) fn language(value: &str) -> Result<&'static str> {
    const LANGUAGES: &[(&str, &str)] = &[
        ("zh", "Chinese"),
        ("en", "English"),
        ("yue", "Cantonese"),
        ("ar", "Arabic"),
        ("de", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("pt", "Portuguese"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ko", "Korean"),
        ("ru", "Russian"),
        ("th", "Thai"),
        ("vi", "Vietnamese"),
        ("ja", "Japanese"),
        ("tr", "Turkish"),
        ("hi", "Hindi"),
        ("ms", "Malay"),
        ("nl", "Dutch"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("pl", "Polish"),
        ("cs", "Czech"),
        ("fil", "Filipino"),
        ("fa", "Persian"),
        ("el", "Greek"),
        ("hu", "Hungarian"),
        ("mk", "Macedonian"),
        ("ro", "Romanian"),
    ];
    LANGUAGES
        .iter()
        .find(|(code, name)| value.eq_ignore_ascii_case(code) || value.eq_ignore_ascii_case(name))
        .map(|(_, name)| *name)
        .ok_or_else(|| "Unsupported ASR language".into())
}

pub(crate) async fn transcription_request(
    mut form: multer::Multipart<'_>,
) -> Result<(Value, bool)> {
    let mut fields = std::collections::BTreeMap::new();
    let mut file = None;
    while let Some(field) = form.next_field().await.map_err(|e| e.to_string())? {
        let name = field
            .name()
            .ok_or("Unnamed transcription field")?
            .to_owned();
        if name == "file" {
            if file.is_some() {
                return Err("Provide one audio file".into());
            }
            let format = field
                .file_name()
                .and_then(|s| s.rsplit('.').next())
                .unwrap_or("wav")
                .to_owned();
            let bytes = field.bytes().await.map_err(|e| e.to_string())?;
            if bytes.is_empty() || bytes.len() > MAX_BYTES {
                return Err("Audio file exceeds the 24 MiB limit".into());
            }
            file = Some(json!({"data":STANDARD.encode(bytes),"format":format}));
        } else {
            if ![
                "model",
                "language",
                "prompt",
                "response_format",
                "temperature",
                "stream",
            ]
            .contains(&name.as_str())
                || fields.contains_key(&name)
            {
                return Err(format!(
                    "Unsupported or repeated transcription field {name}"
                ));
            }
            let value = field.text().await.map_err(|e| e.to_string())?;
            if value.len() > 65536 {
                return Err("Transcription field too large".into());
            }
            fields.insert(name, value);
        }
    }
    let model = fields
        .remove("model")
        .filter(|s| !s.is_empty())
        .ok_or("model is required")?;
    let file = file.ok_or("file is required")?;
    let text = match fields
        .remove("response_format")
        .as_deref()
        .unwrap_or("json")
    {
        "json" => false,
        "text" => true,
        _ => return Err("Supported transcription response formats: json, text".into()),
    };
    if fields.remove("stream").is_some_and(|s| s != "false") {
        return Err(
            "Use streaming Chat Completions with input_audio for incremental transcription output"
                .into(),
        );
    }
    let mut request = json!({"model":model,"messages":[{"role":"system","content":fields.remove("prompt").unwrap_or_default()},{"role":"user","content":[{"type":"input_audio","input_audio":file}]}],"enable_thinking":false,"stream":false,"max_completion_tokens":8192});
    if let Some(value) = fields.remove("language") {
        request["asr_language"] = json!(language(&value)?);
    }
    if let Some(value) = fields.remove("temperature") {
        let temperature = value.parse::<f64>().map_err(|e| e.to_string())?;
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err("temperature must be between 0 and 2".into());
        }
        request["temperature"] = json!(temperature);
    }
    Ok((request, text))
}

pub(crate) async fn transcription_response(
    response: axum::response::Response,
    plain: bool,
) -> axum::response::Response {
    use axum::{Json, http::StatusCode, response::IntoResponse};
    if !response.status().is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let result = async {
        let bytes = axum::body::to_bytes(body, 16 << 20)
            .await
            .map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if value["choices"][0]["finish_reason"] != "stop" {
            return Err("Transcription exhausted its output budget".to_owned());
        }
        let raw = value["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("Missing transcription text")?;
        let text = raw
            .split_once("<asr_text>")
            .map_or(raw, |(_, text)| text)
            .trim()
            .to_owned();
        Ok(text)
    }
    .await;
    match result {
        Ok(text) => {
            let mut output = if plain {
                ([("content-type", "text/plain; charset=utf-8")], text).into_response()
            } else {
                Json(json!({"text":text})).into_response()
            };
            if let Some(value) = parts.headers.get("x-request-id") {
                output.headers_mut().insert("x-request-id", value.clone());
            }
            output
        }
        Err(e) => super::error(StatusCode::UNPROCESSABLE_ENTITY, e),
    }
}

pub(crate) fn multipart(
    headers: &axum::http::HeaderMap,
    bytes: axum::body::Bytes,
) -> Result<multer::Multipart<'static>> {
    let boundary = multer::parse_boundary(
        headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .ok_or("Multipart content-type is required")?,
    )
    .map_err(|e| e.to_string())?;
    let stream = futures_util::stream::once(std::future::ready(Ok::<_, std::io::Error>(bytes)));
    Ok(multer::Multipart::new(stream, boundary))
}
pub(super) async fn transcriptions(
    axum::extract::State(state): axum::extract::State<super::Service>,
    axum::Extension(admission): axum::Extension<super::preparation::Ingress>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> axum::response::Response {
    let bytes = match axum::body::to_bytes(body, 32 << 20).await {
        Ok(v) => v,
        Err(e) => return super::error(axum::http::StatusCode::PAYLOAD_TOO_LARGE, e.to_string()),
    };
    let form = match multipart(&headers, bytes) {
        Ok(v) => v,
        Err(e) => return super::error(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let (request, plain) = match transcription_request(form).await {
        Ok(v) => v,
        Err(e) => return super::error(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let response = super::completions(
        axum::extract::State(state),
        axum::Extension(admission),
        Ok(axum::Json(request)),
    )
    .await;
    transcription_response(response, plain).await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let size = (samples.len() * 2) as u32;
        let mut data = b"RIFF".to_vec();
        data.extend((36 + size).to_le_bytes());
        data.extend(b"WAVEfmt ");
        data.extend(16_u32.to_le_bytes());
        data.extend(1_u16.to_le_bytes());
        data.extend(channels.to_le_bytes());
        data.extend(rate.to_le_bytes());
        data.extend((rate * u32::from(channels) * 2).to_le_bytes());
        data.extend((channels * 2).to_le_bytes());
        data.extend(16_u16.to_le_bytes());
        data.extend(b"data");
        data.extend(size.to_le_bytes());
        for sample in samples {
            data.extend(sample.to_le_bytes());
        }
        data
    }
    #[test]
    #[ignore = "Real audio CPU frontend timings; requires ORINFER_AUDIO_CPU_FIXTURE"]
    fn profile_cpu_frontend() {
        let fixture: Value = serde_json::from_slice(
            &std::fs::read(std::env::var("ORINFER_AUDIO_CPU_FIXTURE").unwrap()).unwrap(),
        )
        .unwrap();
        let output = std::path::Path::new(fixture["output"].as_str().unwrap());
        assert!(!output.exists());
        let mut rows = vec![];
        for case in fixture["cases"].as_array().unwrap() {
            let path = std::path::Path::new(case.as_str().unwrap());
            let bytes = std::fs::read(path).unwrap();
            let mut decode_ms = vec![];
            let mut mel_ms = vec![];
            for iteration in 0..8 {
                let start = std::time::Instant::now();
                let samples = decode(
                    bytes.clone(),
                    path.extension().unwrap().to_str().unwrap(),
                    &super::super::preparation::Context::unbounded(),
                )
                .unwrap();
                let decoded = start.elapsed().as_secs_f64() * 1000.;
                let start = std::time::Instant::now();
                let input = orinfer_engine::audio::log_mel(&samples, &|| Ok(())).unwrap();
                std::hint::black_box(input);
                if iteration > 0 {
                    decode_ms.push(decoded);
                    mel_ms.push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
            decode_ms.sort_by(f64::total_cmp);
            mel_ms.sort_by(f64::total_cmp);
            rows.push(
                json!({"path":path,"decode_resample_median_ms":decode_ms[decode_ms.len()/2],
                "log_mel_median_ms":mel_ms[mel_ms.len()/2]}),
            );
        }
        std::fs::write(output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    }

    #[test]
    fn decoder_downmixes_and_resamples_with_preserved_duration() {
        let mut pcm = vec![];
        for _ in 0..4410 {
            pcm.extend([8192_i16, 0]);
        }
        let decoded = decode(
            wav(44100, 2, &pcm),
            "wav",
            &super::super::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!(decoded.len(), 1600);
        assert!(
            decoded[100..1500]
                .iter()
                .all(|v| (*v - 0.125).abs() < 0.001)
        );
        assert!(
            decode(
                vec![0; 12],
                "wav",
                &super::super::preparation::Context::unbounded()
            )
            .is_err()
        );
    }
    #[test]
    fn resampling_preserves_time_alignment_and_rejects_aliases() {
        let decode_pcm = |rate, samples: &[i16]| {
            decode(
                wav(rate, 1, samples),
                "wav",
                &super::super::preparation::Context::unbounded(),
            )
            .unwrap()
        };
        for rate in [8000_u32, 22050, 44100, 48000, 96000, 192000] {
            let mut impulse = vec![0_i16; rate as usize / 10];
            impulse[rate as usize / 20] = 16000;
            let output = decode_pcm(rate, &impulse);
            assert_eq!(output.len(), 1600);
            let peak = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            assert!(peak.abs_diff(800) <= 1, "rate={rate} peak={peak}");
            for frequency in [1000., 12000.] {
                if frequency >= rate as f64 / 2. {
                    continue;
                }
                let input: Vec<_> = (0..rate / 2)
                    .map(|i| {
                        (8000. * (std::f64::consts::TAU * frequency * i as f64 / rate as f64).sin())
                            as i16
                    })
                    .collect();
                let output = decode_pcm(rate, &input);
                assert_eq!(output.len(), 8000);
                let energy = output[512..7488]
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    / 6976.;
                if frequency == 1000. {
                    let expected = (8000. / 32768_f64).powi(2) / 2.;
                    assert!(
                        (energy / expected - 1.).abs() < 0.01,
                        "rate={rate} passband energy={energy}"
                    );
                } else {
                    assert!(
                        energy.sqrt() < 0.0005,
                        "rate={rate} aliased RMS={}",
                        energy.sqrt()
                    );
                }
            }
        }
    }

    #[test]
    fn language_hints_use_model_names() {
        assert_eq!(language("zh").unwrap(), "Chinese");
        assert_eq!(language("English").unwrap(), "English");
        assert!(language("unsupported").is_err());
    }
    #[tokio::test]
    async fn multipart_accepts_large_audio_and_validates_fields() {
        let mut body = b"--asr\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nspeech\r\n--asr\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nzh\r\n--asr\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.flac\"\r\nContent-Type: audio/flac\r\n\r\n".to_vec();
        body.resize(body.len() + (3 << 20), 0);
        body.extend(b"\r\n--asr--\r\n");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "content-type",
            "multipart/form-data; boundary=asr".parse().unwrap(),
        );
        let (request, plain) = transcription_request(multipart(&headers, body.into()).unwrap())
            .await
            .unwrap();
        assert!(!plain);
        assert_eq!(request["model"], "speech");
        assert_eq!(request["asr_language"], "Chinese");
        assert_eq!(
            request["messages"][1]["content"][0]["input_audio"]["format"],
            "flac"
        );
        let duplicate = b"--asr\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\na\r\n--asr\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nb\r\n--asr--\r\n";
        assert!(
            transcription_request(
                multipart(&headers, axum::body::Bytes::from_static(duplicate)).unwrap()
            )
            .await
            .unwrap_err()
            .contains("repeated")
        );
    }
    #[tokio::test]
    async fn transcription_output_parses_metadata_and_preserves_errors() {
        use axum::{Json, response::IntoResponse};
        let output=transcription_response(Json(json!({"choices":[{"finish_reason":"stop","message":{"content":"language Chinese<asr_text>你好。"}}]})).into_response(),false).await;
        let raw = axum::body::to_bytes(output.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&raw).unwrap()["text"],
            "你好。"
        );
        let truncated = transcription_response(
            Json(json!({"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}))
                .into_response(),
            false,
        )
        .await;
        assert_eq!(
            truncated.status(),
            axum::http::StatusCode::UNPROCESSABLE_ENTITY
        );
    }
}

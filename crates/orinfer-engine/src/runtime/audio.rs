use super::*;
impl ModelRuntime {
    pub(super) fn prepare_audio_capacity(
        &self,
        tokens: &[u32],
        inputs: &[crate::audio::AudioInput],
        capacity: usize,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let Some(a) = &self.manifest.audio else {
            return if inputs.is_empty() {
                Ok(())
            } else {
                Err("Model has no audio adapter".into())
            };
        };
        if inputs.len() > 1 {
            return Err("ASR accepts one recording per request".into());
        }
        let mut index = vec![-1_i32; capacity];
        let mut cursor = 0;
        let mut feature_offset = 0;
        for input in inputs {
            let features = input.features()?;
            let start = tokens
                .iter()
                .skip(cursor)
                .position(|&t| t == a.token_id)
                .map(|p| p + cursor)
                .ok_or("Missing ASR audio placeholder")?;
            let end = start
                .checked_add(features)
                .filter(|&n| n <= tokens.len())
                .ok_or("Audio span exceeds prompt")?;
            if tokens[start..end].iter().any(|&t| t != a.token_id)
                || feature_offset + features > a.max_features
            {
                return Err("Invalid audio placeholder span".into());
            }
            for (i, slot) in index[start..end].iter_mut().enumerate() {
                *slot = (feature_offset + i) as i32;
            }
            let chunks = input.frames.div_ceil(100);
            let padded = chunks * 100;
            for first in (0..chunks).step_by(a.stem_chunks) {
                if cancelled() {
                    return Err("Request cancelled during audio encoding".into());
                }
                let count = a.stem_chunks.min(chunks - first);
                let mut raw = Vec::with_capacity(count * 128 * 100 * 2);
                for chunk in first..first + count {
                    for mel in 0..128 {
                        for t in 0..100 {
                            raw.extend_from_slice(
                                &half::f16::from_f32(input.mel[mel * padded + chunk * 100 + t])
                                    .to_bits()
                                    .to_le_bytes(),
                            );
                        }
                    }
                }
                self.upload_bytes(&a.mel, &raw)?;
                self.launch_rows("audio_stem", count, ExecutionPhase::Vision)?;
                let live = (features - first * 13).min(count * 13);
                self.execution.copy_range(
                    &a.stem_output,
                    &a.encoder_input,
                    first * 13 * 1024 * 2,
                    live * 1024 * 2,
                )?;
            }
            self.upload_ids(&a.length, &[features as u32])?;
            self.launch_rows("audio_encoder", features, ExecutionPhase::Vision)?;
            self.execution.copy_range(
                &a.output,
                &a.features,
                feature_offset * a.hidden * 2,
                features * a.hidden * 2,
            )?;
            feature_offset += features;
            cursor = end;
        }
        if tokens.iter().filter(|&&t| t == a.token_id).count() != feature_offset {
            return Err("Unbound audio placeholder".into());
        }
        self.upload_bytes(
            &a.feature_index,
            &index
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let positions: Vec<u32> = (0..capacity as u32).flat_map(|i| [i, i, i]).collect();
        self.upload_ids(&a.positions, &positions)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(serde::Deserialize)]
    struct Case {
        input: crate::audio::AudioInput,
        reference: std::path::PathBuf,
        output: Option<std::path::PathBuf>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        cases: Vec<Case>,
    }
    #[test]
    #[ignore = "Requires ASR weights, FP16 references and exclusive GPU lock"]
    fn encoder_matches_fp16_and_replays_changed_audio() {
        let f: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_AUDIO_FIXTURE").unwrap(),
        ))
        .unwrap();
        assert!(f.cases.len() >= 2);
        let model = ModelRuntime::load_with_options(
            &f.model,
            LoadOptions {
                cuda_graph: crate::execution::CudaGraphMode::Full,
                ..Default::default()
            },
        )
        .unwrap();
        let a = model.manifest.audio.as_ref().unwrap();
        let mut originals = Vec::new();
        for replay in 0..2 {
            for (i, case) in f.cases.iter().enumerate() {
                let rows = case.input.features().unwrap();
                let tokens = vec![a.token_id; rows];
                model
                    .prepare_audio_capacity(
                        &tokens,
                        std::slice::from_ref(&case.input),
                        rows,
                        &|| false,
                    )
                    .unwrap();
                let raw = model
                    .execution
                    .download_bytes(&a.features, rows * a.hidden * 2)
                    .unwrap();
                let actual = floats(&raw, crate::artifact::Dtype::F16);
                if replay == 0
                    && let Some(path) = &case.output
                {
                    std::fs::write(
                        path,
                        actual
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                }
                let reference: Vec<_> = std::fs::read(&case.reference)
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                assert_eq!(reference.len(), actual.len());
                assert!(actual.iter().all(|v| v.is_finite()));
                let error: f64 = actual
                    .iter()
                    .zip(&reference)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum();
                let energy: f64 = reference.iter().map(|&v| (v as f64).powi(2)).sum();
                let relative_rmse = (error / energy.max(1e-30)).sqrt();
                println!(
                    "audio case={i} replay={replay} rows={rows} relative_rmse={relative_rmse}"
                );
                assert!(relative_rmse < 0.02, "audio FP16 encoder mismatch");
                if replay == 0 {
                    originals.push(raw);
                } else {
                    assert_eq!(raw, originals[i], "graph replay changed audio features");
                }
            }
        }
    }
}

//! Native audio contracts; CPU frontend produces validated log-mel inputs.
pub use orinfer_model_sdk::audio::*;

use crate::artifact::Result;
use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::{Arc, OnceLock};
struct Frontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    filters: Vec<Vec<(usize, f32)>>,
}
fn frontend() -> &'static Frontend {
    static FRONTEND: OnceLock<Frontend> = OnceLock::new();
    FRONTEND.get_or_init(|| {
        let mut planner = RealFftPlanner::new();
        let mel = |hz: f64| {
            if hz < 1000.0 {
                hz / (200.0 / 3.0)
            } else {
                15.0 + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
            }
        };
        let hz = |m: f64| {
            if m < 15.0 {
                m * (200.0 / 3.0)
            } else {
                1000.0 * ((m - 15.0) * (6.4_f64.ln() / 27.0)).exp()
            }
        };
        let points: Vec<_> = (0..130)
            .map(|i| hz(i as f64 * mel(8000.0) / 129.0))
            .collect();
        let filters = (0..128)
            .map(|m| {
                (0..201)
                    .filter_map(|i| {
                        let f = i as f64 * 40.0;
                        let weight = ((f - points[m]) / (points[m + 1] - points[m]))
                            .min((points[m + 2] - f) / (points[m + 2] - points[m + 1]))
                            .max(0.0)
                            * 2.0
                            / (points[m + 2] - points[m]);
                        (weight > 0.0).then_some((i, weight as f32))
                    })
                    .collect()
            })
            .collect();
        Frontend {
            fft: planner.plan_fft_forward(400),
            window: (0..400)
                .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 400.0).cos())
                .collect(),
            filters,
        }
    })
}
/// Whisper-compatible FP32 STFT and Slaney log-mel, with checkpoint padding.
pub fn log_mel(samples: &[f32], check: &(impl Fn() -> Result<()> + Sync)) -> Result<AudioInput> {
    if samples.is_empty()
        || samples.len() > 16000 * 1200
        || samples.iter().any(|v| !v.is_finite() || v.abs() > 16.0)
    {
        return Err("Audio requires finite PCM and a duration up to 1200 seconds".into());
    }
    let n = samples.len().max(8000);
    let frames = n / 160;
    let padded = frames.div_ceil(100) * 100;
    let f = frontend();
    // Frame groups share Rayon's existing work-stealing pool with tokenization.
    // FFT/reduction order stays local to each frame; only independent groups run in parallel.
    let parts: Result<Vec<_>> = (0..frames)
        .into_par_iter()
        .step_by(100)
        .map_init(
            || {
                (
                    vec![0.0; 400],
                    f.fft.make_output_vec(),
                    f.fft.make_scratch_vec(),
                    [0.0_f32; 201],
                )
            },
            |(bins, spectrum, scratch, power), first| {
                check()?;
                let count = (frames - first).min(100);
                let mut data = vec![0.0_f32; 128 * count];
                let mut maximum = -10.0_f32;
                for t in first..first + count {
                    for (i, value) in bins.iter_mut().enumerate() {
                        let index = t as isize * 160 + i as isize - 200;
                        let index = if index < 0 {
                            -index
                        } else if index >= n as isize {
                            2 * n as isize - 2 - index
                        } else {
                            index
                        } as usize;
                        *value = samples.get(index).copied().unwrap_or(0.0) * f.window[i];
                    }
                    f.fft
                        .process_with_scratch(bins, spectrum, scratch)
                        .map_err(|e| e.to_string())?;
                    for (p, value) in power.iter_mut().zip(spectrum.iter()) {
                        *p = value.norm_sqr();
                    }
                    for m in 0..128 {
                        let value = f.filters[m]
                            .iter()
                            .map(|&(i, w)| power[i] * w)
                            .sum::<f32>()
                            .max(1e-10)
                            .log10();
                        maximum = maximum.max(value);
                        data[m * count + t - first] = value;
                    }
                }
                Ok((first, count, maximum, data))
            },
        )
        .collect();
    let parts = parts?;
    let maximum = parts
        .iter()
        .fold(-10.0_f32, |value, part| value.max(part.2));
    let mut mel = vec![0.0_f32; 128 * padded];
    for (first, count, _, data) in parts {
        for m in 0..128 {
            for t in 0..count {
                mel[m * padded + first + t] = (data[m * count + t].max(maximum - 8.0) + 4.0) / 4.0;
            }
        }
    }
    Ok(AudioInput { frames, mel })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mel_is_identical_across_parallel_schedules_and_partial_frame_groups() {
        let samples: Vec<_> = (0..160 * 305)
            .map(|i| (i as f32 * 0.07).sin() * 0.1)
            .collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| log_mel(&samples, &|| Ok(())).unwrap())
        };
        let first = run(1);
        let second = run(4);
        assert_eq!(first.frames, 305);
        assert_eq!(first.mel, second.mel);
        for band in second.mel.as_chunks::<400>().0 {
            assert!(band[305..].iter().all(|&v| v == 0.));
        }
    }
    #[test]
    #[ignore = "Requires Transformers-generated audio references"]
    fn checkpoint_mel_reference() {
        let root = std::path::PathBuf::from(std::env::var("ORINFER_AUDIO_REFERENCE").unwrap());
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("frontend-reference.json")).unwrap())
                .unwrap();
        let read = |name: &str| {
            std::fs::read(root.join(name))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        for case in cases.as_array().unwrap() {
            let pcm = read(case["pcm"].as_str().unwrap());
            let expected = read(case["mel"].as_str().unwrap());
            let actual = log_mel(&pcm, &|| Ok(())).unwrap();
            assert_eq!(actual.frames, case["frames"].as_u64().unwrap() as usize);
            assert_eq!(actual.mel.len(), expected.len());
            let error = actual
                .mel
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("Audio {} max mel error {error}", case["name"]);
            assert!(error < 0.0005);
        }
    }
    #[test]
    fn silence_and_short_tail_have_checkpoint_geometry() {
        let x = log_mel(&[0.0; 160], &|| Ok(())).unwrap();
        assert_eq!(x.frames, 50);
        assert_eq!(x.features().unwrap(), 7);
        assert_eq!(x.mel.len(), 12800);
        for row in x.mel.as_chunks::<100>().0 {
            assert!(row[..50].iter().all(|v| (*v + 1.5).abs() < 1e-6));
            assert!(row[50..].iter().all(|v| *v == 0.0));
        }
        assert!(log_mel(&[f32::NAN], &|| Ok(())).is_err());
        assert!(log_mel(&[0.0; 8000], &|| Err("cancelled".into())).is_err());
    }
}

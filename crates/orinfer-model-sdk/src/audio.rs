//! Qwen ASR audio feature and execution contracts.
use crate::{
    artifact::{Access, Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioInput {
    /// Valid mel frames before right padding; mel is [128, ceil(frames/100)*100].
    pub frames: usize,
    pub mel: Vec<f32>,
}
impl AudioInput {
    pub fn features(&self) -> Result<usize> {
        if self.frames == 0
            || self.frames > 120000
            || self.mel.len() != 128 * self.frames.div_ceil(100) * 100
            || self.mel.iter().any(|v| !v.is_finite())
        {
            return Err("Invalid ASR mel geometry or values".into());
        }
        Ok((self.frames / 100) * 13 + (self.frames % 100).div_ceil(8))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSpec {
    pub hidden: usize,
    pub max_features: usize,
    pub stem_chunks: usize,
    pub mel: String,
    pub stem_output: String,
    pub encoder_input: String,
    pub length: String,
    pub output: String,
    pub features: String,
    pub feature_index: String,
    pub positions: String,
    pub token_id: u32,
}
impl AudioSpec {
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        if self.hidden != 2048
            || self.max_features != 15600
            || self.stem_chunks != 32
            || self.token_id != 151676
        {
            return Err("Unsupported ASR feature contract".into());
        }
        for (name, dtype, shape) in [
            (&self.mel, Dtype::F16, vec![self.stem_chunks, 128, 100]),
            (
                &self.stem_output,
                Dtype::F16,
                vec![self.stem_chunks * 13, 1024],
            ),
            (
                &self.encoder_input,
                Dtype::F16,
                vec![self.max_features, 1024],
            ),
            (
                &self.output,
                Dtype::F16,
                vec![self.max_features, self.hidden],
            ),
            (
                &self.features,
                Dtype::F16,
                vec![self.max_features, self.hidden],
            ),
            (&self.feature_index, Dtype::I32, vec![m.max_context]),
            (&self.positions, Dtype::I32, vec![m.max_context, 3]),
            (&self.length, Dtype::I32, vec![1]),
        ] {
            if m.buffers.iter().find(|b| &b.name == name).is_none_or(|b| {
                b.dtype != dtype
                    || b.shape != shape
                    || b.access != Access::ReadWrite
                    || b.data.is_some()
            }) {
                return Err(format!("Invalid audio buffer {name}"));
            }
        }
        for program in ["audio_stem", "audio_encoder"] {
            if m.programs.get(program).is_none_or(Vec::is_empty) {
                return Err(format!("Missing audio program {program}"));
            }
        }
        Ok(())
    }
    pub fn expand(&self, tokens: &[u32], audio: &[AudioInput], context: usize) -> Result<Vec<u32>> {
        let mut inputs = audio.iter();
        let mut output = Vec::new();
        for &token in tokens {
            if token == self.token_id {
                let count = inputs
                    .next()
                    .ok_or("Audio placeholder has no input")?
                    .features()?;
                if output.len().checked_add(count).is_none_or(|n| n > context) {
                    return Err("Audio exceeds context".into());
                }
                output.extend(std::iter::repeat_n(token, count));
            } else {
                output.push(token);
            }
            if output.len() > context {
                return Err("ASR prompt exceeds context".into());
            }
        }
        if inputs.next().is_some() {
            return Err("Audio input has no placeholder".into());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> AudioSpec {
        AudioSpec {
            hidden: 2048,
            max_features: 15600,
            stem_chunks: 32,
            mel: "mel".into(),
            stem_output: "stem".into(),
            encoder_input: "encoder".into(),
            length: "length".into(),
            output: "output".into(),
            features: "features".into(),
            feature_index: "index".into(),
            positions: "positions".into(),
            token_id: 151676,
        }
    }
    #[test]
    fn partial_second_audio_expands_exactly_and_enforces_placeholder_ownership() {
        let audio = AudioInput {
            frames: 101,
            mel: vec![0.0; 25600],
        };
        assert_eq!(audio.features().unwrap(), 14);
        let a = spec();
        assert_eq!(
            a.expand(&[1, a.token_id, 2], std::slice::from_ref(&audio), 16)
                .unwrap()
                .len(),
            16
        );
        assert!(
            a.expand(&[1, a.token_id, 2], std::slice::from_ref(&audio), 15)
                .is_err()
        );
        assert!(a.expand(&[1, 2], std::slice::from_ref(&audio), 16).is_err());
        assert!(a.expand(&[a.token_id], &[], 16).is_err());
        let mut invalid = audio;
        invalid.mel.pop();
        assert!(invalid.features().is_err());
    }
}

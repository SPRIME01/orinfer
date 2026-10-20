//! GPU sampling shares the ordinary request's history, seed and grammar law.
use super::*;
impl ModelRuntime {
    pub(super) fn gpu_sample(
        &self,
        history: &[u32],
        options: &crate::sampling::Options,
        step: usize,
        mask: Option<&[u32]>,
    ) -> Result<Option<u32>> {
        let Some(spec) = &self.manifest.sampling else {
            return Ok(None);
        };
        // Probability reporting and speculative p/q laws retain the reference
        // path until their compact GPU reporting/verification ABI is available.
        if options.temperature == 0.0 || options.top_logprobs.is_some() {
            return Ok(None);
        }
        options.validate()?;
        if history.len() > self.manifest.max_context {
            return Err("Sampling history exceeds context".into());
        }
        let use_history = options.repetition_penalty != 1.
            || options.presence_penalty != 0.
            || options.frequency_penalty != 0.;
        if use_history {
            self.upload_ids(&spec.history, history)?;
        }
        let bias: Vec<_> = options
            .logit_bias
            .iter()
            .filter(|(id, _)| **id < self.manifest.vocab as u32)
            .collect();
        let ids: Vec<_> = bias.iter().map(|(id, _)| **id).collect();
        let values: Vec<_> = bias.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
        self.upload_ids(&spec.bias_ids, &ids)?;
        self.upload_bytes(&spec.bias_values, &values)?;
        if let Some(mask) = mask {
            // Tokenizers may omit padded output-head IDs. Missing mask words
            // prohibit those IDs, matching the CPU law's bounds-checked lookup.
            self.upload_ids(&spec.mask, &padded_mask(mask, self.manifest.vocab))?;
        }
        self.upload_ids(
            &spec.control,
            &[
                if use_history { history.len() as u32 } else { 0 },
                bias.len() as u32,
                u32::from(mask.is_some()),
                options.top_k.min(self.manifest.vocab) as u32,
            ],
        )?;
        let parameters: Vec<_> = [
            options.temperature,
            options.top_p,
            options.presence_penalty,
            options.frequency_penalty,
            options.repetition_penalty,
            crate::sampling::counter_uniform(options.seed, 0, step as u64),
        ]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect();
        self.upload_bytes(&spec.parameters, &parameters)?;
        self.execution
            .submit_program(&spec.program, ExecutionPhase::Decode)?;
        let output = self.read_controls(&spec.output, 2)?;
        if output[1] != 0 || output[0] as usize >= self.manifest.vocab {
            return Err("GPU sampling has invalid logits or empty probability support".into());
        }
        Ok(Some(output[0]))
    }
}

fn padded_mask(mask: &[u32], vocab: usize) -> Vec<u32> {
    let mut result = vec![0; vocab.div_ceil(32)];
    for (destination, source) in result.iter_mut().zip(mask) {
        *destination = *source;
    }
    if !vocab.is_multiple_of(32)
        && let Some(last) = result.last_mut()
    {
        *last &= u32::MAX >> (32 - vocab % 32);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::padded_mask;
    #[test]
    fn tokenizer_mask_excludes_head_padding_and_unused_tail_bits() {
        assert_eq!(padded_mask(&[u32::MAX], 65), [u32::MAX, 0, 0]);
        assert_eq!(padded_mask(&[u32::MAX; 4], 65), [u32::MAX, u32::MAX, 1]);
        assert_eq!(padded_mask(&[], 32), [0]);
        assert!(padded_mask(&[1], 0).is_empty());
    }
}

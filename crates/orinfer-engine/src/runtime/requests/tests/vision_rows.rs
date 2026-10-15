//! Real-weight continuous vision lengths, tail guards, Graph replay and multi-image isolation.
use super::*;
use std::path::Path;
#[derive(serde::Deserialize)]
struct VisionFixture {
    baseline: PathBuf,
    model: PathBuf,
    output: PathBuf,
}
#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn validate_flash_dynamic_vision() {
    let f: VisionFixture =
        crate::model::read(Path::new(&std::env::var("ORINFER_VISION_FIXTURE").unwrap())).unwrap();
    let grids = [
        (2, 2),
        (10, 14),
        (16, 16),
        (18, 30),
        (32, 32),
        (48, 64),
        (64, 128),
    ];
    let mut reference = vec![];
    for (kind, root) in [(0, &f.baseline), (1, &f.model)] {
        let model = ModelRuntime::load_with_options(
            root,
            LoadOptions {
                cuda_graph: crate::execution::CudaGraphMode::Full,
                ..Default::default()
            },
        )
        .unwrap();
        let v = model.manifest.vision.as_ref().unwrap();
        for (index, &(height, width)) in grids.iter().enumerate() {
            let pixels = (0..height * width * v.patch_values().unwrap())
                .map(|n| ((n * 37 % 199) as f32 - 99.) / 100.)
                .collect();
            let image = crate::vision::ImageInput {
                grid_height: height,
                grid_width: width,
                pixels,
            };
            let tokens = std::iter::once(v.vision_start_token_id)
                .chain(std::iter::repeat_n(v.image_token_id, height * width / 4))
                .chain([v.vision_end_token_id])
                .collect::<Vec<_>>();
            let n = height * width / 4 * v.hidden * 2;
            for replay in 0..2 {
                // Changed pixels must survive cached graph reuse.
                let mut changed = image.clone();
                if replay == 1 {
                    for value in &mut changed.pixels {
                        *value = -*value;
                    }
                }
                model
                    .prepare_visual(&tokens, &[changed], &|| false)
                    .unwrap();
                let actual = model.execution.download_bytes(&v.features, n).unwrap();
                if kind == 0 {
                    reference.push(actual);
                } else {
                    assert_eq!(
                        actual,
                        reference[index * 2 + replay],
                        "grid={height}x{width}, replay={replay}"
                    );
                }
            }
        }
        // Different images use the same workspace and retain their feature order.
        let images: Vec<_> = [(10, 14), (18, 30)]
            .into_iter()
            .enumerate()
            .map(|(i, (h, w))| crate::vision::ImageInput {
                grid_height: h,
                grid_width: w,
                pixels: (0..h * w * v.patch_values().unwrap())
                    .map(|n| ((n * 37 % 199) as f32 - 99.) / 100. * if i == 0 { 1. } else { -1. })
                    .collect(),
            })
            .collect();
        let tokens = images
            .iter()
            .flat_map(|image| {
                std::iter::once(v.vision_start_token_id)
                    .chain(std::iter::repeat_n(
                        v.image_token_id,
                        image.grid_height * image.grid_width / 4,
                    ))
                    .chain([v.vision_end_token_id])
            })
            .collect::<Vec<_>>();
        model.prepare_visual(&tokens, &images, &|| false).unwrap();
        let n = (140 + 540) / 4 * v.hidden * 2;
        let actual = model.execution.download_bytes(&v.features, n).unwrap();
        if kind == 0 {
            reference.push(actual);
        } else {
            assert_eq!(actual, *reference.last().unwrap());
        }
    }
    std::fs::write(&f.output,serde_json::to_vec_pretty(&serde_json::json!({"complete":true,"image_lengths":[4,140,256,540,1024,3072,8192],"replays":2,"multi_image":true,"features_exact":true})).unwrap()).unwrap();
}

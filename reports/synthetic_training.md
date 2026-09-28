# Synthetic training run

Run date: 2026-09-08

- Dataset: 16 synthetic subjects × 2 sessions × 8 synchronized timesteps × 4 camera views (1,024 generated cartoon crops).
- Split: 12 synthetic subjects for training; 4 unseen synthetic subjects for validation.
- Model: `QualityAwareMultiViewFER(tiny=True)`, quality-aware four-view fusion with the lightweight CNN pathway.
- Training: 12 epochs, batch size 4, learning rate 0.003, image size 64 × 64.
- Final synthetic validation frame accuracy: **92.19%** (59 / 64 frames).

This result evaluates only an intentionally simplified line-art fixture with class-coloured backgrounds. It is a pipeline smoke-test, not a clinical, demographic, or real-world facial-expression result. The machine-readable epoch history is in `runs/logs/synthetic_fer.metrics.json`; descriptive output for the eight validation sessions is in `data/features/synthetic_session_features.csv`.

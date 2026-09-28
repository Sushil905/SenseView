# Multi-View Facial Expression Analytics for Social Play

An ethical deep-learning research prototype inspired by *Multi-View Facial Expressions Analysis of Autistic Children in Social Play* (Zeng et al., IEEE TAFFC, 2025, DOI: 10.1109/TAFFC.2025.3557458).

It accepts synchronized face crops from multiple cameras, predicts per-frame facial-expression distributions, selects the lowest-yaw view at each timestep, and creates **session-level descriptive features**: expression proportions, transition rate, and multiscale entropy. A separately trained, clinician-approved research classifier may consume those features.

> **Important:** This project is for research and educational use. It must not diagnose autism, label a child, replace clinical assessment, or be used for surveillance. Any human-subject data needs ethics approval, informed consent/assent, secure storage, de-identification, and a qualified clinical interpretation.

## Project layout

```
configs/                       # training, inference, dataset YAML templates
data/
  raw/view1..view4/            # optional source images (ignored by git)
  processed/                   # prepared crops and manifests
  features/                    # exported session descriptors
  labels.csv                   # empty expression-label manifest template
  synthetic/                   # artificial cartoon fixture for the local demo
notebooks/                     # EDA and feature review
reports/figures/               # exported plots
reports/metrics/                # evaluation summaries
runs/checkpoints/              # model weights
runs/logs/                     # training histories
runs/tensorboard/              # TensorBoard event files
src/mvfer/
  dataset/                      # loader, transforms, session sampler
  detection/, pose/             # optional interfaces; models are not bundled
  fer/, fusion/                 # expression model and fusion interfaces
  features/                     # probability summaries, entropy, extraction
  classifiers/, evaluation/     # exploratory estimators and metrics
  utils/                        # seed, logging, I/O helpers
  train.py, infer.py, analytics.py
web/                            # dashboard HTML, CSS, JavaScript
tests/                          # dependency-light analytics checks
```

## Setup

Use Python 3.10+ and create an isolated environment. Install PyTorch appropriate for your machine first, then:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

To open the local FastAPI dashboard:

```bash
.venv/bin/python app.py
```

Then visit `http://127.0.0.1:8000`. FastAPI docs are available at `http://127.0.0.1:8000/docs`. Install the optional RetinaFace dependency with `pip install -r requirements-vision.txt`.

## Data contract

Do not store names, dates of birth, diagnoses, or identifying information in the manifest. Use a random `subject_id`, a random `session_id`, and paths to **already face-cropped** images. Each `(session_id, timestep)` must have the same expected views (`cam_01` through `cam_04` by default). `expression` is optional for inference and is one of `neutral,happiness,sadness,anger,fear,disgust,surprise` for supervised FER training.

```bash
python -m mvfer.train \
  --manifest data/manifest.example.csv --data-root . \
  --epochs 20 --batch-size 4 --output runs/checkpoints/fer.pt

python -m mvfer.infer \
  --manifest data/manifest.example.csv --data-root . \
  --checkpoint runs/checkpoints/fer.pt --output data/features/session_features.csv
```

The example manifest contains placeholder paths and will intentionally fail until you supply consented, de-identified crops. Split data by **subject**, never by frame, to avoid leakage. Report balanced accuracy, macro F1, sensitivity, specificity, calibration, confidence intervals, and performance stratified by relevant demographic and capture-quality groups.

## Synthetic demo dataset and training

The project includes a reproducible, artificial line-art dataset so the full pipeline can be tested without collecting or distributing biometric data. It contains cartoon faces with synthetic multi-view shifts/occlusions; successful performance on it says nothing about real children or clinical use.

```bash
export PYTHONPATH=src
.venv/bin/python -m mvfer.synthetic --output data/synthetic
.venv/bin/python -m mvfer.train --manifest data/synthetic/train.csv --val-manifest data/synthetic/val.csv \
  --data-root . --tiny --image-size 64 --epochs 12 --batch-size 4 --output runs/checkpoints/synthetic_fer.pt
.venv/bin/python -m mvfer.infer --manifest data/synthetic/val.csv --data-root . \
  --checkpoint runs/checkpoints/synthetic_fer.pt --output data/features/synthetic_session_features.csv
```

The included local run is documented in `reports/synthetic_training.md`; its checkpoint and history live under `runs/checkpoints/` and `runs/logs/`. `--tiny` is deliberately a legacy synthetic smoke-test path; use the standard model only after a properly governed real-data study is in place.

### Requested model stack

- **Dataset:** RAF-DB basic-expression aligned images are read from the official partition annotation list by `RAFDBDataset`. The RAF-DB images train a frame-level expression model; the dataset contains still images and does not provide synchronized camera sequences.
- **Face detector:** RetinaFace is an optional preprocessing step for raw multi-view clips. It writes face crops and a landmark-based yaw proxy into a processed manifest. Already aligned RAF-DB face images do not need this step.
- **Emotion model:** torchvision ViT-Base/16, fine-tuned on RAF-DB with `python -m mvfer.train_rafdb`.
- **Temporal module and fusion:** for actual synchronized clips, `ViTBaseLSTM` selects the view with the lowest absolute yaw at each timestep, then models selected ViT embeddings with an LSTM. Multi-view manifests must include `yaw_degrees`; use subject-disjoint train/validation splits.
- **Classifier:** an optional session-feature SVM can be trained with `python -m mvfer.train_svm`. It needs caller-provided session targets; evaluation holds out subjects. It is a generic research classifier and is not an autism diagnostic system.
- **Dashboard:** FastAPI + Uvicorn serves the local UI and accepts synchronized image sequences from four cameras.

RAF-DB and the RetinaFace package are not bundled. Obtain RAF-DB from its owners under its terms. The YAML files document paths and settings; the current training CLI still takes explicit command-line options.

## RAF-DB and synchronized-clip workflow

Put the RAF-DB aligned images and `list_patition_label.txt` under `data/raw/rafdb/` as shown in [data/rafdb_layout.md](data/rafdb_layout.md), then fine-tune the frame model:

```bash
.venv/bin/python -m mvfer.train_rafdb \
  --image-root data/raw/rafdb/aligned \
  --annotations data/raw/rafdb/list_patition_label.txt \
  --output runs/checkpoints/rafdb_vit_base.pt
```

For synchronized clips, create a manifest with `subject_id,session_id,timestep,view_id,image_path,expression`, run RetinaFace preprocessing to add `yaw_degrees`, split by subject, then train the temporal model from the RAF-DB ViT checkpoint:

```bash
.venv/bin/python -m mvfer.detection.preprocess_manifest --manifest data/raw/multiview_manifest.csv --data-root .
.venv/bin/python -m mvfer.train --manifest data/processed/multiview_train.csv \
  --val-manifest data/processed/multiview_val.csv --data-root . \
  --fer-checkpoint runs/checkpoints/rafdb_vit_base.pt \
  --output runs/checkpoints/fer.pt
```

The preprocessing command writes `data/processed/multiview_manifest.csv`; prepare subject-level train and validation manifests from it before training. The dashboard uses `runs/checkpoints/fer.pt` when present and otherwise keeps the bundled synthetic demo model.

Train an optional generic session SVM after exporting features and providing a separate session target table with `subject_id`, `session_id`, and a caller-selected target column:

```bash
.venv/bin/python -m mvfer.train_svm \
  --features data/features/session_features.csv \
  --labels data/session_targets.csv --target-column target
```

The SVM evaluates a subject-disjoint holdout. Do not put names or direct identifiers in these files; use approved, de-identified study labels.

## Model

The current stack uses torchvision ViT-Base/16 frame embeddings, a bidirectional LSTM across timesteps, and lowest-absolute-yaw camera selection. The earlier quality-fusion Transformer remains only to load the included artificial demo checkpoint. Neither model claims to reproduce the referenced paper's private dataset, annotations, or published outcomes.

## Responsible deployment checklist

- Obtain approval, consent, retention/deletion rules, access logging, and encryption before data collection.
- Keep a human in the loop; show model confidence and capture quality, not categorical child-level conclusions.
- Train and validate on held-out subjects, preferably an external site; audit error by age, gender, ethnicity, camera angle, and occlusion.
- Version datasets/models, document annotation agreement, and prohibit downstream diagnostic/eligibility decisions.

## Reference

Zeng, J. et al. (2025). *Multi-View Facial Expressions Analysis of Autistic Children in Social Play.* IEEE Transactions on Affective Computing, 16(3), 2200–2214. https://doi.org/10.1109/TAFFC.2025.3557458

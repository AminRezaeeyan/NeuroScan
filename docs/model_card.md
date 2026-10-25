# Model card: NeuroScan brain tumor classifier

| | |
| --- | --- |
| Registered model | `neuroscan-classifier`, alias `champion` (version 7) |
| Architecture | EfficientNetB0 (ImageNet-pretrained, fine-tuned), 4.05M parameters |
| Package | Network + temperature scaling + out-of-distribution detector, one MLflow `pyfunc` |
| Input | One 2D brain MRI slice (PNG, JPEG or BMP), resized to 224x224 |
| Output | Status, predicted class, calibrated probabilities for 4 classes, OOD score, Grad-CAM heatmap |
| Classes | glioma, meningioma, no tumor, pituitary tumor |
| Author | Amin Rezaeeyan |
| License | MIT (code); dataset CC BY 4.0 |

## Intended use

- **Intended:** education, research and demonstration of a careful ML workflow for
  medical image classification: leakage-aware evaluation, calibration, refusal of
  unsuitable inputs and explainability checks.
- **Not intended:** diagnosis, screening, triage or any decision about a patient. The
  model is not a medical device and has not been clinically validated.
- **Users:** ML practitioners and students. Any output shown to others must carry the
  "research demo, not a medical device" notice, as the API and demo page do.

## Training data

The [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(Masoud Nickparvar, CC BY 4.0, 7,200-image version): 2D slices from several public
collections, mixed views, no patient IDs.

- Training used 4,897 images; validation 1,050; test 1,050 (`splits/v1.csv`).
- Near-duplicates were grouped and every group kept in one split; 203 pre-augmented files
  were excluded. See [data_audit.md](data_audit.md).
- Augmentation: horizontal flip, ±10% rotation and zoom, ±15% brightness.

**Known data issues:** class is partly predictable from image source (a metadata-only
probe reaches 79% accuracy); 27 duplicate groups carry conflicting labels; no patient-level
separation is possible.

## Evaluation

Held-out test split (1,050 images). "Group-weighted" means each duplicate group counts
once; intervals are 95% cluster-bootstrap.

| Metric | Value |
| --- | --- |
| Macro F1 (group-weighted) | 0.955 (0.936–0.972) |
| Accuracy (per image) | 96.0% |
| Tumor sensitivity (group-weighted) | 99.0% |
| Tumor specificity (group-weighted) | 97.2% |

### Per class (per image)

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Glioma | 0.962 | 0.944 | 0.953 |
| Meningioma | 0.929 | 0.921 | 0.925 |
| No tumor | 0.982 | 0.993 | 0.987 |
| Pituitary | 0.964 | 0.978 | 0.971 |

Meningioma is the weakest class; most errors are glioma/meningioma confusions.

### With refusal rules applied (the served behavior)

| | Test split |
| --- | ---: |
| Refused as out of distribution | 2.2% |
| Flagged uncertain (calibrated confidence < 0.86) | 8.7% |
| Accuracy on the answered rest | 98.8% |

### Calibration

Temperature T = 2.02, fitted on validation. Expected calibration error on test: 0.024
before, 0.011 after. Probabilities returned by the API are calibrated.

### Out-of-distribution rejection

| Source (public images) | Refused |
| --- | ---: |
| Natural photos | 100% |
| Chest X-ray | 100% |
| Lumbar spine MRI | 100% |
| CT slices | 94.3% |

AUROC 0.998 over all sources (maximum softmax probability, for comparison: 0.932). The
threshold refuses about 2% of real validation MRIs by design.

### Explainability sanity check

The Grad-CAM peak lies inside the brain for 99.6% of test images (54% would be expected
by chance) and for 97.6% of misclassified ones.

### Shortcut check

The model is 10.5 points more accurate on test images whose class a metadata-only probe
can guess than on those it cannot: part of its performance comes from image-source cues
rather than anatomy.

## Limitations and failure modes

- **Image-level, single-dataset evaluation.** No patient IDs; one public collection.
  Performance on other scanners, protocols or hospitals is unknown.
- **Source shortcut.** See above; tumor-versus-no-tumor results in particular are likely
  optimistic.
- **OOD means "unlike the training data".** Genuine brain MRIs framed differently (for
  example, a sagittal slice with text overlays and more neck) can be refused. Some CT
  slices (about 6% in testing) are accepted.
- **2D slices only**, no volumetric context, no other tumor types or pathologies.
- **Label noise** between glioma and meningioma in the source data.
- **Grad-CAM** shows where the model looked, not why; it is not a clinical explanation.

## Ethical considerations

Misplaced trust is the main risk: a fluent interface with confident numbers can look
authoritative. The system counters this with calibrated probabilities, explicit
"uncertain" and "refused" answers, a disclaimer on every page and response, and no public
deployment. Uploaded images are processed in memory and never stored or logged.

## Provenance

Training and evaluation are tracked in MLflow: each run records its git commit, split
hash, config, per-epoch history and evaluation artifacts. The champion was chosen by a
rule fixed before training ([model_selection.md](model_selection.md)), using validation
data only. Full rationale: [design.md](design.md).

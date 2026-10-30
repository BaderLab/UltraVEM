# UltraVEM — Classification benchmark and embedding analysis

Evaluation code for the UltraVEM 3D foundation model for volume electron microscopy (vEM).
It takes a pre-trained encoder, extracts a single embedding vector per volume, and measures
how well those embeddings separate biological categories (organism, tissue, imaging
modality) with **frozen features** — no fine-tuning.

Pre-training lives in the companion `nnssl` repository; this one starts from a checkpoint.

| Script | Purpose |
|---|---|
| [`extract_embeddings.py`](extract_embeddings.py) | Run volumes through the frozen encoder → `embeddings.npz` |
| [`classification_benchmark.py`](classification_benchmark.py) | k-NN + linear probe + data-efficiency curves → results JSON and figures |

---

## The dataset annotation table

Both scripts share one table that maps each source dataset to its biological and imaging
metadata. Nothing dataset-specific is hard-coded in the scripts — point `--metadata` at your
table. [`metadata/dataset_metadata.template.csv`](metadata/dataset_metadata.template.csv)
documents the schema with example rows.

| Column | Required | Meaning |
|---|---|---|
| `dataset_id` | yes | Source dataset identifier; must match the per-image label in `embeddings.npz` |
| `organism` | no | e.g. `Mouse`, `Drosophila`, `C. elegans` |
| `sample_type` | no | e.g. `Tissue`, `Cell line`, `Primary cell` |
| `tissue_fine` / `tissue_coarse` | no | Anatomy at two granularities |
| `umap_fine` / `umap_coarse` | no | Plot labels. Derived as `"<organism> - <tissue>"` when absent |
| `modality` | no | e.g. `FIB-SEM`, `SBF-SEM`, `ssTEM` |
| `voxel_size_nm_xyz` | no | `"x,y,z"` in nanometres, e.g. `"4,4,30"` |
| `aliases` | no | Semicolon-separated alternative names resolving to this `dataset_id` |

`.csv`, `.tsv` and `.xlsx` are all accepted. Column names are matched case-insensitively,
and the longer spreadsheet-style names (`Tissue_Organ_Fine`, `UMAP_Label_Coarse`, …) work
too, so an existing export can be used unchanged.

By default the scripts look for `metadata/dataset_metadata.csv`; supply `--metadata` to use
a table elsewhere.

---

## 1. Extract embeddings

Requires `nnssl` to be importable (`pip install -e /path/to/nnssl`) and the nnssl
environment variables to be set, since volumes are read from `nnssl_preprocessed`.

```bash
python extract_embeddings.py \
    --dataset_id <DATASET_ID> \
    --config <CONFIGURATION> \
    --trainer <TRAINER_CLASS> \
    --plans <PLANS_IDENTIFIER> \
    --checkpoint /path/to/checkpoint.pth \
    --use_patch_size \
    --output_file embeddings.npz \
    --metadata /path/to/dataset_metadata.csv
```

`--use_patch_size` reads the patch size from the training plan, which is the recommended
option; otherwise set it explicitly with `--patch_size <D> <H> <W>`. Every volume must be
cropped/padded to a common size to be batched, so one of the two is required — the script
will not silently invent a size. `--trainer` must name the trainer class the checkpoint was
produced with, since that determines the architecture the weights are loaded into.

The output `.npz` holds `embeddings` (N × D), `names` (N) and `labels` (N).

**Where labels come from.** Normally the `pretrain_data.json` already groups images by
source dataset and that key *is* the label. If the JSON lumps everything under a single key
(e.g. it was built from one pooled folder), labels are instead parsed from the filename and
matched against `dataset_id`/`aliases` in the metadata table. This is automatic; force it
either way with `--infer_labels_from_filename` / `--no_infer_labels_from_filename`.

## 2. Run the benchmark

```bash
python classification_benchmark.py \
    --embeddings pretrained.npz random_init.npz \
    --model_names "UltraVEM (pretrained)" "UltraVEM (random init)" \
    --metadata /path/to/dataset_metadata.csv \
    --label_levels coarse organism modality \
    --data_efficiency \
    --output results.json
```

Reports, per label level:

- **k-NN** at several *k* (cosine metric) — no training at all, measures whether the
  embedding space is locally organized by category.
- **Linear probe** (multinomial logistic regression on standardized features) — measures
  whether categories are linearly separable.
- **Data-efficiency curves** — both classifiers at 1%, 5%, 10%, 30%, 60%, 100% of the
  training labels, repeated for error bars.

Embeddings are L2-normalized by default (`--no_normalize` to disable). This matters when
comparing against a random-init baseline: untrained encoders produce dataset-specific
feature *magnitudes* that a classifier can exploit as a shortcut, inflating the baseline.

Figures can be regenerated from a saved JSON without recomputing:

```bash
python classification_benchmark.py --from_json results.json --output figures
```

### Choosing the split — this changes what the numbers mean

By default the split is stratified over individual volumes. Because many volumes are crops
from the **same** source dataset, train and test then contain highly correlated neighbours,
and the score partly reflects "can it recognize a volume it has already seen a neighbour
of". That is the standard linear-probe protocol, but it is optimistic as a measure of
generalization.

`--group_split` holds out **whole source datasets**, so no dataset contributes to both
sides. This answers the stricter question: does the representation transfer to a dataset
never seen during evaluation? Expect substantially lower numbers. It requires each class to
be covered by at least two source datasets; if a class ends up with no training data, its
test samples are dropped with a warning rather than crashing.

The protocol used is recorded in the results JSON as `split`, together with the held-out
datasets in `test_source_datasets`.

### Tunable parameters

Every hyperparameter has a documented default and a flag:

| Flag | Default | Meaning |
|---|---|---|
| `--knn_k` | 5 | Primary *k* used in summaries and data-efficiency curves |
| `--knn_k_values` | `1 3 5 10 20` | Full *k* sweep (the primary *k* is always included) |
| `--de_fractions` | `0.01 0.05 0.1 0.3 0.6 1.0` | Data-efficiency training fractions |
| `--de_repeats` | 3 | Subsampling repeats per fraction (1 run at 100%) |
| `--linear_max_iter` | 2000 | L-BFGS iterations for the linear probe |
| `--linear_C` | 1.0 | Inverse L2 regularization strength |
| `--test_fraction` | 0.2 | Test split size |
| `--min_samples` | 5 | Minimum samples per class to keep the class |
| `--seed` | 42 | Seed for the split and all subsampling |

---

## Requirements

```
numpy  scikit-learn  matplotlib  torch  tqdm  pandas
```

`torch` and `nnssl` are needed only for `extract_embeddings.py`; the benchmark runs on the
`.npz` files alone. `pandas` is needed only to read `.xlsx` metadata.

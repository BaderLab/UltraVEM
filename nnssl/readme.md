# UltraVEM — Self-supervised pre-training for 3D volume electron microscopy

This repository contains the **self-supervised pre-training** stage of the UltraVEM 3D
foundation model for volume electron microscopy (vEM).

It is a fork of [**nnssl**](https://github.com/MIC-DKFZ/nnssl) (MIC-DKFZ / Helmholtz
Imaging), the framework released with *An OpenMind for 3D medical vision self-supervised
learning*. nnssl targets 3D **medical** imaging (MRI/CT); this fork adapts it to
**isotropic volume EM** and adds the multi-node training path used to pre-train UltraVEM on
16 GPUs. See [`CHANGES_VS_UPSTREAM.md`](CHANGES_VS_UPSTREAM.md) for a complete,
file-by-file list of what was changed and why.

> **Citation.** If you use this code, please cite both the UltraVEM paper and the original
> nnssl paper — see [Citation](#citation).

---

## What is in this repository

| Stage | Entry point | Description |
|---|---|---|
| 1. Dataset description | `nnssl_convert_ultravem` | Scans the curated vEM volumes and writes the `pretrain_data.json` that nnssl consumes. |
| 2. Fingerprint + planning | `nnssl_extract_fingerprint`, `nnssl_plan_experiment` | Derives spacing/intensity statistics and the training plan. |
| 3. Preprocessing | `nnssl_plan_and_preprocess` | Resamples and normalizes volumes into Blosc2 chunked arrays. Shardable across a job array. |
| 4. Pre-training | `nnssl_train_multinode` | Masked-autoencoder pre-training with multi-node DDP. |

Downstream use lives in separate repositories: embedding extraction and the
classification benchmark are in the UltraVEM `classification` repository, and
segmentation fine-tuning in its own. This one covers pre-training only.

---

## The pre-trained model

The released UltraVEM encoder was pre-trained with the following configuration.

**Architecture** — `ResidualEncoderUNet` ("ResEnc-L"), 6 stages

| | |
|---|---|
| Features per stage | `[32, 64, 128, 256, 320, 320]` |
| Blocks per stage | `[1, 3, 4, 6, 6, 6]` |
| Strides | `[1,1,1]` then `[2,2,2]` ×5 |
| Kernels | `3×3×3` throughout |
| Normalization | `InstanceNorm3d` (affine) |
| Non-linearity | `LeakyReLU` |
| Input channels | 1 (grayscale EM) |

**Pre-training objective** — masked autoencoding (MAE), 75% mask ratio, reconstruction loss
on masked voxels.

| Setting | Value |
|---|---|
| Trainer | `BaseMAETrainer_BS32_ep10000_Arch_Width_B_Depth_B` |
| Configuration | `onemmiso` (isotropic resampling) |
| Plans | `nnsslPlans` |
| Patch size | `224 × 192 × 224` |
| Total batch size | 32 (2 per GPU × 16 GPUs) |
| Epochs | 10,000 × 250 iterations = 2.5 M steps |
| Optimizer | SGD, Nesterov momentum 0.99 |
| Initial LR | `1e-2`, polynomial decay (exponent 0.9) |
| Weight decay | `1e-6` |
| Precision | AMP (fp16 autocast + `GradScaler`) |

**Compute** — 4 nodes × 4 NVIDIA H100 = **16 GPUs**, DDP over NCCL/InfiniBand. One process
per GPU; rank, world size and local rank are derived from `SLURM_PROCID`, `SLURM_NTASKS`
and `SLURM_LOCALID`.

> The patch size `224×192×224` is deliberately anisotropic in the second axis: it is the
> largest volume that fits in 80 GB of H100 memory at this depth/width with a batch of 2.

---

## Installation

Python ≥ 3.10 (3.12 used for the reported runs) and a CUDA-capable PyTorch build.

```bash
git clone <this-repo> && cd nnssl
pip install -e .
pip install -e ./dynamic-network-architectures   # vendored, see note below
```

`dynamic-network-architectures/` is vendored rather than resolved from a git URL: upstream
pins it to a moving `@main` reference, so the in-tree copy fixes the exact revision
(v0.4.2, commit `6d9f47e`) the released model was trained against. Provenance and license
are recorded in [`CHANGES_VS_UPSTREAM.md`](CHANGES_VS_UPSTREAM.md).

More detail in [`documentation/installation_instructions.md`](documentation/installation_instructions.md).

### Environment variables

nnssl locates data through three variables:

| Variable | Contents |
|---|---|
| `nnssl_raw` | Dataset folders holding `pretrain_data.json` |
| `nnssl_preprocessed` | Preprocessed Blosc2 arrays and plans |
| `nnssl_results` | Checkpoints and training logs |

[`scripts/env.sh`](scripts/env.sh) sets all three plus the experiment defaults; edit it once
and every script below picks it up. Details in
[`documentation/setting_up_paths.md`](documentation/setting_up_paths.md).

---

## Reproducing the pre-training

All SLURM scripts are submitted **from the repository root** and read their settings from
`scripts/env.sh`. They were written for a SLURM cluster with node-local `$SLURM_TMPDIR`
scratch and a local wheelhouse (compute nodes without outbound network access); adjust the
`module load` block and `WHEELHOUSE` for your site.

### 1. Describe the corpus

```bash
nnssl_convert_ultravem \
    --parent_dir /path/to/curated_vem_volumes \
    --output_root "$nnssl_raw" \
    --dataset-id 6 --collection-name UltraVEM \
    --layout flat
```

Use `--layout nested` when the volumes are organized as one sub-directory per source
dataset (`<parent>/<dataset>/stage3_curated/*.nii.gz`), which preserves per-dataset
provenance in the collection. This writes
`$nnssl_raw/Dataset006_UltraVEM/pretrain_data.json`; no image data is copied, the JSON
references the volumes in place.

### 2. Fingerprint, plan and preprocess

The fingerprint must see the whole dataset, so it runs once:

```bash
sbatch --account=<your-account> scripts/preprocess_array.slurm fingerprint
```

Preprocessing is then fanned out over a 20-way job array (shards are disjoint and
order-independent, so failed shards can be re-run individually):

```bash
sbatch --account=<your-account> scripts/preprocess_array.slurm
```

### 3. Pre-train

```bash
sbatch --account=<your-account> scripts/pretrain_multinode.slurm
```

The script passes `--c` by default so
that a run spanning multiple time-limited jobs resumes from the latest checkpoint; set
`CONTINUE=0` to force a fresh start. Any setting can be overridden per submission:

```bash
DATASET_ID=<id> TRAINER=<trainer-class> \
    sbatch --account=<your-account> scripts/pretrain_multinode.slurm
```

The scale (`--nodes`, `--gpus-per-node`, `--ntasks-per-node`, `--time`) is set by the
`#SBATCH` directives inside the script; request one task per GPU.

Checkpoints land in
`$nnssl_results/Dataset006_UltraVEM/<TRAINER>__<PLANS>__onemmiso/fold_all/`.

---

## Available trainers

The UltraVEM MAE trainers are defined in
[`src/nnssl/training/nnsslTrainer/masked_image_modeling/base_mae_derivates/var_arch_size.py`](src/nnssl/training/nnsslTrainer/masked_image_modeling/base_mae_derivates/var_arch_size.py):

| Trainer | Batch | Epochs | Used for |
|---|---|---|---|
| `BaseMAETrainer_BS32_ep10000_Arch_Width_B_Depth_B` | 32 | 10,000 | **Released UltraVEM model** |
| `BaseMAETrainer_BS8_ep5000_Arch_Width_B_Depth_B` | 16 | 5,000 | Scaling comparison |
| `BaseMAETrainer_BS8_ep1000_Arch_Width_B_Depth_B` | 8 | 4,000 | Scaling comparison |

All upstream nnssl pre-training methods (VoCo, VolumeFusion, Models Genesis, SimMIM,
SPARK-3D, SwinUNETR, SimCLR) and the Primus transformer architecture are retained
unmodified, so the baselines from the OpenMind study can still be reproduced from this
fork. They were not used for UltraVEM.

---

## Repository layout

```
scripts/                    SLURM scripts + shared environment  (added)
src/nnssl/
  dataset_conversion/       Dataset006_UltraVEM.py               (added)
  run/run_training_multinode.py   Multi-node DDP entry point    (added)
  training/nnsslTrainer/    Trainers; MAE derivates modified    (modified)
  experiment_planning/      Fingerprinting, planning, sharding  (modified)
  preprocessing/, data/     Blosc2 preprocessing pipeline       (modified)
dynamic-network-architectures/   Vendored architecture library
documentation/              Upstream nnssl documentation
tests/                      Upstream tests
```

---

## License

This fork inherits the license of the upstream nnssl repository:
**Creative Commons Attribution-NonCommercial 4.0 International** (see [`LICENSE.md`](LICENSE.md)).
Note the **non-commercial** restriction — it applies to this fork as well.

Copyright for the original framework: German Cancer Research Center (DKFZ) and contributors.

## Citation

Please cite the framework this work builds on:

```bibtex
@article{wald2025openmind,
  title   = {An OpenMind for 3D medical vision self-supervised learning},
  author  = {Wald, Tassilo and others},
  journal = {arXiv preprint arXiv:2412.17041},
  year    = {2025}
}
```

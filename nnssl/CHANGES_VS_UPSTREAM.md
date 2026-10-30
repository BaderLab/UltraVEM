# Changes relative to upstream nnssl

This repository is a fork of [MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl), the
framework released with *An OpenMind for 3D medical vision self-supervised learning*.
Upstream targets 3D medical imaging (MRI/CT); this fork adapts it to isotropic volume
electron microscopy and adds the multi-node training path used for UltraVEM.

Everything not listed below is upstream code, unmodified. All upstream pre-training
methods and architectures are retained so that the OpenMind baselines remain reproducible
from this fork.

---

## Added files

| File | Purpose |
|---|---|
| `src/nnssl/run/run_training_multinode.py` | Multi-node DDP entry point. Derives rank / world size / local rank from `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID` and rendezvouses via `MASTER_ADDR`/`MASTER_PORT`. Upstream's `run_training.py` supports single-node multi-GPU only (`mp.spawn`), which caps training at 4 GPUs on our cluster. |
| `src/nnssl/dataset_conversion/Dataset006_UltraVEM.py` | Builds `pretrain_data.json` for the vEM corpus. Supports a flat directory of volumes and a nested one-directory-per-source-dataset layout. References volumes in place; copies nothing. |
| `src/nnssl/training/nnsslTrainer/AbstractTrainer_noddp.py` | Single-process variant of `AbstractTrainer`, used for debugging and profiling without the DDP wrapper. |
| `scripts/env.sh` | Shared data paths and experiment defaults for all SLURM scripts. |
| `scripts/setup_node_env.sh` | Builds the Python environment in node-local `$SLURM_TMPDIR`. Run once per node before the training tasks. |
| `scripts/pretrain_multinode.slurm` | The 4-node × 4-GPU pre-training job. |
| `scripts/preprocess_array.slurm` | Fingerprinting, planning and sharded preprocessing as a SLURM job array. |

## Modified files

### Multi-node distributed training

**`training/nnsslTrainer/AbstractTrainer.py`** — *the substantive change enabling 16-GPU training*

Upstream conflates local and global rank: `self.local_rank = dist.get_rank()`, and that value
is used both to select the CUDA device (`torch.device("cuda", index=self.local_rank)`) and
to gate rank-0-only work such as checkpoint saving and logging. This is correct on a single
node, where global rank and device index coincide, but wrong across nodes — on a 4-node ×
4-GPU job, global rank 7 would try to select `cuda:7` on a node that only has devices 0–3.

The two concepts are now separated:

- `self.local_rank` — device index within the node, taken from `SLURM_LOCALID` when
  present, otherwise `dist.get_rank() % torch.cuda.device_count()`. Used only to select
  the CUDA device.
- `self.global_rank` — rank across the whole job, from `dist.get_rank()`. Used for all
  rank-0-only guards: `print_to_log_file`, `print_plans`, `save_checkpoint`,
  `_save_debug_information`, and checkpoint cleanup.

Without this split, a multi-node run either crashes on device selection or has one process
per node writing checkpoints concurrently to the same path.

Also in this file:
- `total_batch_size` default `2` → `8`.
- Validation split: `n_val_subjects` is now clamped to at least 1
  (`min(200, len/100)` yields 0 for corpora under 100 subjects, producing an empty
  validation set and a division-by-zero later).

### Volume EM adaptation

**`training/nnsslTrainer/masked_image_modeling/BaseMAETrainer.py`**
- Default patch size `(160,160,160)` → `(224,192,224)`: the largest volume fitting in 80 GB
  of H100 memory at this depth/width with a per-GPU batch of 2. EM structures of interest
  (organelles, membranes) benefit from the larger receptive field.
- Added `BaseMAETrainer_BS8_5000ep` (batch 16, 5,000 epochs) and
  `BaseMAETrainer_BS32_10000ep` (batch 32, 10,000 epochs) for the longer schedules the
  larger vEM corpus supports.
- `BaseMAETrainer_BS8_1000ep`: epochs 1,000 → 4,000; weight decay set to `1e-6`.

**`training/nnsslTrainer/masked_image_modeling/base_mae_derivates/var_arch_size.py`**
- Added `BaseMAETrainer_BS8_ep5000_Arch_Width_B_Depth_B` and
  `BaseMAETrainer_BS32_ep10000_Arch_Width_B_Depth_B`. The latter is the trainer used for the
  released UltraVEM model.
- All three UltraVEM trainers pin the configuration patch size to `224×192×224` before
  building the architecture.

**`training/nnsslTrainer/masked_image_modeling/BaseEvaMAETrainer.py`**
- Patch size `(160,160,160)` → `(224,192,224)`.
- LR `3e-4` → `1e-2`, weight decay `5e-2` → `3e-5`, deep supervision enabled — aligning the
  Eva-MAE variant with the CNN trainer settings that worked on vEM. *(Not used for the
  released model.)*

**`training/nnsslTrainer/masked_image_modeling/base_mae_derivates/var_patch_sizes.py`**
- `BaseMAETrainer_BS8_ep1000_patch_224`: `[224,224,224]` → `[224,192,224]` for consistency
  with the trainers above.

### Preprocessing correctness

**`data/dataloading/dataset.py`** — *bug fix*
Anonymization and anatomy masks are stored without a channel dimension, but were being
written to Blosc2 with the 4D image chunk/block shapes (`chunks_seg`, `blocks_seg`). The
masks are now squeezed and written with `chunks_seg[1:]` / `blocks_seg[1:]`, matching their
actual 3D rank. Upstream is unaffected in practice because its MRI datasets always carry
both masks with a leading singleton axis.

**`experiment_planning/plan_and_preprocess_api.py`** — *bug fix*
`num_processes = list(num_processes)` raises `TypeError` when `-np` is a single `int`
(argparse's default). Corrected to `[num_processes]`.

### Scaling to a large corpus

**`experiment_planning/plan_and_preprocess_entrypoints.py`**
- Added `--part` / `--total_parts` so preprocessing can be split across a SLURM job array;
  shards are disjoint and order-independent. Preprocessing the full corpus on a single node
  is prohibitively slow.
- Added `--no_fp` to skip fingerprint extraction. The fingerprint must see the whole dataset
  and is computed once; without this flag every array task would redundantly recompute it.
  Argument validation rejects `--part` outside `[0, --total_parts)`.

**`ssl_data/dataloading/base_data_loader.py`**
- `determine_shapes()` now raises a clear error when the index set is empty, instead of
  failing later with an opaque `IndexError`.

### Miscellaneous

**`pyproject.toml`**
- Fixed the broken `nnssl_convert_openmind` console script: it pointed at
  `nnssl.data.dataset_conversion.Dataset001_Openmind`, but the module is
  `nnssl.dataset_conversion.Dataset001_OpenMind` (wrong package path and capitalization).
- Pinned `dynamic-network-architectures` to an exact commit instead of a moving `@main`.
- Registered `nnssl_convert_ultravem` and `nnssl_train_multinode`.

**`dataset_conversion/Dataset001_OpenMind.py`** — *upstream bug fix*
First line read `mport argparse`; the module could not be imported at all. Corrected to
`import argparse`. (Unrelated to UltraVEM, but the `nnssl_convert_openmind` entry point
above targets this module.)

**`.gitignore`**
- The blanket `*.txt` rule left `requirements.txt` untracked, so a fresh clone was missing
  the file the environment setup installs from. Added negations for it, plus rules for
  SLURM job output and local `env.sh` overrides.

**`training/lr_scheduler/polylr.py`**, **`data/utils.py`**,
**`preprocessing/preprocessors/default_preprocessor.py`**, **`run/run_training.py`**
- Whitespace and debug-statement cleanup only; no behavioural change.

---

## Vendored dependency

`dynamic-network-architectures/` is a vendored copy of the architecture library nnssl
depends on, pinned to the exact revision the UltraVEM pre-training was run against:

| | |
|---|---|
| Source | [TaWald/dynamic-network-architectures](https://github.com/TaWald/dynamic-network-architectures) |
| Version | 0.4.2 |
| Commit | `6d9f47ebc53dfbc679c6c68a93f4f92d34cb3766` (2025-06-24, *"Remove typo"*) |
| License | Apache 2.0 (see `dynamic-network-architectures/LICENCE`) |

It is included in-tree rather than resolved from PyPI because upstream `pyproject.toml`
pins it to a moving `@main` git reference, and because the compute nodes used for training
have no outbound network access. Vendoring the exact revision makes the released tree
self-contained and reproducible.

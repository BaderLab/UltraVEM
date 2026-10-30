# UltraVEM — Segmentation

Semantic segmentation for 3D volume electron microscopy (vEM), using the standard
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) `ResEncUNet-L` pipeline.

Unlike the denoising and super-resolution tasks, segmentation needs **no custom
trainer or inference script** — it is fully-supervised nnU-Net, unmodified, optionally
warm-started from a UltraVEM self-supervised encoder checkpoint. Everything here is
generic to any labelled dataset; nothing is specific to any one dataset or organism.

| Component | Location |
|---|---|
| Trainer | any class in `nnunetv2/training/nnUNetTrainer/` (e.g. the epoch-count variants in `variants/training_length/nnUNetTrainer_Xepochs.py`, or plain `nnUNetTrainer`) |
| Pretrained-weight loading | `nnunetv2/run/load_pretrained_weights.py` (adapts SSL checkpoint keys/shapes to the segmentation network) |
| Inference | `nnunetv2/inference/predict_from_raw_data.py` (nnU-Net's own prediction entry point) |

This repository holds the SLURM entry points, the shared environment, and a demo
notebook.

---

## Data layout

Standard nnU-Net raw format, with paired image/label volumes:

```
$nnUNet_raw/<Dataset>/
    imagesTr/<case>_0000.nii.gz    raw EM volumes (training)
    labelsTr/<case>.nii.gz         integer class-label volumes (same class IDs as dataset.json)
    imagesTs/<case>_0000.nii.gz    volumes to segment at inference time
    dataset.json                   channel_names, labels, numTraining, file_ending
```

Preprocess as usual:

```bash
nnUNetv2_plan_and_preprocess -d <DATASET> -c <CONFIGURATION> \
    -pl <PLANNER> --verify_dataset_integrity
```

---

## Choosing a trainer

Any nnU-Net trainer class name works with `-tr`/`TRAINER`. Two common choices:

- An epoch-count variant, e.g. `nnUNetTrainer_<N>epochs` — see
  `nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py`
  for the full list of `<N>` values already defined, or add your own by subclassing and
  setting `self.num_epochs`.
- Plain `nnUNetTrainer` for nnU-Net's own default schedule.

There is nothing UltraVEM-specific to configure here — the reader picks the trainer and
plans identifier that fit their dataset size and compute budget, exactly as in stock
nnU-Net.

---

## Usage

Edit [`scripts/env.sh`](scripts/env.sh) once for your cluster and experiment, then
submit from the repository root.

### Train

```bash
DATASET=<id> sbatch --account=<your-account> scripts/train_segment.slurm
```

Override any setting per submission:

```bash
DATASET=<id> TRAINER=<trainer-class> PLANS=<plans-identifier> FOLD=<fold> \
    sbatch --account=<your-account> scripts/train_segment.slurm
```

`FOLD` is an integer `0`-`4` for cross-validation, or `all` to train once on the full
training set (the default here, since a single final model is the usual target for
downstream use).

To warm-start the encoder from pre-trained weights — for example the UltraVEM MAE
checkpoint from the pre-training repository — set `PRETRAINED_WEIGHTS` to the
checkpoint path. Leave it unset to train from scratch. Loading is handled by
`load_pretrained_weights.py`, which matches SSL checkpoint keys to the segmentation
network and adapts kernel shapes where needed (e.g. isotropic 3D kernels warm-starting
an anisotropic configuration); it skips the segmentation head, which always starts
randomly initialized.

### Predict

```bash
INPUT_FOLDER=<dir> OUTPUT_FOLDER=<dir> MODEL_FOLDER=<dir> \
    sbatch --account=<your-account> scripts/predict_segment.slurm
```

`MODEL_FOLDER` is the trainer output directory:
`$nnUNet_results/<Dataset>/<trainer>__<plans>__<configuration>`. Uses nnU-Net's own
`nnUNetv2_predict_from_modelfolder` entry point — no custom inference script needed.
Output is one `<case>.nii.gz` per input file, each voxel containing the predicted class
index.

Both scripts can also be run directly without SLURM using the corresponding
`nnUNetv2_train` / `nnUNetv2_predict_from_modelfolder` console commands (installed with
the nnU-Net repository); see `nnUNetv2_predict_from_modelfolder --help`.

### Demo

[`ultravem_Seg_demo.ipynb`](ultravem_Seg_demo.ipynb) walks through checking a pre-trained
encoder against the segmentation network, preparing labelled data, training, running
inference, and visualizing a segmentation overlaid on the raw volume.

---

## Citation

Built on nnU-Net; please cite:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021}
}
```

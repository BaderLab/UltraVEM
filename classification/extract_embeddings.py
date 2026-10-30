"""
nnSSL MAE embedding extractor
=============================

Extracts one embedding vector per volume from a pre-trained nnSSL MAE encoder,
for downstream classification or clustering.

Architecture context:
  - nnSSL uses a U-Net (e.g. ResEnc-L) for MAE pre-training.
  - `self.network` is the FULL encoder-decoder.
  - For embeddings only the ENCODER is needed.
  - The encoder produces multi-scale feature maps; the deepest (bottleneck)
    is the most semantically rich.
  - The bottleneck features are global-average-pooled into one vector per volume.

Usage:
  python extract_embeddings.py \
      --dataset_id <DATASET_ID> \
      --config <CONFIGURATION> \
      --trainer <TRAINER_CLASS> \
      --plans <PLANS_IDENTIFIER> \
      --checkpoint /path/to/checkpoint.pth \
      --use_patch_size \
      --output_file embeddings.npz \
      --metadata /path/to/dataset_metadata.csv

  Run with --help for the full list of options and their defaults.

  The output .npz file contains:
    - "embeddings": (N, D) float32 array
    - "names":      (N,) volume identifiers
    - "labels":     (N,) source-dataset label per volume
"""

import argparse
import csv
import json
import os
import re
import sys
from os.path import join
from pathlib import Path
from typing import Union, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# nnSSL imports
#
# nnssl must be importable, i.e. installed from the pre-training repository:
#     pip install -e /path/to/nnssl
# ──────────────────────────────────────────────────────────────────────
try:
    import nnssl
    from nnssl.utilities.find_class_by_name import recursive_find_python_class
    from nnssl.paths import nnssl_preprocessed
    from nnssl.experiment_planning.experiment_planners.plan import Plan
    from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json
except ImportError as e:
    print(f"[ERROR] Cannot import nnssl: {e}")
    print("Install it from the pre-training repository: pip install -e /path/to/nnssl")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# 1. Reuse nnSSL's get_trainer_from_args to build the trainer+network
# ──────────────────────────────────────────────────────────────────────

def get_trainer_from_args(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int = 0,
    trainer_name: str = None,
    plans_identifier: str = "nnsslPlans",
    device: torch.device = torch.device("cuda"),
) -> AbstractBaseTrainer:
    """Instantiate the nnSSL trainer (same logic as run_training.py)."""
    nnssl_trainer_cls: Type[AbstractBaseTrainer] = recursive_find_python_class(
        join(nnssl.__path__[0], "training", "nnsslTrainer"),
        trainer_name,
        "nnssl.training.nnsslTrainer",
    )
    if nnssl_trainer_cls is None:
        raise RuntimeError(
            f"Could not find trainer {trainer_name} in nnssl.training.nnsslTrainer "
            f"({join(nnssl.__path__[0], 'training', 'nnsslTrainer')})"
        )

    if isinstance(dataset_name_or_id, str) and not dataset_name_or_id.startswith("Dataset"):
        dataset_name_or_id = int(dataset_name_or_id)

    preprocessed_folder = join(nnssl_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_folder, plans_identifier + ".json")
    plans: Plan = Plan.load_from_file(plans_file)
    pretrain_json = load_json(join(preprocessed_folder, f"pretrain_data__{configuration}.json"))

    trainer: AbstractBaseTrainer = nnssl_trainer_cls(
        plan=plans,
        configuration_name=configuration,
        fold=fold,
        pretrain_json=pretrain_json,
        device=device,
    )
    return trainer


# ──────────────────────────────────────────────────────────────────────
# 2. Encoder feature extraction wrapper
# ──────────────────────────────────────────────────────────────────────

class EncoderFeatureExtractor(nn.Module):
    """
    Wraps the full nnSSL MAE network and extracts ENCODER features only.

    nnSSL uses a U-Net-like architecture (ResEnc-L or Primus-M).
    The network typically has:
      - network.encoder  (the encoder part)
      - network.decoder  (the MAE decoder part)

    The encoder produces a list of multi-scale feature maps:
      [stage0_features, stage1_features, ..., bottleneck_features]

    We extract the bottleneck (deepest stage) and global-average-pool it.

    If the architecture is different, we hook into the encoder's forward pass.
    """

    def __init__(self, network: nn.Module, pool_stage: str = "bottleneck"):
        """
        Args:
            network: the full nnSSL network (encoder + decoder)
            pool_stage: which encoder stage to use
                - "bottleneck": deepest encoder features (recommended, most semantic)
                - "all_concat": concatenate GAP of ALL encoder stages
                - "all_stages": return all stages separately (for analysis)
        """
        super().__init__()
        self.network = network
        self.pool_stage = pool_stage
        self._encoder = None
        self._hook_features = {}
        # One-shot flags so per-batch warnings do not flood the log.
        self._warned_fallback = False
        self._warned_forward_error = False
        self._warned_bottleneck = False

        # Try to find the encoder sub-module
        self._detect_encoder()

    def _detect_encoder(self):
        """Detect how to access the encoder in the network."""
        # Strategy 1: network has an explicit .encoder attribute
        if hasattr(self.network, 'encoder'):
            self._encoder = self.network.encoder
            print(f"[INFO] Found encoder as network.encoder: {type(self._encoder).__name__}")
            return

        # Strategy 2: For nnU-Net-style architectures, the encoder might be
        # accessed differently. Check common patterns.
        for attr_name in ['conv_blocks_context', 'stages', 'down_blocks', 'encoder_stages']:
            if hasattr(self.network, attr_name):
                self._encoder = getattr(self.network, attr_name)
                print(f"[INFO] Found encoder-like module as network.{attr_name}")
                return

        # Strategy 3: Use hooks to capture intermediate features
        print("[WARN] Could not find explicit encoder. Will use forward hooks.")
        print(f"[INFO] Network type: {type(self.network).__name__}")
        print(f"[INFO] Network sub-modules: {[n for n, _ in self.network.named_children()]}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns pooled encoder features.

        Args:
            x: (B, C, D, H, W) input 3D volume

        Returns:
            features: (B, D_embed) pooled feature vector
        """
        # ─── Try direct encoder forward ───────────────────────────
        if self._encoder is not None:
            try:
                # Most nnU-Net / nnSSL encoders return a list of multi-scale features
                encoder_outputs = self._encoder(x)

                if isinstance(encoder_outputs, (list, tuple)):
                    return self._pool_encoder_outputs(encoder_outputs)
                elif isinstance(encoder_outputs, torch.Tensor):
                    # Single output — just pool it
                    return self._global_avg_pool(encoder_outputs)
                else:
                    raise TypeError(f"Unexpected encoder output type: {type(encoder_outputs)}")
            except torch.cuda.OutOfMemoryError:
                # Never mask an OOM as an architecture problem: the hook fallback
                # would OOM too, and silently reducing quality here would corrupt
                # every embedding that follows.
                raise
            except Exception as e:
                # Fall back only once, and say so loudly -- if this fires for every
                # batch the embeddings are not coming from where you think.
                if not self._warned_fallback:
                    self._warned_fallback = True
                    print(f"[WARN] Direct encoder forward failed: {type(e).__name__}: {e}")
                    print("[WARN] Falling back to hook-based extraction for ALL batches. "
                          "Verify the reported bottleneck layer below is the one you expect.")

        # ─── Fallback: hook-based extraction ──────────────────────
        return self._extract_via_hooks(x)

    def _extract_via_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Register forward hooks on all Conv/BatchNorm layers, run forward,
        and grab the deepest feature map.
        """
        features_by_depth = []
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 5:  # (B,C,D,H,W)
                    features_by_depth.append((name, output))
            return hook_fn

        # Register hooks on all modules
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Conv3d, nn.BatchNorm3d, nn.InstanceNorm3d)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run the full forward pass; we only want the intercepted activations.
        # An MAE network may expect masked input and raise partway through --
        # that is tolerated, because the encoder half has already run by then and
        # its activations are captured. But the failure is recorded (once) rather
        # than swallowed, since "no encoder ran at all" and "decoder tripped at
        # the end" look identical otherwise.
        forward_error = None
        try:
            _ = self.network(x)
        except torch.cuda.OutOfMemoryError:
            for h in hooks:
                h.remove()
            raise
        except Exception as e:
            forward_error = e

        # Remove hooks
        for h in hooks:
            h.remove()

        if not features_by_depth:
            raise RuntimeError(
                "No features captured via hooks -- the forward pass produced no 5D "
                "activations, so no embedding can be formed. "
                + (f"Forward raised {type(forward_error).__name__}: {forward_error}"
                   if forward_error is not None else "Check the network architecture.")
            )

        if forward_error is not None and not self._warned_forward_error:
            self._warned_forward_error = True
            print(f"[WARN] Forward pass raised {type(forward_error).__name__}: {forward_error}")
            print("[WARN] Continuing with the activations captured before the failure. "
                  "Confirm the bottleneck layer below sits in the ENCODER.")

        # Deepest = smallest spatial extent. Ties (several layers at bottleneck
        # resolution) resolve to the first registered, i.e. earliest in the network.
        features_by_depth.sort(key=lambda item: item[1].shape[2] * item[1].shape[3] * item[1].shape[4])
        bottleneck_name, bottleneck_feat = features_by_depth[0]
        if not self._warned_bottleneck:
            self._warned_bottleneck = True
            print(f"[INFO] Bottleneck feature from '{bottleneck_name}': shape={bottleneck_feat.shape}")

        return self._global_avg_pool(bottleneck_feat)

    def _pool_encoder_outputs(self, encoder_outputs: list) -> torch.Tensor:
        """Pool multi-scale encoder outputs based on strategy."""
        if self.pool_stage == "bottleneck":
            # Last element = deepest features (bottleneck)
            bottleneck = encoder_outputs[-1]
            return self._global_avg_pool(bottleneck)

        elif self.pool_stage == "all_concat":
            # GAP each stage, concatenate into one long vector
            pooled = [self._global_avg_pool(feat) for feat in encoder_outputs]
            return torch.cat(pooled, dim=1)

        else:
            raise ValueError(f"Unknown pool_stage: {self.pool_stage}")

    @staticmethod
    def _global_avg_pool(feat: torch.Tensor) -> torch.Tensor:
        """
        Global average pooling for 3D feature maps.
        (B, C, D, H, W) → (B, C)
        """
        if feat.ndim == 5:
            return F.adaptive_avg_pool3d(feat, 1).flatten(1)
        elif feat.ndim == 4:
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim == 3:
            return F.adaptive_avg_pool1d(feat, 1).flatten(1)
        else:
            return feat.flatten(1)


# ──────────────────────────────────────────────────────────────────────
# 3. Dataset: iterate over ALL preprocessed images (not patches)
# ──────────────────────────────────────────────────────────────────────

# Known dataset name prefixes — maps filename prefix → dataset label
# Used when all images are lumped under one JSON dataset key.
# ──────────────────────────────────────────────────────────────────────
# Mapping image filenames back to their source dataset
# ──────────────────────────────────────────────────────────────────────
#
# Preprocessing flattens volumes to "<dataset><subject>_<channel>.nii.gz", which
# loses the separators of the original dataset id (e.g. "example_dataset-2" becomes
# the prefix "exampledataset2"). To label an embedding by its source dataset we
# need to map that prefix back.
#
# The list of dataset ids is NOT hard-coded here: it is read from the same
# annotation table used by the benchmark and the plots (metadata/dataset_metadata.csv,
# column `dataset_id`, plus the optional `aliases` column). Point --metadata at
# your own table to use this on a different corpus.
#
# Prefer taking the label straight from pretrain_data.json (the default); this
# filename-based path is only used when the JSON lumps everything under a single
# dataset key (see --infer_labels_from_filename).

DEFAULT_METADATA_CSV = Path(__file__).resolve().parent / "metadata" / "dataset_metadata.csv"


# Populated by load_dataset_ids(); ordered longest-first so that a longer id wins
# over a shorter one that happens to be its prefix (e.g. "exampleds8" vs "exampleds").
KNOWN_DATASET_IDS: list = []
_DATASET_ID_BY_CLEAN_KEY: dict = {}


def _clean_key(name: str) -> str:
    """Strip separators and case so 'example_dataset-2' and 'exampledataset2' match."""
    return re.sub(r"[-_ .,]+", "", str(name).lower().strip())


def load_dataset_ids(metadata_path=None) -> list:
    """Read known dataset ids (and aliases) from the annotation table.

    Returns the ids sorted longest-cleaned-key first. Missing table is not fatal:
    filename inference then falls back to the raw prefix.
    """
    global KNOWN_DATASET_IDS, _DATASET_ID_BY_CLEAN_KEY
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_CSV
    if not path.is_file():
        print(f"[WARN] No dataset table at {path}; filename inference will use raw prefixes.")
        KNOWN_DATASET_IDS, _DATASET_ID_BY_CLEAN_KEY = [], {}
        return KNOWN_DATASET_IDS

    if path.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        rows = pd.read_excel(path).to_dict("records")
    else:
        delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=delim))

    mapping = {}
    for row in rows:
        row = {(k or "").strip().lower(): ("" if v is None else str(v).strip())
               for k, v in row.items()}
        did = row.get("dataset_id", "")
        if not did or did.lower() == "nan":
            continue
        mapping.setdefault(_clean_key(did), did)
        for alias in row.get("aliases", "").split(";"):
            if alias.strip():
                mapping.setdefault(_clean_key(alias), did)

    _DATASET_ID_BY_CLEAN_KEY = mapping
    # Longest first so "exampleds8" is matched before "exampleds".
    KNOWN_DATASET_IDS = sorted(mapping, key=len, reverse=True)
    print(f"[INFO] Loaded {len(KNOWN_DATASET_IDS)} dataset ids/aliases from {path}")
    return KNOWN_DATASET_IDS


def _infer_dataset_from_filename(image_name: str) -> str:
    """
    Infer the source dataset id from an image filename.

        examplebrain_00001_0000     -> example_brain
        exampledataset2_00005_0000  -> example_dataset-2
        exampleds3_00012_0000       -> exampleds3

    Strips the trailing numeric "_<subject>_<channel>" fields, then matches the
    remaining prefix against the ids loaded by load_dataset_ids(). Falls back to
    the cleaned prefix itself when nothing matches, so unknown datasets still get
    a stable (if unmapped) label rather than being silently merged.
    """
    parts = str(image_name).lower().strip().split("_")
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    prefix = "_".join(parts)
    prefix_clean = _clean_key(prefix)

    # Exact match first, then longest prefix match.
    if prefix_clean in _DATASET_ID_BY_CLEAN_KEY:
        return _DATASET_ID_BY_CLEAN_KEY[prefix_clean]
    for known_clean in KNOWN_DATASET_IDS:
        if prefix_clean.startswith(known_clean):
            return _DATASET_ID_BY_CLEAN_KEY[known_clean]
    return prefix



class NnSSLPreprocessedDataset(Dataset):
    """
    Loads ALL preprocessed 3D volumes from the nnSSL preprocessed folder.

    nnSSL preprocesses data into .npy (or blosc2) files in nnssl_preprocessed/.
    The pretrain_data.json lists all image paths.

    For embedding extraction, we load each FULL volume (or center-crop it to
    the plan's patch_size if it's too large for GPU memory).
    """

    def __init__(
        self,
        pretrain_json: dict,
        preprocessed_folder: str,
        patch_size: tuple,
        max_samples: int = None,
        infer_labels_from_filename: bool = None,
        lumped_json_max_keys: int = None,
    ):
        """
        Args:
            pretrain_json: loaded pretrain_data.json dict
            preprocessed_folder: path to nnssl_preprocessed/DatasetXXX_name/
            patch_size: (D, H, W) every volume is center-cropped/padded to; required,
                        because a batch can only be stacked at a common size
            max_samples: limit total number of samples
            infer_labels_from_filename: None = auto, True/False = force
            lumped_json_max_keys: auto-mode threshold (see _parse_pretrain_json).
                Required when infer_labels_from_filename is left as auto (None);
                unused, and may be omitted, when it is forced True/False.
        """
        self.preprocessed_folder = Path(preprocessed_folder)
        self.patch_size = patch_size
        self.infer_labels_from_filename = infer_labels_from_filename
        self.lumped_json_max_keys = lumped_json_max_keys

        # Collect all image paths and names from pretrain_data.json
        self.image_paths = []
        self.image_names = []
        self.image_labels = []  # dataset_index as a proxy label for clustering

        self._parse_pretrain_json(pretrain_json)

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            self.image_names = self.image_names[:max_samples]
            self.image_labels = self.image_labels[:max_samples]

        print(f"[INFO] Dataset: {len(self.image_paths)} images found")
        unique_labels = sorted(set(self.image_labels))
        print(f"[INFO] Unique labels (dataset sources): {len(unique_labels)}")
        from collections import Counter
        counts = Counter(self.image_labels)
        for label in unique_labels:
            print(f"  {label:<40s} → {counts[label]:4d} images")

    def _parse_pretrain_json(self, pj: dict):
        """
        Walk the pretrain_data.json hierarchy and collect all image paths.

        FAST VERSION: builds a filename→path index once from the filesystem,
        then does O(1) lookups per image instead of rglob per image.
        """
        datasets = pj.get("datasets", {})
        collection_name = pj.get("collection_name", "")

        # Auto-detect the plans subfolder
        plans_subfolder = self._detect_plans_subfolder()

        # Build file index ONCE for fast lookups (the key optimization)
        print("[INFO] Building file index (one-time scan)...")
        file_index = self._build_file_index()
        print(f"[INFO] Indexed {len(file_index)} files")

        # Count total images for progress bar
        total_images = sum(
            len(img)
            for ds_val in datasets.values()
            for subj_val in ds_val.get("subjects", {}).values()
            for sess_val in subj_val.get("sessions", {}).values()
            for img in [sess_val.get("images", [])]
        )
        print(f"[INFO] Processing {total_images} image entries...")

        found_count = 0
        not_found_count = 0
        processed = 0

        # Where do per-image labels come from?
        #
        # Normally the pretrain_data.json already groups images by source dataset,
        # and that key IS the label -- always prefer it. But a JSON built from a
        # single pooled folder lumps everything under one key, which would collapse
        # every image into one label; there we fall back to parsing the filename.
        #
        # `infer_labels_from_filename` is tri-state:
        #   None  -> auto: infer only if the JSON has fewer keys than the threshold
        #   True  -> always infer from filenames
        #   False -> always trust the JSON keys
        n_dataset_keys = len(datasets)
        if self.infer_labels_from_filename is None:
            if self.lumped_json_max_keys is None:
                raise ValueError(
                    "Labeling mode was left automatic but --lumped_json_max_keys was not "
                    "set, so there is no threshold to decide whether this pretrain_data.json "
                    "is 'lumped'. Either pass --lumped_json_max_keys <n>, or force the mode "
                    "explicitly with --infer_labels_from_filename / --no_infer_labels_from_filename."
                )
            is_lumped = n_dataset_keys <= self.lumped_json_max_keys
        else:
            is_lumped = bool(self.infer_labels_from_filename)
        if is_lumped:
            print(f"[INFO] {n_dataset_keys} dataset key(s) in JSON -> "
                  f"inferring per-image labels from filenames.")
        else:
            print(f"[INFO] {n_dataset_keys} dataset key(s) in JSON -> "
                  f"using JSON dataset keys as labels.")

        for ds_key, ds_val in datasets.items():
            dataset_label_json = ds_val.get("dataset_index", ds_key)
            subjects = ds_val.get("subjects", {})

            for subj_key, subj_val in subjects.items():
                sessions = subj_val.get("sessions", {})

                for sess_key, sess_val in sessions.items():
                    images = sess_val.get("images", [])

                    for img_entry in images:
                        img_name = img_entry.get("name", "")

                        # Strategy 1 (FAST): O(1) dict lookup in pre-built index
                        found_path = file_index.get(img_name)

                        if found_path is not None:
                            # Determine label: use JSON dataset_index if there are
                            # many datasets, otherwise infer from filename
                            if is_lumped:
                                label = _infer_dataset_from_filename(img_name)
                            else:
                                label = str(dataset_label_json)

                            self.image_paths.append(found_path)
                            self.image_names.append(img_name)
                            self.image_labels.append(label)
                            found_count += 1
                        else:
                            not_found_count += 1
                            if not_found_count <= 3:
                                print(f"[WARN] Not found: {img_name}")

                        processed += 1
                        if processed % 50000 == 0:
                            print(f"[INFO] Progress: {processed}/{total_images} "
                                  f"({found_count} found, {not_found_count} missing)")

        if not_found_count > 3:
            print(f"[WARN] ... and {not_found_count - 3} more files not found.")
        print(f"[INFO] Found {found_count} / {found_count + not_found_count} images")

    def _build_file_index(self) -> dict:
        """
        Scan the preprocessed folder ONCE and build a dict:
            {image_name_without_ext: full_path}

        This replaces the per-image rglob which was O(N * filesystem_size).
        Now it's O(filesystem_size) once + O(1) per lookup.
        """
        index = {}
        if not self.preprocessed_folder.exists():
            return index

        for ext in [".b2nd", ".npy", ".npz"]:
            for path in self.preprocessed_folder.rglob(f"*{ext}"):
                stem = path.name
                # Strip extension to get the image name
                # .b2nd → just strip .b2nd
                # .nii.gz → strip both
                if stem.endswith(".nii.gz"):
                    name = stem[:-7]
                elif stem.endswith(ext):
                    name = stem[:-len(ext)]
                else:
                    name = path.stem

                # Only store the first match (don't overwrite)
                if name not in index:
                    index[name] = str(path)

        return index

    def _detect_plans_subfolder(self) -> str:
        """
        Auto-detect the plans subfolder inside the preprocessed directory.
        e.g., nnsslPlans_onemmiso, nnsslPlans, etc.
        """
        if not self.preprocessed_folder.exists():
            return ""

        for item in self.preprocessed_folder.iterdir():
            if item.is_dir() and "plan" in item.name.lower():
                print(f"[INFO] Detected plans subfolder: {item.name}")
                return item.name

        # Also check for any subfolder that contains another folder matching
        # the dataset name (the preprocessed structure nests dataset inside plans)
        for item in self.preprocessed_folder.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check if this subfolder contains the dataset structure
                for sub in item.iterdir():
                    if sub.is_dir():
                        print(f"[INFO] Detected plans subfolder: {item.name}")
                        return item.name

        print(f"[WARN] Could not auto-detect plans subfolder in {self.preprocessed_folder}")
        print(f"[INFO] Contents: {[x.name for x in self.preprocessed_folder.iterdir() if x.is_dir()]}")
        return ""

    def __len__(self):
        return len(self.image_paths)

    # Class-level list to track all corrupt files (shared across workers via main process)
    _corrupt_files = []

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        name = self.image_names[idx]
        label = self.image_labels[idx]

        try:
            data = self._load_volume(path)  # (C, D, H, W)
            if self.patch_size is not None:
                data = self._center_crop_or_pad(data, self.patch_size)
        except Exception as e:
            # Log the corrupt file — print ALL of them (not suppressed)
            print(f"[CORRUPT] {path} — {type(e).__name__}: {e}")
            # Return zero tensor with sentinel label so it can be filtered out
            ps = self.patch_size
            data = np.zeros((1, *ps), dtype=np.float32)
            label = "__CORRUPT__"

        return {
            "data": torch.from_numpy(data).float(),
            "name": name,
            "label": label,
        }

    @staticmethod
    def _load_volume(path: str) -> np.ndarray:
        """
        Load a 3D volume. Returns shape (C, D, H, W).
        Supports: .npy, .npz, .nii.gz, .b2nd (blosc2)
        """
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".npz"):
            f = np.load(path)
            data = f[f.files[0]]
        elif path.endswith(".nii.gz") or path.endswith(".nii"):
            import nibabel as nib
            data = nib.load(path).get_fdata()
        elif path.endswith(".b2nd") or path.endswith(".blosc2"):
            # nnSSL uses blosc2 .b2nd for compressed storage
            try:
                import blosc2
                data = blosc2.open(path)[:]
            except ImportError:
                try:
                    # Fallback: try nnSSL's own loader
                    from nnssl.preprocessing.blosc2_io import load_blosc2
                    data = load_blosc2(path)
                except ImportError:
                    try:
                        import blosc2 as b2
                        nd = b2.open(path)
                        data = nd[:]
                    except Exception:
                        raise ImportError(
                            f"Cannot load .b2nd file: {path}\n"
                            "Install blosc2: pip install blosc2"
                        )
        else:
            # Try numpy as fallback
            data = np.load(path)

        # Ensure shape is (C, D, H, W)
        if data.ndim == 3:
            data = data[np.newaxis]  # (D, H, W) → (1, D, H, W)
        elif data.ndim == 4:
            pass  # Already (C, D, H, W)
        elif data.ndim == 5:
            data = data[0]  # (B, C, D, H, W) → take first
        return data

    @staticmethod
    def _center_crop_or_pad(data: np.ndarray, patch_size: tuple) -> np.ndarray:
        """
        Center-crop AND/OR pad (C, D, H, W) to exactly (C, *patch_size).

        - If a dimension is larger than patch_size: center-crop.
        - If a dimension is smaller than patch_size: zero-pad symmetrically.
        - This guarantees the output is ALWAYS exactly patch_size.
        """
        _, d, h, w = data.shape
        pd, ph, pw = patch_size

        # Step 1: Pad dimensions that are too small
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            data = np.pad(
                data,
                ((0, 0),
                 (pad_d // 2, pad_d - pad_d // 2),
                 (pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2)),
                mode='constant', constant_values=0,
            )
            _, d, h, w = data.shape

        # Step 2: Center-crop dimensions that are too large
        sd = (d - pd) // 2
        sh = (h - ph) // 2
        sw = (w - pw) // 2
        data = data[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]

        return data


# ──────────────────────────────────────────────────────────────────────
# 3b. RawDataDataset: scan directories for raw .nii.gz/.b2nd/.npy files
# ──────────────────────────────────────────────────────────────────────

class RawDataDataset(Dataset):
    """
    Loads 3D volumes directly from one or more raw data directories.
    No pretrain_data.json needed — just scans for supported file types.

    Directory structure expected:
        raw_root/
            dataset_A/
                stage3_curated/ (or any subfolder)
                    image_00001.nii.gz
                    image_00002.nii.gz
            dataset_B/
                ...

    Or a single flat directory:
        raw_root/
            image_00001.nii.gz
            image_00002.nii.gz

    Labels are inferred from the parent directory name or the filename.
    """
    SUPPORTED_EXTS = (".nii.gz", ".nii", ".b2nd", ".blosc2", ".npy", ".npz")

    def __init__(
        self,
        root_paths: list,
        patch_size: tuple = None,
        max_samples: int = None,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.image_paths = []
        self.image_names = []
        self.image_labels = []

        print(f"[INFO] Scanning {len(root_paths)} raw data path(s)...")
        for root in root_paths:
            self._scan_directory(root)

        total_found = len(self.image_paths)
        print(f"[INFO] Found {total_found} volumes across {len(root_paths)} path(s)")

        # Show per-label counts
        from collections import Counter
        counts = Counter(self.image_labels)
        unique_labels = sorted(counts.keys())
        print(f"[INFO] {len(unique_labels)} dataset labels detected:")
        for label in unique_labels:
            print(f"  {label:<40s} → {counts[label]:6d} volumes")

        # Sample if max_samples is set
        if max_samples is not None and max_samples < total_found:
            import random
            random.seed(seed)
            indices = sorted(random.sample(range(total_found), max_samples))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.image_names = [self.image_names[i] for i in indices]
            self.image_labels = [self.image_labels[i] for i in indices]
            print(f"[INFO] Sampled {max_samples} from {total_found} volumes (seed={seed})")

            # Recount
            counts = Counter(self.image_labels)
            for label in sorted(counts.keys()):
                print(f"  {label:<40s} → {counts[label]:6d} (sampled)")

    def _scan_directory(self, root: str):
        """Recursively find all supported volumes under root."""
        root = Path(root)
        if not root.exists():
            print(f"[WARN] Path does not exist: {root}")
            return

        # Infer dataset label from the top-level directory name
        # e.g., /path/to/nnssl_pretraindata/example_dataset-3/stage3_curated
        # → label = "example_dataset-3"
        #
        # Strategy: walk up from root to find the dataset name
        # If root contains "stage3_curated" or similar, use parent
        dataset_label = self._infer_label_from_path(root)

        found = 0
        for ext in self.SUPPORTED_EXTS:
            for path in root.rglob(f"*{ext}"):
                # Skip double-counting .nii vs .nii.gz
                if ext == ".nii" and str(path).endswith(".nii.gz"):
                    continue

                stem = path.stem
                if stem.endswith(".nii"):
                    stem = stem[:-4]

                self.image_paths.append(str(path))
                self.image_names.append(stem)
                # Infer label from filename (more granular) or fall back to directory
                label = _infer_dataset_from_filename(stem)
                if label == stem:
                    # Filename inference didn't find a known prefix — use directory
                    label = dataset_label
                self.image_labels.append(label)
                found += 1

        print(f"  {root}: {found} volumes (label='{dataset_label}')")

    @staticmethod
    def _infer_label_from_path(path: Path) -> str:
        """Infer dataset label from directory path."""
        # If path ends in stage3_curated or similar, use parent
        name = path.name.lower()
        if name in ("stage3_curated", "curated", "raw", "data", "images", "volumes"):
            return path.parent.name
        return path.name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        name = self.image_names[idx]
        label = self.image_labels[idx]

        try:
            data = NnSSLPreprocessedDataset._load_volume(path)
            if self.patch_size is not None:
                data = NnSSLPreprocessedDataset._center_crop_or_pad(data, self.patch_size)
        except Exception as e:
            print(f"[CORRUPT] {path} — {type(e).__name__}: {e}")
            ps = self.patch_size
            data = np.zeros((1, *ps), dtype=np.float32)
            label = "__CORRUPT__"

        return {
            "data": torch.from_numpy(data).float(),
            "name": name,
            "label": label,
        }


# ──────────────────────────────────────────────────────────────────────
# 4. Main extraction loop
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_embeddings(
    feature_extractor: EncoderFeatureExtractor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run all images through the encoder and collect embedding vectors.
    Automatically filters out corrupt/failed images (labeled "__CORRUPT__").

    Returns:
        dict with keys: "embeddings", "names", "labels"
    """
    feature_extractor.eval()
    feature_extractor.to(device)

    all_embeddings = []
    all_names = []
    all_labels = []
    n_corrupt = 0
    corrupt_names = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        data = batch["data"].to(device, non_blocking=True)
        names = batch["name"]
        labels = batch["label"]

        # Forward through encoder only → (B, D_embed)
        embeddings = feature_extractor(data)
        emb_np = embeddings.cpu().numpy()

        # Filter out corrupt entries per item in batch
        for i in range(len(names)):
            if labels[i] == "__CORRUPT__":
                n_corrupt += 1
                corrupt_names.append(names[i])
            else:
                all_embeddings.append(emb_np[i:i+1])
                all_names.append(names[i])
                all_labels.append(labels[i])

    embeddings_array = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0))

    print(f"\n[RESULT] Extracted embeddings: shape = {embeddings_array.shape}")
    print(f"[RESULT] Embedding dimension: {embeddings_array.shape[1] if len(embeddings_array) > 0 else 'N/A'}")
    print(f"[RESULT] Valid images: {embeddings_array.shape[0]}")
    if n_corrupt > 0:
        print(f"[RESULT] Corrupt/skipped: {n_corrupt} files:")
        for cn in corrupt_names:
            print(f"  ✗ {cn}")

    return {
        "embeddings": embeddings_array,
        "names": np.array(all_names),
        "labels": np.array(all_labels),
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract embedding vectors from nnSSL MAE pretrained encoder"
    )
    parser.add_argument(
        "--dataset_id", type=str, nargs="+", default=None,
        help="Dataset ID(s) for nnSSL preprocessed data. Accepts multiple. "
             "e.g.: --dataset_id 7 8 9. Not needed if --raw_data_path is used."
    )
    parser.add_argument(
        "--raw_data_path", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Scan raw data directories for .nii.gz/.b2nd/.npy volumes. "
             "Bypasses nnSSL preprocessing entirely. "
             "e.g.: --raw_data_path /path/to/nnssl_pretraindata "
             "or: --raw_data_path /path/to/example_dataset-3/stage3_curated "
             "/path/to/exampleds/stage3_curated"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Configuration name (must match pretrain_data__<config>.json)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained MAE checkpoint .pth file. "
             "Not needed if --random_init is used."
    )
    parser.add_argument(
        "--random_init", action="store_true",
        help="Use randomly initialized weights (no checkpoint). "
             "Useful as a baseline to compare against pretrained embeddings."
    )
    parser.add_argument(
        "--trainer", type=str, required=True,
        help="Trainer class the checkpoint was produced with. This determines the "
             "architecture that the weights are loaded into, so it must match the "
             "pre-training run exactly."
    )
    parser.add_argument(
        "--plans", type=str, default="nnsslPlans",
        help="Plans identifier (default: nnsslPlans)"
    )
    parser.add_argument(
        "--output_file", type=str, default="embeddings.npz",
        help="Output .npz file for embeddings"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for inference (default: 2, keep small for 3D volumes)"
    )
    parser.add_argument(
        "--pool_stage", type=str, default="bottleneck",
        choices=["bottleneck", "all_concat"],
        help="Which encoder stage to pool (default: bottleneck)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit total number of samples"
    )
    parser.add_argument(
        "--use_patch_size", action="store_true",
        help="Read patch_size from the plan (recommended)"
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=3, default=None,
        metavar=("D", "H", "W"),
        help="Manually specify patch size as D H W, e.g. --patch_size <D> <H> <W>. "
             "Overrides --use_patch_size if both are given."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of dataloader workers for parallel data loading (default: 4)"
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help=f"Dataset annotation table (CSV/TSV/XLSX) supplying the known dataset ids used "
             f"to map image filenames back to their source dataset. "
             f"Default: {DEFAULT_METADATA_CSV}"
    )
    parser.add_argument(
        "--infer_labels_from_filename", dest="infer_labels_from_filename",
        action="store_true", default=None,
        help="Always derive per-image labels from filenames, ignoring the dataset keys in "
             "pretrain_data.json. Default is automatic (see --lumped_json_max_keys)."
    )
    parser.add_argument(
        "--no_infer_labels_from_filename", dest="infer_labels_from_filename",
        action="store_false",
        help="Always use the dataset keys from pretrain_data.json as labels."
    )
    parser.add_argument(
        "--lumped_json_max_keys", type=int, default=None,
        help="[REQUIRED if labeling mode is left automatic] If pretrain_data.json has at "
             "most this many dataset keys it is treated as 'lumped' and labels are parsed "
             "from filenames instead. Not needed if --infer_labels_from_filename or "
             "--no_infer_labels_from_filename forces the mode explicitly."
    )
    args = parser.parse_args()

    # Dataset ids used by the filename -> dataset mapping.
    load_dataset_ids(args.metadata)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Step 1: Build trainer & load network ──────────────────────
    # Use the FIRST dataset_id for trainer/network initialization
    # For raw_data_path mode, still need a dataset_id for the trainer architecture
    if args.dataset_id:
        primary_id = args.dataset_id[0]
    else:
        # Even in --raw_data_path mode a dataset id is needed, because the plans
        # and dataset.json of that dataset define the architecture the checkpoint
        # is loaded into. Guessing one would build the wrong network and load the
        # weights into it, so require it explicitly.
        raise SystemExit(
            "[ERROR] --dataset_id is required: its plans define the network architecture "
            "the checkpoint is loaded into.\n"
            "        Pass the dataset the model was trained on, even when reading volumes "
            "from --raw_data_path."
        )

    print(f"\n[STEP 1] Initializing trainer: {args.trainer} (using dataset {primary_id})")
    trainer = get_trainer_from_args(
        dataset_name_or_id=primary_id,
        configuration=args.config,
        fold=0,
        trainer_name=args.trainer,
        plans_identifier=args.plans,
        device=device,
    )

    # The trainer has a .network attribute and an .initialize() or similar method.
    # We need to make sure the network is built before loading weights.
    print("[STEP 1] Initializing trainer (building network)...")
    # Most nnSSL trainers have an initialize() method that builds the network
    if hasattr(trainer, 'initialize'):
        trainer.initialize()
    elif hasattr(trainer, 'build_network_architecture'):
        trainer.build_network_architecture()

    # ── Step 2: Load pretrained checkpoint (or skip for random baseline) ─
    if args.random_init:
        print(f"\n[STEP 2] *** RANDOM INIT MODE — skipping checkpoint loading ***")
        print(f"[INFO] Using randomly initialized weights as baseline.")
        print(f"[INFO] Compare k-NN accuracy against pretrained to measure pretraining gain.")
    elif args.checkpoint:
        print(f"\n[STEP 2] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # nnSSL checkpoints may store weights under different keys
        if isinstance(checkpoint, dict):
            # Try common key patterns
            for key in ['network_weights', 'state_dict', 'model_state_dict', 'model', 'encoder']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"[INFO] Found weights under key '{key}'")
                    break
            else:
                # If no known key, assume the dict IS the state_dict
                state_dict = checkpoint
                print("[INFO] Using checkpoint dict directly as state_dict")
        else:
            state_dict = checkpoint

        # Load weights into the network
        # Use strict=False in case the checkpoint has decoder weights we don't need
        missing, unexpected = trainer.network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[INFO] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print("[INFO] Checkpoint loaded successfully.")
    else:
        print("[ERROR] Must specify either --checkpoint or --random_init")
        sys.exit(1)

    # ── Step 3: Build feature extractor ───────────────────────────
    print(f"\n[STEP 3] Building feature extractor (pool_stage={args.pool_stage})")
    feature_extractor = EncoderFeatureExtractor(
        network=trainer.network,
        pool_stage=args.pool_stage,
    )
    feature_extractor.to(device)
    feature_extractor.eval()

    # ── Step 4: Build dataset(s) & dataloader ───────────────────────
    # Validate: must specify one of --dataset_id or --raw_data_path
    if not args.dataset_id and not args.raw_data_path:
        print("[ERROR] Must specify either --dataset_id or --raw_data_path")
        sys.exit(1)

    # Determine patch_size ONCE (same architecture for all shards)
    patch_size = None
    if args.use_patch_size:
        try:
            patch_size = tuple(trainer.config_plan.patch_size)
            print(f"[INFO] Using patch_size from plan: {patch_size}")
        except Exception as e:
            print(f"[WARN] Could not get patch_size from plan: {e}")
    if patch_size is None and args.patch_size is not None:
        patch_size = tuple(args.patch_size)
        print(f"[INFO] Using user-specified patch_size: {patch_size}")
    if patch_size is None:
        raise SystemExit(
            "[ERROR] No patch size determined.\n"
            "        All volumes must be cropped/padded to a common size before batching.\n"
            "        Pass --use_patch_size to read it from the training plan (recommended,\n"
            "        and what the released model was trained with), or set it explicitly,\n"
            "        e.g. --patch_size 224 192 224."
        )

    def collate_fn(batch):
        """Custom collate that handles variable-size volumes."""
        return {
            "data": torch.stack([b["data"] for b in batch]),
            "name": [b["name"] for b in batch],
            "label": [b["label"] for b in batch],
        }

    if args.raw_data_path:
        # ══════════════════════════════════════════════════════════
        # RAW DATA MODE: scan directories directly
        # ══════════════════════════════════════════════════════════
        print(f"\n[STEP 4] RAW DATA MODE — scanning {len(args.raw_data_path)} path(s)")
        dataset = RawDataDataset(
            root_paths=args.raw_data_path,
            patch_size=patch_size,
            max_samples=args.max_samples,
        )

        if len(dataset) == 0:
            print("[ERROR] No volumes found! Check --raw_data_path.")
            sys.exit(1)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # ── Step 5: Extract ───────────────────────────────────────
        print(f"\n[STEP 5] Extracting embeddings from {len(dataset)} volumes...")
        results = extract_all_embeddings(feature_extractor, dataloader, device)

        all_embeddings = [results["embeddings"]]
        all_names = list(results["names"])
        all_labels = list(results["labels"])

    else:
        # ══════════════════════════════════════════════════════════
        # PREPROCESSED MODE: use nnSSL dataset_id(s)
        # ══════════════════════════════════════════════════════════
        print(f"\n[STEP 4] Building dataset from {len(args.dataset_id)} dataset(s)")

        # Collect from all dataset_ids
        all_embeddings = []
        all_names = []
        all_labels = []
        total_images = 0

        # Calculate per-dataset sample budget if max_samples is set
        if args.max_samples is not None:
            per_dataset_budget = args.max_samples // len(args.dataset_id)
            print(f"[INFO] max_samples={args.max_samples} across {len(args.dataset_id)} datasets "
                  f"→ ~{per_dataset_budget} per dataset")
        else:
            per_dataset_budget = None

        for ds_idx, ds_id in enumerate(args.dataset_id):
            print(f"\n{'─'*60}")
            print(f"[STEP 4.{ds_idx+1}] Loading dataset {ds_id} "
                  f"({ds_idx+1}/{len(args.dataset_id)})")

            dataset_name = maybe_convert_to_dataset_name(ds_id)
            preprocessed_folder = join(nnssl_preprocessed, dataset_name)

            pretrain_json_path = join(preprocessed_folder, f"pretrain_data__{args.config}.json")
            if not os.path.exists(pretrain_json_path):
                alt_path = join(preprocessed_folder, "pretrain_data.json")
                if os.path.exists(alt_path):
                    pretrain_json_path = alt_path
                else:
                    print(f"[WARN] Not found: {pretrain_json_path}")
                    print(f"[WARN] Skipping dataset {ds_id}")
                    continue

            print(f"[INFO] Loading: {pretrain_json_path}")
            pretrain_json = load_json(pretrain_json_path)

            dataset = NnSSLPreprocessedDataset(
                pretrain_json=pretrain_json,
                preprocessed_folder=preprocessed_folder,
                patch_size=patch_size,
                max_samples=per_dataset_budget,
                infer_labels_from_filename=args.infer_labels_from_filename,
                lumped_json_max_keys=args.lumped_json_max_keys,
            )

            if len(dataset) == 0:
                print(f"[WARN] No images found in {ds_id}, skipping.")
                continue

            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_fn,
                prefetch_factor=2 if args.num_workers > 0 else None,
            )

            # ── Step 5: Extract embeddings for this dataset ───────
            print(f"[STEP 5.{ds_idx+1}] Extracting embeddings from {len(dataset)} images...")
            results = extract_all_embeddings(feature_extractor, dataloader, device)

            all_embeddings.append(results["embeddings"])
            all_names.extend(results["names"])
            all_labels.extend(results["labels"])
            total_images += results["embeddings"].shape[0]
            print(f"[INFO] Running total: {total_images} embeddings")

            # Early stop if we've hit the budget
            if args.max_samples is not None and total_images >= args.max_samples:
                print(f"[INFO] Reached max_samples={args.max_samples}, stopping.")
                break

        if not all_embeddings:
            print("[ERROR] No embeddings extracted from any dataset!")
            sys.exit(1)

    # ── Step 6: Merge & Save (shared by both modes) ──────────────
    merged_embeddings = np.concatenate(all_embeddings, axis=0)
    merged_names = np.array(all_names)
    merged_labels = np.array(all_labels)

    # Trim to exact max_samples if slightly over
    if args.max_samples is not None and len(merged_embeddings) > args.max_samples:
        merged_embeddings = merged_embeddings[:args.max_samples]
        merged_names = merged_names[:args.max_samples]
        merged_labels = merged_labels[:args.max_samples]

    output_path = args.output_file
    np.savez(
        output_path,
        embeddings=merged_embeddings,
        names=merged_names,
        labels=merged_labels,
    )
    print(f"\n{'='*60}")
    print(f"[DONE] Embeddings saved to: {output_path}")
    print(f"  Total embeddings: {merged_embeddings.shape[0]}")
    print(f"  Embedding dim: {merged_embeddings.shape[1]}")
    print(f"  File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    print(f"  Unique labels: {len(np.unique(merged_labels))}")
    if args.raw_data_path:
        print(f"  Source: raw data from {args.raw_data_path}")
    else:
        print(f"  Source: dataset_id {args.dataset_id}")
    print(f"\nNext step: python plot_embeddings.py --input {output_path} --color_by umap_all")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
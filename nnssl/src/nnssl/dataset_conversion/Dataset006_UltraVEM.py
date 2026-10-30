#!/usr/bin/env python3
"""Build the ``pretrain_data.json`` describing a UltraVEM volume-EM pre-training corpus.

The corpus is referenced in place -- no image data is copied. Two on-disk layouts
are supported:

``flat``
    All volumes sit directly in ``--parent_dir``::

        <parent_dir>/<dataset>_<subject>_<channel>.nii.gz

    Every volume is grouped under a single dataset entry (``--dataset-name``).

``nested``
    One sub-directory per source dataset, each holding a curated sub-folder::

        <parent_dir>/<dataset>/<curated_subdir>/<dataset>_<subject>_<channel>.nii.gz

    Each sub-directory becomes its own dataset entry in the collection, which
    keeps the provenance of the individual source datasets intact.

Volume EM has no notion of longitudinal sessions, so every image is filed under a
single synthetic ``ses-DEFAULT`` session; the subject grouping is taken from the
second underscore-separated field of the filename.
"""

import argparse
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm

from nnssl.data.raw_dataset import (
    AssociatedMasks,
    Collection,
    Dataset,
    Image,
    Session,
    Subject,
)

MODALITY = "EM"
SESSION_ID = "ses-DEFAULT"


def normalize_dataset_id(folder_name: str) -> str:
    """Normalize a folder name into a dataset identifier (e.g. ``my-dataset-2`` -> ``MY_DATASET_2``)."""
    return folder_name.replace("jrc_", "").replace("-", "_").upper()


def add_images(collection: Collection, dataset_name: str, nii_files: list[Path]) -> int:
    """Register ``nii_files`` under ``dataset_name``, returning the number of images added."""
    if dataset_name not in collection.datasets:
        collection.datasets[dataset_name] = Dataset(dataset_index=dataset_name, name=None, dataset_info={})
    dataset = collection.datasets[dataset_name]

    n_added = 0
    for file in tqdm(nii_files, desc=f"Processing {dataset_name}"):
        name = file.stem
        if name.endswith(".nii"):
            name = name[:-4]

        # Expected filename layout: <dataset>_<subject_id>_<channel>.nii.gz
        parts = name.split("_")
        if len(parts) < 2:
            print(f"Skipping file with unexpected name: {file.name}")
            continue
        subject_id = parts[1]

        if subject_id not in dataset.subjects:
            dataset.subjects[subject_id] = Subject(subject_id=subject_id, subject_info={})
        subject = dataset.subjects[subject_id]

        if SESSION_ID not in subject.sessions:
            subject.sessions[SESSION_ID] = Session(session_id=SESSION_ID, images=[])

        subject.sessions[SESSION_ID].images.append(
            Image(
                name=name,
                image_path=str(file.resolve()),
                modality=MODALITY,
                image_info={},
                associated_masks=AssociatedMasks(anonymization_mask=None, anatomy_mask=None),
            )
        )
        n_added += 1
    return n_added


def create_collection(
    parent_dir: Path,
    collection_name: str,
    layout: str,
    dataset_name: str,
    curated_subdir: str,
) -> Collection:
    assert parent_dir.is_dir(), f"{parent_dir} is not a valid directory"
    print(f"Scanning data in: {parent_dir} (layout={layout})")

    collection = Collection(collection_name=collection_name, collection_index=1)
    total = 0

    if layout == "flat":
        nii_files = sorted(parent_dir.glob("*.nii.gz"))
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI files found in {parent_dir}")
        print(f"Found {len(nii_files)} NIfTI files")
        total += add_images(collection, dataset_name, nii_files)
    else:
        for dataset_dir in sorted(d for d in parent_dir.iterdir() if d.is_dir()):
            curated_dir = dataset_dir / curated_subdir if curated_subdir else dataset_dir
            if not curated_dir.is_dir():
                print(f"No {curated_subdir!r} folder in {dataset_dir.name}, skipping...")
                continue
            nii_files = sorted(curated_dir.glob("*.nii.gz"))
            if not nii_files:
                print(f"No NIfTI files found in {curated_dir}")
                continue
            print(f"Found {len(nii_files)} NIfTI files in {dataset_dir.name}/{curated_subdir}")
            total += add_images(collection, normalize_dataset_id(dataset_dir.name), nii_files)

    if total == 0:
        raise FileNotFoundError(f"No usable NIfTI files were found under {parent_dir}")
    print(f"\nCollected {total} images across {len(collection.datasets)} dataset(s)")
    return collection


def main():
    parser = argparse.ArgumentParser(
        description="Create the pretrain_data.json for a UltraVEM volume-EM pre-training corpus."
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        required=True,
        help="Directory holding the curated .nii.gz volumes (see --layout).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="nnssl_raw directory; the dataset folder is created inside it.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=6,
        help="Numeric nnssl dataset id; the output folder is DatasetXXX_<collection-name>. Default: 6.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="UltraVEM",
        help="Name of the collection and suffix of the output folder. Default: UltraVEM.",
    )
    parser.add_argument(
        "--layout",
        choices=("flat", "nested"),
        default="flat",
        help="'flat': all volumes directly in --parent_dir. 'nested': one sub-directory per source dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ULTRAVEM_CURATED",
        help="Dataset entry name used for the 'flat' layout. Default: ULTRAVEM_CURATED.",
    )
    parser.add_argument(
        "--curated-subdir",
        type=str,
        default="stage3_curated",
        help="Sub-folder searched inside each dataset dir for the 'nested' layout. "
        "Pass an empty string to read the dataset dir itself. Default: stage3_curated.",
    )
    args = parser.parse_args()

    folder_name = f"Dataset{args.dataset_id:03d}_{args.collection_name}"
    collection = create_collection(
        parent_dir=args.parent_dir,
        collection_name=folder_name,
        layout=args.layout,
        dataset_name=args.dataset_name,
        curated_subdir=args.curated_subdir,
    )

    output_dir = args.output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pretrain_data.json"
    save_json(collection.to_dict(relative_paths=False), str(output_path), indent=4, sort_keys=True)
    print(f"Saved pretrain_data.json to: {output_path}")


if __name__ == "__main__":
    main()

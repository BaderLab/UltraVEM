#!/usr/bin/env python3
"""
classification_benchmark.py
============================
k-NN and linear probe classification benchmark for UltraVEM embeddings.

This script does:
  1. Loads embeddings.npz (from extract_embeddings.py)
  2. Loads the dataset annotation table (metadata/dataset_metadata.csv)
     to get coarse & fine labels
  3. Maps each image → coarse_label + fine_label via the dataset name
  4. Splits into train/test (stratified)
  5. Runs k-NN classification (no training needed)
  6. Runs linear probe (LogisticRegression on frozen features)
  7. Optionally runs data-efficiency curves over reader-chosen training fractions
  8. Saves results as JSON + prints paper-ready tables

Can benchmark multiple embeddings files (pretrained vs random init) side by side.

No hyperparameter that affects a reported number has a built-in default (see
--help): every one is required on the command line, so a result can always be
traced back to the exact settings that produced it.

Usage:
  # Single model evaluation:
  python classification_benchmark.py \
      --embeddings /path/to/embeddings.npz \
      --metadata metadata/dataset_metadata.csv \
      --model_name "<model name>" \
      --knn_k <k> --knn_k_values <k1> <k2> <k3> \
      --test_fraction <fraction> --min_samples <n> --seed <seed> \
      --linear_max_iter <max_iter> --linear_C <C>

  # Compare pretrained vs random init:
  python classification_benchmark.py \
      --embeddings /path/to/pretrained_embeddings.npz /path/to/random_embeddings.npz \
      --model_names "<model A>" "<model B>" \
      --metadata metadata/dataset_metadata.csv \
      --knn_k <k> --knn_k_values <k1> <k2> <k3> \
      --test_fraction <fraction> --min_samples <n> --seed <seed> \
      --linear_max_iter <max_iter> --linear_C <C>

  # With data-efficiency curve (adds --de_fractions and --de_repeats):
  python classification_benchmark.py \
      --embeddings /path/to/embeddings.npz \
      --metadata metadata/dataset_metadata.csv \
      --knn_k <k> --knn_k_values <k1> <k2> <k3> \
      --test_fraction <fraction> --min_samples <n> --seed <seed> \
      --linear_max_iter <max_iter> --linear_C <C> \
      --data_efficiency --de_fractions <f1> <f2> ... --de_repeats <n>
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)

warnings.filterwarnings('ignore', category=UserWarning)

# Label granularities this benchmark knows how to evaluate. Used both to
# validate --label_levels and to recognise a label level when merging saved
# result JSONs (see _load_and_merge_jsons) -- a single source of truth so the
# two can't silently drift apart.
VALID_LABEL_LEVELS = ('coarse', 'fine', 'organism', 'modality')

# No hyperparameter here has a built-in numeric default (see main()'s argparse
# section): every value that would change a reported number is required from
# the command line, so a run can never silently pick up a number nobody chose.


# ============================================================================
# Embedding preprocessing
# ============================================================================

def preprocess_embeddings(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess embeddings before classification.
    
    L2 normalization is CRITICAL for fair comparison between pretrained
    and random init models. Why?
    
    Random init encoders produce features with dataset-specific magnitude 
    patterns — different microscopes/staining produce different overall 
    intensity statistics, which translate to different feature vector norms.
    A classifier can exploit these magnitude differences as a "shortcut" 
    to achieve high accuracy without learning any real biological structure.
    
    L2 normalization removes this shortcut by projecting all embeddings 
    onto the unit hypersphere, forcing the classifier to use DIRECTIONAL 
    structure (angular relationships between features) rather than magnitude.
    This is the standard practice in contrastive learning evaluation
    (SimCLR, MoCo, DINO, EM-DINO all use L2-normalized features for 
    classification benchmarks).
    """
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        X = X / norms
    return X

DEFAULT_METADATA_CSV = Path(__file__).resolve().parent / "metadata" / "dataset_metadata.csv"

# Columns are matched case-insensitively, accepting either the names used by
# metadata/dataset_metadata.csv or the longer ones from the original spreadsheet
# export, so both work without editing the table.
_COLUMN_ALIASES = {
    'dataset_id':    ('dataset_id',),
    'organism':      ('organism',),
    'tissue_fine':   ('tissue_fine', 'tissue_organ_fine'),
    'tissue_coarse': ('tissue_coarse', 'tissue_organ_coarse'),
    'fine_label':    ('umap_fine', 'umap_label_fine'),
    'coarse_label':  ('umap_coarse', 'umap_label_coarse'),
    'modality':      ('modality',),
}


def load_metadata(path: str) -> list:
    """Read the dataset annotation table into a list of lower-cased dict rows.

    Accepts .csv/.tsv (stdlib) and .xlsx/.xls (via pandas).
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"[ERROR] Dataset metadata not found: {path}\n"
            f"        Pass --metadata /path/to/table.csv "
            f"(schema: {DEFAULT_METADATA_CSV})"
        )
    if path.suffix.lower() in ('.xlsx', '.xls'):
        try:
            df = pd.read_excel(path, sheet_name='Combined_All_Datasets')
        except Exception:
            df = pd.read_excel(path, sheet_name=0)
        rows = df.to_dict('records')
    else:
        delim = '\t' if path.suffix.lower() in ('.tsv', '.tab') else ','
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f, delimiter=delim))
    return [{(k or '').strip().lower(): ('' if v is None else str(v).strip())
             for k, v in row.items()} for row in rows]


def build_dataset_label_map(metadata_rows: list) -> dict:
    """
    Build mapping: dataset_id -> {fine_label, coarse_label, organism, ...}

    The dataset_id must correspond to the per-image label stored in
    embeddings.npz; resolve_label() below also does fuzzy/prefix matching.
    """
    def pick(row, field):
        for name in _COLUMN_ALIASES[field]:
            val = row.get(name, '')
            if val and val.lower() != 'nan':
                return val
        return ''

    label_map = {}
    for row in metadata_rows:
        did = pick(row, 'dataset_id')
        if not did:
            continue
        organism = pick(row, 'organism') or 'Unknown'
        tissue_fine = pick(row, 'tissue_fine') or 'Unknown'
        tissue_coarse = pick(row, 'tissue_coarse') or 'Unknown'

        # The explicit umap_* columns win. When absent, derive "<organism> - <tissue>"
        # so that the class is organism-aware -- otherwise e.g. mouse brain and
        # Drosophila brain would collapse into a single "Brain" class.
        def derive(explicit, tissue):
            if explicit:
                return explicit
            if organism != 'Unknown' and tissue != 'Unknown':
                return f'{organism} - {tissue}'
            return tissue

        label_map[did] = {
            'fine_label': derive(pick(row, 'fine_label'), tissue_fine),
            'coarse_label': derive(pick(row, 'coarse_label'), tissue_coarse),
            'organism': organism,
            'tissue_fine': tissue_fine,
            'tissue_coarse': tissue_coarse,
            'modality': pick(row, 'modality') or 'Unknown',
            # Alternative names that should resolve to this dataset.
            'aliases': [a.strip() for a in row.get('aliases', '').split(';') if a.strip()],
        }
    if not label_map:
        raise SystemExit("[ERROR] No usable rows in the metadata table "
                         "(is the 'dataset_id' column present?)")
    return label_map


def build_fuzzy_lookup(label_map: dict) -> dict:
    """
    Build a fuzzy lookup: normalized_key -> dataset_id.
    Normalization removes hyphens, underscores and spaces, and lowercases.
    Alternative names from the table's `aliases` column are included, so a label
    that bears no resemblance to the canonical id still resolves.
    """
    def norm(s):
        return s.replace('-', '').replace('_', '').replace(' ', '').lower()

    fuzzy = {}
    for did, entry in label_map.items():
        fuzzy[norm(did)] = did
    # Aliases are added second, and never overwrite a real dataset id.
    for did, entry in label_map.items():
        for alias in entry.get('aliases', []):
            fuzzy.setdefault(norm(alias), did)
    return fuzzy


def resolve_label(raw_label: str, label_map: dict, fuzzy_lookup: dict) -> dict:
    """
    Resolve a raw label (from embeddings.npz) to its metadata entry.
    Returns the metadata dict or None if not found.
    """
    # Direct match
    if raw_label in label_map:
        return label_map[raw_label]
    
    # Fuzzy match
    normalized = raw_label.replace('-', '').replace('_', '').replace(' ', '').lower()
    if normalized in fuzzy_lookup:
        return label_map[fuzzy_lookup[normalized]]
    
    # Partial match: try if raw_label is a prefix of any Dataset_ID
    for did in label_map:
        did_norm = did.replace('-', '').replace('_', '').replace(' ', '').lower()
        if did_norm.startswith(normalized) or normalized.startswith(did_norm):
            return label_map[did]
    
    return None


# ============================================================================
# Core classification experiments
# ============================================================================

def run_knn(X_train, y_train, X_test, y_test, k_values):
    """
    Run k-NN classification for multiple k values.

    This is the simplest evaluation: no training at all.
    Just finds the k nearest neighbors in the training set and votes.
    Reports TOP-1 accuracy (like Cryo-IEF's "TOP1 k-NN" metric).
    """
    results = {}
    for k in k_values:
        if k > len(X_train):
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[f'knn_k{k}'] = {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
        }
    
    return results


def run_linear_probe(X_train, y_train, X_test, y_test,
                     max_iter, C, return_model=False):
    """
    Run linear probe classification (LogisticRegression).
    
    === What is a linear probe? ===
    
    A linear probe is the simplest possible classifier: a single linear layer
    (no hidden layers, no nonlinearities). It takes the frozen encoder features
    and learns a weight matrix W and bias b such that:
    
        predicted_class = argmax(W @ feature_vector + b)
    
    That's it. It's equivalent to drawing straight-line decision boundaries
    in the feature space.
    
    WHY does this matter? If a linear probe gets HIGH accuracy, it means the 
    pretrained features are already well-separated in feature space — the 
    different classes form distinct clusters that can be separated by hyperplanes.
    This is a direct measure of FEATURE QUALITY from pretraining.
    
    If a random init encoder gets low linear probe accuracy but pretrained gets
    high accuracy, that proves pretraining learned meaningful representations.
    
    WHY does it get high accuracy in papers? Because good pretraining 
    (contrastive learning, MAE, etc.) explicitly encourages the encoder to 
    produce features where similar things are close and different things are 
    far apart. A linear classifier then trivially separates them.
    It's "easy" BY DESIGN — that's the point of the evaluation.
    ===
    """
    # Normalize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # L-BFGS solver handles multi-class well, with L2 regularization
    clf = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs',
        C=C,  # Regularization strength (inverse)
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }
    # Callers that also want the per-sample predictions (e.g. to build a
    # classification report) take them from here rather than refitting.
    return (metrics, y_pred) if return_model else metrics


def run_data_efficiency(X_train, y_train, X_test, y_test,
                        fractions, n_repeats, seed, knn_k,
                        linear_max_iter, linear_C, extra_knn_ks=None):
    """
    Run BOTH k-NN and linear probe with varying amounts of training data.
    This produces the data-efficiency curve.

    At each fraction, we subsample the training set (stratified),
    run both classifiers, and report accuracy. Repeated n_repeats times
    for error bars (at 100% there's no variance since no subsampling).

    k-NN data efficiency: fewer training points = fewer neighbors to look up.
    This tests whether the feature space is structured enough that even a
    handful of labeled examples suffice to classify new images.

    extra_knn_ks: additional k values to also evaluate at each fraction,
    stored as knn_k{k} keys alongside the primary 'knn' key.
    """
    extra_knn_ks = [k for k in (extra_knn_ks or []) if k != knn_k]
    results = {}
    rng = np.random.RandomState(seed)

    for frac in fractions:
        lp_accs, lp_f1s = [], []
        knn_accs, knn_f1s = [], []
        extra_accs = {k: [] for k in extra_knn_ks}
        extra_f1s  = {k: [] for k in extra_knn_ks}

        # At 100% there is no subsampling, so every repeat is identical: run it
        # once rather than n_repeats times (the std is 0 by construction).
        reps = 1 if frac >= 1.0 else n_repeats
        n_sub_actual = len(X_train)

        for rep in range(reps):
            if frac >= 1.0:
                X_sub, y_sub = X_train, y_train
            else:
                n_sub = max(1, int(len(X_train) * frac))
                n_sub_actual = n_sub
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub,
                                                random_state=seed + rep)
                    idx, _ = next(sss.split(X_train, y_train))
                    X_sub = X_train[idx]
                    y_sub = y_train[idx]
                except ValueError:
                    idx = rng.choice(len(X_train), n_sub, replace=False)
                    X_sub = X_train[idx]
                    y_sub = y_train[idx]

            if len(np.unique(y_sub)) < 2:
                continue

            # Linear probe
            lp_res = run_linear_probe(X_sub, y_sub, X_test, y_test,
                                      max_iter=linear_max_iter, C=linear_C)
            lp_accs.append(lp_res['accuracy'])
            lp_f1s.append(lp_res['f1_macro'])

            # Primary k-NN (adjust k if fewer samples than k)
            effective_k = min(knn_k, len(X_sub))
            if effective_k < 1:
                effective_k = 1
            knn = KNeighborsClassifier(n_neighbors=effective_k, metric='cosine', n_jobs=-1)
            knn.fit(X_sub, y_sub)
            y_pred_knn = knn.predict(X_test)
            knn_accs.append(float(accuracy_score(y_test, y_pred_knn)))
            knn_f1s.append(float(f1_score(y_test, y_pred_knn, average='macro', zero_division=0)))

            # Extra k values
            for ek in extra_knn_ks:
                eff_ek = min(ek, len(X_sub))
                if eff_ek < 1:
                    eff_ek = 1
                knn_ek = KNeighborsClassifier(n_neighbors=eff_ek, metric='cosine', n_jobs=-1)
                knn_ek.fit(X_sub, y_sub)
                y_pred_ek = knn_ek.predict(X_test)
                extra_accs[ek].append(float(accuracy_score(y_test, y_pred_ek)))
                extra_f1s[ek].append(float(f1_score(y_test, y_pred_ek, average='macro', zero_division=0)))

        if lp_accs:
            frac_pct = f"{int(frac * 100)}%"
            primary_knn_entry = {
                'accuracy_mean': float(np.mean(knn_accs)),
                'accuracy_std': float(np.std(knn_accs)),
                'f1_macro_mean': float(np.mean(knn_f1s)),
                'f1_macro_std': float(np.std(knn_f1s)),
            }
            results[frac_pct] = {
                'linear_probe': {
                    'accuracy_mean': float(np.mean(lp_accs)),
                    'accuracy_std': float(np.std(lp_accs)),
                    'f1_macro_mean': float(np.mean(lp_f1s)),
                    'f1_macro_std': float(np.std(lp_f1s)),
                },
                'knn': primary_knn_entry,
                f'knn_k{knn_k}': primary_knn_entry,  # named key for explicit lookup
                'n_train_samples': int(n_sub_actual),
            }
            for ek in extra_knn_ks:
                if extra_accs[ek]:
                    results[frac_pct][f'knn_k{ek}'] = {
                        'accuracy_mean': float(np.mean(extra_accs[ek])),
                        'accuracy_std': float(np.std(extra_accs[ek])),
                        'f1_macro_mean': float(np.mean(extra_f1s[ek])),
                        'f1_macro_std': float(np.std(extra_f1s[ek])),
                    }

    return results


# ============================================================================
# Main pipeline
# ============================================================================

def evaluate_embeddings(
    embeddings: np.ndarray,
    labels_raw: np.ndarray,
    names: np.ndarray,
    label_map: dict,
    fuzzy_lookup: dict,
    knn_k: int,
    knn_k_values: list,
    test_fraction: float,
    min_samples_per_class: int,
    seed: int,
    linear_max_iter: int,
    linear_C: float,
    label_level: str = 'coarse',
    data_efficiency: bool = False,
    normalize: bool = True,
    extra_knn_ks: list = None,
    de_fractions: list = None,
    de_repeats: int = None,
    group_split: bool = False,
):
    """
    Full evaluation pipeline for one set of embeddings.
    
    Returns a results dict with k-NN, linear probe, and optionally data-efficiency.
    """
    # Step 0: L2-normalize embeddings
    embeddings = preprocess_embeddings(embeddings, normalize=normalize)
    
    # Step 1: Map raw labels to coarse/fine classification labels
    mapped_labels = []
    unmapped_count = 0
    unmapped_datasets = set()
    
    for raw in labels_raw:
        raw_str = str(raw)
        meta = resolve_label(raw_str, label_map, fuzzy_lookup)
        if meta is not None:
            # Map label_level to the appropriate metadata field
            level_to_key = {
                'coarse': 'coarse_label',
                'fine': 'fine_label',
                'organism': 'organism',
                'modality': 'modality',
            }
            key = level_to_key.get(label_level, 'coarse_label')
            mapped_labels.append(meta[key])
        else:
            mapped_labels.append(None)
            unmapped_count += 1
            unmapped_datasets.add(raw_str)
    
    mapped_labels = np.array(mapped_labels)

    # Report mapping. mapped_labels is an object array, so compare elementwise
    # rather than with `!=`, which broadcasts oddly against None.
    valid_mask = np.array([l is not None for l in mapped_labels])

    print(f"\n  Label mapping ({label_level}):")
    print(f"    Mapped: {valid_mask.sum()} / {len(mapped_labels)}")
    if unmapped_count > 0:
        print(f"    Unmapped ({unmapped_count}): {sorted(unmapped_datasets)[:10]}")
    
    # Filter to valid only. The source-dataset id of every kept sample is carried
    # along so that it can be used as a grouping key for the split below.
    labels_raw_arr = np.asarray([str(r) for r in labels_raw])
    embeddings_valid = embeddings[valid_mask]
    labels_valid = mapped_labels[valid_mask].astype(str)
    groups_valid = labels_raw_arr[valid_mask]

    # Step 2: Filter classes with too few samples
    unique, counts = np.unique(labels_valid, return_counts=True)
    keep_classes = unique[counts >= min_samples_per_class]
    class_mask = np.isin(labels_valid, keep_classes)

    embeddings_valid = embeddings_valid[class_mask]
    labels_valid = labels_valid[class_mask]
    groups_valid = groups_valid[class_mask]

    print(f"    Classes with >= {min_samples_per_class} samples: {len(keep_classes)} / {len(unique)}")
    print(f"    Total samples after filtering: {len(labels_valid)}")
    
    # Encode labels as integers
    le = LabelEncoder()
    y = le.fit_transform(labels_valid)
    X = embeddings_valid
    num_classes = len(le.classes_)
    
    print(f"    Number of classes: {num_classes}")
    
    # Print class distribution
    unique_y, counts_y = np.unique(y, return_counts=True)
    print(f"\n    Class distribution:")
    for cls_idx, cnt in sorted(zip(unique_y, counts_y), key=lambda x: -x[1]):
        print(f"      [{cls_idx:2d}] {le.classes_[cls_idx]}: {cnt}")
    
    # Step 3: Train/test split
    #
    # Default (--group_split off): a stratified split over individual volumes.
    # Note that several volumes are typically cropped from the SAME source
    # dataset, so train and test can contain highly correlated neighbours. This
    # measures how well the features separate classes, but it is optimistic as a
    # measure of generalization to an unseen dataset.
    #
    # --group_split on: whole source datasets are held out, so no dataset
    # contributes to both train and test. This is the stricter protocol; it
    # requires each class to be covered by at least two source datasets.
    if group_split:
        n_groups = len(np.unique(groups_valid))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups_valid))
        held_out = sorted({str(g) for g in groups_valid[test_idx]})
        print(f"\n    Group-aware split: {len(held_out)} / {n_groups} source datasets held out")
        print(f"      Held out: {held_out[:8]}{' ...' if len(held_out) > 8 else ''}")

        # Holding out whole datasets can strip every training example of a class.
        # Such a class is unlearnable, and sklearn would otherwise fail later with
        # an opaque "solver needs samples of at least 2 classes" error. Drop those
        # test samples and report it, rather than crashing cryptically.
        trainable = set(np.unique(y[train_idx]).tolist())
        missing = sorted(set(np.unique(y).tolist()) - trainable)
        if missing:
            keep = np.isin(y[test_idx], list(trainable))
            dropped = int((~keep).sum())
            test_idx = test_idx[keep]
            print(f"      WARNING: {len(missing)} class(es) have no training data after the "
                  f"group split: {[le.classes_[c] for c in missing]}")
            print(f"      Dropped {dropped} test sample(s) of those classes. To avoid this, "
                  f"lower --test_fraction, use a coarser --label_levels, or ensure every "
                  f"class is covered by at least two source datasets.")
        if len(trainable) < 2:
            raise SystemExit(
                f"[ERROR] --group_split left only {len(trainable)} class(es) in the training "
                f"split; at least 2 are required.\n"
                f"        This corpus has too few source datasets per class for a dataset-level "
                f"split at label level '{label_level}'.\n"
                f"        Use a coarser --label_levels, lower --test_fraction, or omit --group_split."
            )
        if len(test_idx) == 0:
            raise SystemExit("[ERROR] --group_split left an empty test set. "
                             "Lower --test_fraction or omit --group_split.")
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\n    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 4: Run benchmarks
    results = {
        'num_classes': num_classes,
        'num_train': len(X_train),
        'num_test': len(X_test),
        'embedding_dim': X.shape[1],
        'class_names': list(le.classes_),
        'label_level': label_level,
        'knn_k': knn_k,
        'split': 'group' if group_split else 'stratified',
        'test_source_datasets': sorted({str(g) for g in groups_valid[test_idx]}),
    }
    
    # k-NN. The primary k must be present in the evaluated set, otherwise the
    # summary/plotting lookup (_get_knn_key) silently falls back to another k.
    k_values = sorted(set(knn_k_values) | {knn_k})
    print(f"\n    Running k-NN classification (k={k_values}, primary k={knn_k})...")
    knn_results = run_knn(X_train, y_train, X_test, y_test, k_values=k_values)
    results['knn'] = knn_results

    best_knn_k = max(knn_results, key=lambda k: knn_results[k]['accuracy'])
    print(f"    Best k-NN: {best_knn_k} -> acc={knn_results[best_knn_k]['accuracy']:.4f}")

    # Linear probe. Keep the fitted predictions so the classification report
    # below describes exactly this model instead of an independently refit one.
    print(f"    Running linear probe...")
    lp_results, y_pred = run_linear_probe(X_train, y_train, X_test, y_test,
                                          max_iter=linear_max_iter, C=linear_C,
                                          return_model=True)
    results['linear_probe'] = lp_results
    print(f"    Linear probe: acc={lp_results['accuracy']:.4f}, F1={lp_results['f1_macro']:.4f}")

    # Data efficiency
    if data_efficiency:
        print(f"    Running data-efficiency curve (k-NN k={knn_k} + linear probe)...")
        de_results = run_data_efficiency(X_train, y_train, X_test, y_test,
                                         fractions=de_fractions,
                                         n_repeats=de_repeats,
                                         seed=seed,
                                         knn_k=knn_k,
                                         extra_knn_ks=extra_knn_ks,
                                         linear_max_iter=linear_max_iter,
                                         linear_C=linear_C)
        results['data_efficiency'] = de_results
        for frac, vals in de_results.items():
            lp_a = vals['linear_probe']['accuracy_mean']
            lp_s = vals['linear_probe']['accuracy_std']
            knn_a = vals['knn']['accuracy_mean']
            knn_s = vals['knn']['accuracy_std']
            print(f"      {frac:>4s}: Linear={lp_a*100:5.1f}±{lp_s*100:.1f}%  k-NN={knn_a*100:5.1f}±{knn_s*100:.1f}%  (n={vals['n_train_samples']})")
    
    # Detailed per-class report for the linear probe fitted above. (This used to
    # refit an identical LogisticRegression, doubling the cost of the most
    # expensive step and risking a report that disagreed with the reported
    # metrics if the two fits were ever configured differently.)
    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
    results['classification_report'] = report
    
    return results


def _get_knn_key(results: dict) -> str:
    """Get the primary knn key from results, respecting stored knn_k."""
    knn_results = results.get('knn', {})
    if 'knn_k' in results:
        key = f"knn_k{results['knn_k']}"
        if key in knn_results:
            return key
    # 'knn_k' missing or its k wasn't evaluated (older/malformed result file):
    # fall back to the smallest k actually present, rather than guessing one.
    if knn_results:
        return min(knn_results, key=lambda k: int(k.replace('knn_k', '')))
    # No k-NN results at all; any string is safe here since callers only ever
    # do `.get(key, {})` against this key.
    return 'knn_k'


def print_comparison_table(all_results: dict, label_level: str):
    """Print a paper-ready comparison table."""
    print(f"\n{'='*90}")
    print(f"  CLASSIFICATION BENCHMARK RESULTS ({label_level.upper()} labels)")
    print(f"{'='*90}")
    
    # Determine k from first model
    first_results = list(all_results.values())[0]
    knn_key = _get_knn_key(first_results)
    k_val = knn_key.replace('knn_k', '')
    
    # Header
    header = f"{'Model':<35} {'k-NN(k='+k_val+') Acc':<14} {'k-NN F1':<10} {'LP Acc':<10} {'LP F1':<10}"
    print(header)
    print("-" * 90)
    
    for model_name, results in all_results.items():
        knn_acc = results.get('knn', {}).get(_get_knn_key(results), {}).get('accuracy', -1)
        knn_f1 = results.get('knn', {}).get(_get_knn_key(results), {}).get('f1_macro', -1)
        lp_acc = results.get('linear_probe', {}).get('accuracy', -1)
        lp_f1 = results.get('linear_probe', {}).get('f1_macro', -1)
        
        knn_acc_s = f"{knn_acc*100:.2f}%" if knn_acc >= 0 else "N/A"
        knn_f1_s = f"{knn_f1*100:.2f}%" if knn_f1 >= 0 else "N/A"
        lp_acc_s = f"{lp_acc*100:.2f}%" if lp_acc >= 0 else "N/A"
        lp_f1_s = f"{lp_f1*100:.2f}%" if lp_f1 >= 0 else "N/A"
        
        print(f"{model_name:<35} {knn_acc_s:<10} {knn_f1_s:<10} {lp_acc_s:<10} {lp_f1_s:<10}")
    
    print("-" * 90)
    
    # Data efficiency table if available. The fractions actually present are
    # read from the results rather than assumed, since --de_fractions is
    # reader-chosen and may not match any particular fixed list.
    has_de = any('data_efficiency' in r for r in all_results.values())
    if has_de:
        frac_keys = set()
        for r in all_results.values():
            frac_keys.update(r.get('data_efficiency', {}).keys())
        fracs = sorted(frac_keys, key=lambda s: int(s.replace('%', '')))

        for method, method_label in [('knn', 'k-NN'), ('linear_probe', 'Linear Probe')]:
            for metric, metric_label in [('accuracy_mean', 'Accuracy'), ('f1_macro_mean', 'F1 Macro')]:
                std_key = metric.replace('_mean', '_std')
                print(f"\n  Data Efficiency — {method_label} {metric_label} (%)")
                print(f"  {'Model':<30}", end="")
                for f in fracs:
                    print(f" {f:>10}", end="")
                print()
                print("  " + "-" * 90)
                
                for model_name, results in all_results.items():
                    de = results.get('data_efficiency', {})
                    print(f"  {model_name:<30}", end="")
                    for f in fracs:
                        if f in de and method in de[f]:
                            val = de[f][method][metric]
                            std = de[f][method][std_key]
                            print(f" {val*100:>5.1f}±{std*100:.1f}", end="")
                        else:
                            print(f" {'N/A':>10}", end="")
                    print()
        print()


# ============================================================================
# Figure plotting
# ============================================================================

def plot_results(all_output: dict, output_prefix: str = 'classification_figure',
                 show_std: bool = True):
    """
    Generate figures from benchmark results.

    Produces:
      1. Data-efficiency line plots — one per label level, with k-NN and the
         linear probe side by side
      2. Summary bar chart of k-NN and linear probe at the full training set

    Style: clean, no grid, muted but distinguishable colors, thin lines,
    small markers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    # ── Plot style ────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })
    
    # Color palette — distinct and colorblind-friendly
    COLORS = [
        '#2166AC',  # deep blue
        '#B2182B',  # deep red
        '#1B7837',  # dark green
        '#FF7F00',  # orange
        '#984EA3',  # purple
        '#A65628',  # brown
        '#377EB8',  # lighter blue
        '#E41A1C',  # bright red
    ]

    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    # Alternating thick/thin widths so a thick line is still visible beneath a thin one
    LINEWIDTHS = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    # Clearly distinct dash patterns using explicit on/off tuples
    LINESTYLES = [
        (0, ()),            # solid
        (0, (6, 2)),        # long dash
        (0, (3, 1, 1, 1)),  # dash-dot
        (0, (1, 1)),        # dense dots
        (0, (5, 1, 1, 1, 1, 1)),  # dash-dot-dot
        (0, (4, 2)),        # medium dash
        (0, (2, 1)),        # short dash
        (0, (1, 2)),        # sparse dots
    ]

    # Parse the fraction strings into numeric x-values
    def frac_str_to_num(s):
        return int(s.replace('%', ''))
    
    for label_level, level_results in all_output.items():
        model_names = list(level_results.keys())
        
        # Check if data_efficiency exists
        has_de = any('data_efficiency' in r for r in level_results.values())
        
        if has_de:
            # ── Figure 1: Data Efficiency Curves (2×2: metric × method) ──
            # Top row: Accuracy,  Bottom row: F1 Macro
            # Left col: k-NN,    Right col: Linear Probe
            
            # Determine k from first model
            first_r = level_results[model_names[0]]
            plot_k = first_r.get('knn_k', 5)
            knn_title = f'k-NN (k={plot_k})'
            
            grid = [
                # (row, col, method_key, metric_key, title, ylabel)
                (0, 0, 'knn',          'accuracy_mean', 'accuracy_std', knn_title,       'Accuracy (%)'),
                (0, 1, 'linear_probe', 'accuracy_mean', 'accuracy_std', 'Linear Probe',  'Accuracy (%)'),
                (1, 0, 'knn',          'f1_macro_mean', 'f1_macro_std', knn_title,       'F1 Macro (%)'),
                (1, 1, 'linear_probe', 'f1_macro_mean', 'f1_macro_std', 'Linear Probe',  'F1 Macro (%)'),
            ]
            
            # Adaptive sizing: wider figure + external legend for many models
            n_models = len(model_names)
            fig_w = 6.0 if n_models <= 4 else 6.5
            fig_h = 4.8 if n_models <= 4 else 5.2
            fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
            
            # Adaptive marker size: smaller when many models
            ms = max(3, 6 - n_models * 0.4)
            lw = max(1.0, 1.6 - n_models * 0.08)
            
            # Collect all y-values per panel to set smart y-limits
            panel_y_data = {}
            
            for row, col, method_key, metric_key, std_key, title, ylabel in grid:
                ax = axes[row, col]
                key = (row, col)
                panel_y_data[key] = []
                
                for i, model_name in enumerate(model_names):
                    de = level_results[model_name].get('data_efficiency', {})
                    if not de:
                        continue
                    
                    fracs_str = sorted(de.keys(), key=frac_str_to_num)
                    x_vals = [frac_str_to_num(f) for f in fracs_str]
                    y_vals = [de[f][method_key][metric_key] * 100 for f in fracs_str]
                    y_errs = [de[f][method_key][std_key] * 100 for f in fracs_str]
                    
                    panel_y_data[key].extend(y_vals)
                    
                    color = COLORS[i % len(COLORS)]
                    marker = MARKERS[i % len(MARKERS)]
                    ls = LINESTYLES[i % len(LINESTYLES)]
                    lw_i = LINEWIDTHS[i % len(LINEWIDTHS)]

                    if show_std:
                        ax.errorbar(x_vals, y_vals, yerr=y_errs,
                                   color=color, marker=marker, markersize=ms,
                                   markerfacecolor=color, markeredgecolor='white',
                                   markeredgewidth=0.5, linewidth=lw_i,
                                   linestyle=ls,
                                   capsize=2, capthick=0.5,
                                   label=model_name if row == 0 else None,
                                   zorder=3 + i * 0.1)
                    else:
                        ax.plot(x_vals, y_vals,
                               color=color, marker=marker, markersize=ms,
                               markerfacecolor=color, markeredgecolor='white',
                               markeredgewidth=0.5, linewidth=lw_i,
                               linestyle=ls,
                               label=model_name if row == 0 else None,
                               zorder=3 + i * 0.1)

                # Smart y-axis: zoom to data range
                if panel_y_data[key]:
                    y_min_data = min(panel_y_data[key])
                    y_max_data = max(panel_y_data[key])
                    y_range = y_max_data - y_min_data
                    padding = max(y_range * 0.15, 2.0)
                    ax.set_ylim(max(0, y_min_data - padding), min(100, y_max_data + padding))
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(direction='out')
                ax.set_xlim(-3, 105)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                
                if row == 0:
                    ax.set_title(title, fontweight='semibold')
                if row == 1:
                    ax.set_xlabel('Training Data (%)')
                if col == 0:
                    ax.set_ylabel(ylabel)
            
            # Legend — adaptive placement
            handles, labels = axes[0, 0].get_legend_handles_labels()
            n_legend = len(labels)
            if n_legend <= 3:
                axes[0, 1].legend(handles, labels, loc='lower right',
                                 frameon=False, handletextpad=0.3, borderpad=0.2,
                                 fontsize=7)
            else:
                # External legend below for many models — cleaner
                ncol = min(n_legend, 3)
                fig.legend(handles, labels, loc='lower center',
                          ncol=ncol, frameon=False,
                          bbox_to_anchor=(0.5, -0.06),
                          handletextpad=0.3, columnspacing=1.0,
                          fontsize=6.5)
            
            n_classes = level_results[model_names[0]].get('num_classes', '?')
            fig.suptitle(f'Classification — {label_level.capitalize()} Labels ({n_classes} classes)',
                        fontsize=9, fontweight='semibold', y=1.01)
            
            plt.tight_layout()
            fname = f'{output_prefix}_data_efficiency_{label_level}.png'
            fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"  Saved: {fname}")
        
        # ── Figure 2: Summary bar chart (Accuracy + F1) ─────────────
        fig, axes_bar = plt.subplots(1, 2, figsize=(6.0, 2.5))
        
        n_models = len(model_names)
        x = np.arange(n_models)
        bar_width = 0.35
        
        for ax_idx, (metric_label, metric_key_knn, metric_key_lp) in enumerate([
            ('Accuracy (%)', 'accuracy', 'accuracy'),
            ('F1 Macro (%)', 'f1_macro', 'f1_macro'),
        ]):
            ax = axes_bar[ax_idx]
            knn_vals = []
            lp_vals = []
            for model_name in model_names:
                r = level_results[model_name]
                knn_vals.append(r.get('knn', {}).get(_get_knn_key(r), {}).get(metric_key_knn, 0) * 100)
                lp_vals.append(r.get('linear_probe', {}).get(metric_key_lp, 0) * 100)
            
            # Label with the k actually evaluated; a fixed "k=5" here mislabels
            # every figure produced with a different --knn_k.
            knn_k_used = level_results[model_names[0]].get('knn_k', '?') if model_names else '?'
            bars1 = ax.bar(x - bar_width / 2, knn_vals, bar_width,
                           color=COLORS[0], edgecolor='white', linewidth=0.5,
                           label=f'k-NN (k={knn_k_used})' if ax_idx == 0 else None,
                           zorder=3, alpha=0.85)
            bars2 = ax.bar(x + bar_width / 2, lp_vals, bar_width,
                           color=COLORS[1], edgecolor='white', linewidth=0.5,
                           label='Linear Probe' if ax_idx == 0 else None, zorder=3, alpha=0.85)
            
            for bar in bars1:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                       f'{h:.1f}', ha='center', va='bottom', fontsize=5.5, color=COLORS[0])
            for bar in bars2:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                       f'{h:.1f}', ha='center', va='bottom', fontsize=5.5, color=COLORS[1])
            
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=6)
            ax.set_ylabel(metric_label)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='out')
            
            y_min = max(0, min(knn_vals + lp_vals) - 10)
            ax.set_ylim(y_min, 105)
        
        axes_bar[0].legend(frameon=False, fontsize=6.5, loc='lower right')
        
        n_classes = level_results[model_names[0]].get('num_classes', '?')
        fig.suptitle(f'{label_level.capitalize()} Labels ({n_classes} classes)',
                    fontsize=9, fontweight='semibold', y=1.02)
        
        plt.tight_layout()
        fname = f'{output_prefix}_summary_{label_level}.png'
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {fname}")
    
    # ── Figure 3: Combined multi-level comparison (if both coarse & fine) ──
    if len(all_output) >= 2 and any(
        'data_efficiency' in r for lvl in all_output.values() for r in lvl.values()
    ):
        levels = list(all_output.keys())
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0), sharey='row')
        
        for row, label_level in enumerate(levels[:2]):
            level_results = all_output[label_level]
            model_names = list(level_results.keys())
            
            for col, (method_key, method_title) in enumerate([
                ('knn', 'k-NN'), ('linear_probe', 'Linear Probe')
            ]):
                ax = axes[row, col]
                
                for i, model_name in enumerate(model_names):
                    de = level_results[model_name].get('data_efficiency', {})
                    if not de:
                        continue
                    
                    fracs_str = sorted(de.keys(), key=frac_str_to_num)
                    x_vals = [frac_str_to_num(f) for f in fracs_str]
                    y_vals = [de[f][method_key]['accuracy_mean'] * 100 for f in fracs_str]
                    y_errs = [de[f][method_key]['accuracy_std'] * 100 for f in fracs_str]
                    
                    color = COLORS[i % len(COLORS)]
                    marker = MARKERS[i % len(MARKERS)]
                    ls = LINESTYLES[i % len(LINESTYLES)]
                    lw_i = LINEWIDTHS[i % len(LINEWIDTHS)]

                    ax.errorbar(x_vals, y_vals, yerr=y_errs,
                               color=color, marker=marker, markersize=4,
                               markerfacecolor=color, markeredgecolor='white',
                               markeredgewidth=0.5, linewidth=lw_i,
                               linestyle=ls,
                               capsize=2, capthick=0.6,
                               label=model_name if row == 0 else None,
                               zorder=3 + i * 0.1)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(direction='out')
                ax.set_xlim(-3, 105)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                
                if row == 1:
                    ax.set_xlabel('Training Data (%)')
                if col == 0:
                    n_classes = level_results[model_names[0]].get('num_classes', '?')
                    ax.set_ylabel(f'{label_level.capitalize()}\n({n_classes} classes)\nAccuracy (%)')
                if row == 0:
                    ax.set_title(method_title, fontweight='semibold')
        
        # Single legend at top
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                  ncol=min(len(labels), 4), frameon=False,
                  bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
                  fontsize=7)
        
        fig.suptitle('Classification Data Efficiency', fontsize=10,
                    fontweight='semibold', y=1.01)
        
        plt.tight_layout()
        fname = f'{output_prefix}_combined.png'
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_knn_k_lineplot(all_output: dict, knn_k: int, output_prefix: str,
                        show_std: bool = True):
    """
    Data-efficiency figure identical in layout to plot_results() but using
    knn_k{knn_k} results stored per fraction in data_efficiency.

    Requires the JSON to have been computed with --extra_de_knn_ks {knn_k}
    (or with --knn_k {knn_k} as the primary k).

    Falls back to the primary 'knn' key if knn_k{knn_k} is not found.
    Saved as {output_prefix}_knn_k{knn_k}_lineplot.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })

    COLORS = [
        '#2166AC', '#B2182B', '#1B7837', '#FF7F00',
        '#984EA3', '#A65628', '#377EB8', '#E41A1C',
    ]
    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    LINEWIDTHS = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    LINESTYLES = [
        (0, (6, 2)),            # long dash
        (0, (3, 2)),            # medium dash
        (0, (1, 1)),            # dense dots
        (0, (3, 2, 1, 2)),      # dash-dot
        (0, (6, 2, 1, 2)),      # long dash-dot
        (0, (3, 2, 1, 2, 1, 2)),# dash-dot-dot
        (0, (2, 1)),            # short dash
        (0, (1, 2)),            # sparse dots
    ]

    def frac_str_to_num(s):
        return int(s.replace('%', ''))

    for label_level, level_results in all_output.items():
        model_names = list(level_results.keys())
        has_de = any('data_efficiency' in r for r in level_results.values())
        if not has_de:
            print(f"  WARNING: no data_efficiency in {label_level} results, skipping knn_k_lineplot")
            continue

        # Determine which data_efficiency key to use for knn:
        #   1. explicit extra key  knn_k{k}  (stored by --extra_de_knn_ks)
        #   2. primary 'knn' key, if it was computed with the same k
        #   3. warn and skip
        first_r = list(level_results.values())[0]
        first_de = first_r.get('data_efficiency', {})
        first_frac = list(first_de.keys())[0] if first_de else None
        extra_key = f'knn_k{knn_k}'

        if first_frac and extra_key in first_de.get(first_frac, {}):
            de_knn_key = extra_key
        elif first_r.get('knn_k') == knn_k:
            # Primary k matches — the 'knn' entry in data_efficiency is already k=knn_k
            de_knn_key = 'knn'
        else:
            stored_k = first_r.get('knn_k', '?')
            avail = [k for k in first_de.get(first_frac, {}) if k.startswith('knn')]
            print(f"  WARNING: k={knn_k} not found in data_efficiency for {label_level}.")
            print(f"  Primary k stored in JSON is k={stored_k}. Available keys: {avail}.")
            print(f"  Options:")
            print(f"    - Re-run computation with --knn_k {knn_k}  (makes it the primary k)")
            print(f"    - Re-run computation with --knn_k {stored_k} --extra_de_knn_ks {knn_k}")
            print(f"  Skipping {label_level}.")
            continue

        knn_title = f'k-NN (k={knn_k})'
        grid = [
            (0, 0, de_knn_key,     'accuracy_mean', 'accuracy_std', knn_title,      'Accuracy (%)'),
            (0, 1, 'linear_probe', 'accuracy_mean', 'accuracy_std', 'Linear Probe', 'Accuracy (%)'),
            (1, 0, de_knn_key,     'f1_macro_mean', 'f1_macro_std', knn_title,      'F1 Macro (%)'),
            (1, 1, 'linear_probe', 'f1_macro_mean', 'f1_macro_std', 'Linear Probe', 'F1 Macro (%)'),
        ]

        n_models = len(model_names)
        fig_w = 6.0 if n_models <= 4 else 6.5
        fig_h = 4.8 if n_models <= 4 else 5.2
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

        ms = max(3, 6 - n_models * 0.4)
        lw = max(1.0, 1.6 - n_models * 0.08)
        panel_y_data = {}

        for row, col, method_key, metric_key, std_key, title, ylabel in grid:
            ax = axes[row, col]
            key = (row, col)
            panel_y_data[key] = []

            for i, model_name in enumerate(model_names):
                de = level_results[model_name].get('data_efficiency', {})
                if not de:
                    continue

                fracs_str = sorted(de.keys(), key=frac_str_to_num)
                x_vals, y_vals, y_errs = [], [], []
                for f in fracs_str:
                    entry = de[f].get(method_key, {})
                    if metric_key not in entry:
                        continue
                    x_vals.append(frac_str_to_num(f))
                    y_vals.append(entry[metric_key] * 100)
                    y_errs.append(entry.get(std_key, 0) * 100)

                if not x_vals:
                    continue

                panel_y_data[key].extend(y_vals)
                color = COLORS[i % len(COLORS)]
                marker = MARKERS[i % len(MARKERS)]
                ls = LINESTYLES[i % len(LINESTYLES)]
                lw_i = LINEWIDTHS[i % len(LINEWIDTHS)]

                if show_std:
                    ax.errorbar(x_vals, y_vals, yerr=y_errs,
                               color=color, marker=marker, markersize=ms,
                               markerfacecolor=color, markeredgecolor='white',
                               markeredgewidth=0.5, linewidth=lw_i,
                               linestyle=ls,
                               capsize=2, capthick=0.5,
                               label=model_name if row == 0 else None,
                               zorder=3 + i * 0.1)
                else:
                    ax.plot(x_vals, y_vals,
                           color=color, marker=marker, markersize=ms,
                           markerfacecolor=color, markeredgecolor='white',
                           markeredgewidth=0.5, linewidth=lw_i,
                           linestyle=ls,
                           label=model_name if row == 0 else None,
                           zorder=3 + i * 0.1)

            if panel_y_data[key]:
                y_min_data = min(panel_y_data[key])
                y_max_data = max(panel_y_data[key])
                y_range = y_max_data - y_min_data
                padding = max(y_range * 0.15, 2.0)
                ax.set_ylim(max(0, y_min_data - padding), min(100, y_max_data + padding))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='out')
            ax.set_xlim(-3, 105)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

            if row == 0:
                ax.set_title(title, fontweight='semibold')
            if row == 1:
                ax.set_xlabel('Training Data (%)')
            if col == 0:
                ax.set_ylabel(ylabel)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        n_legend = len(labels)
        if n_legend <= 3:
            axes[0, 1].legend(handles, labels, loc='lower right',
                             frameon=False, handletextpad=0.3, borderpad=0.2,
                             fontsize=7)
        else:
            ncol = min(n_legend, 3)
            fig.legend(handles, labels, loc='lower center',
                      ncol=ncol, frameon=False,
                      bbox_to_anchor=(0.5, -0.06),
                      handletextpad=0.3, columnspacing=1.0,
                      fontsize=6.5)

        n_classes = level_results[model_names[0]].get('num_classes', '?')
        fig.suptitle(f'Classification — {label_level.capitalize()} Labels ({n_classes} classes)',
                    fontsize=9, fontweight='semibold', y=1.01)

        plt.tight_layout()
        fname = f'{output_prefix}_knn_k{knn_k}_lineplot_{label_level}.png'
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="k-NN and linear probe classification benchmark for UltraVEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate pretrained embeddings with coarse labels. Every numeric flag below
  # is required -- there is no built-in default for any of them:
  python classification_benchmark.py \\
      --embeddings embeddings.npz \\
      --metadata metadata/dataset_metadata.csv \\
      --model_name "<model name>" \\
      --knn_k <k> --knn_k_values <k1> <k2> <k3> \\
      --test_fraction <fraction> --min_samples <n> --seed <seed> \\
      --linear_max_iter <max_iter> --linear_C <C>

  # Compare pretrained vs random init, both label levels:
  python classification_benchmark.py \\
      --embeddings pretrained_emb.npz random_emb.npz \\
      --model_names "<model A>" "<model B>" \\
      --metadata metadata/dataset_metadata.csv \\
      --label_levels coarse fine \\
      --knn_k <k> --knn_k_values <k1> <k2> <k3> \\
      --test_fraction <fraction> --min_samples <n> --seed <seed> \\
      --linear_max_iter <max_iter> --linear_C <C> \\
      --data_efficiency --de_fractions <f1> <f2> ... --de_repeats <n>

  # ===== PLOT FROM PRE-COMPUTED JSON (no recalculation) =====
  # from_json mode needs none of the numeric flags above -- it only reads a
  # JSON this script already saved.
  # Single JSON:
  python classification_benchmark.py \\
      --from_json results_pretrained.json \\
      --output figures_out

  # Multiple JSONs (merged into one plot):
  python classification_benchmark.py \\
      --from_json results_a.json results_b.json \\
      --from_json_names "Random init" "Pretrained" \\
      --output figures_out
        """
    )
    
    # ── Plot-from-JSON mode ──────────────────────────────────────
    parser.add_argument('--from_json', type=str, nargs='+', default=None,
                        help='Path(s) to pre-computed result JSON file(s). '
                             'Skips all computation and just generates figures. '
                             'Each JSON should be a file previously saved by this script.')
    parser.add_argument('--from_json_names', type=str, nargs='+', default=None,
                        help='Display name(s) for each JSON file (overrides names in the JSON). '
                             'Must match the number of --from_json files.')
    parser.add_argument('--select_models', type=str, nargs='+', default=None,
                        help='When a single JSON contains multiple models, select which to plot. '
                             'Use substrings to match, e.g. --select_models "epoch0" "epoch1000". '
                             'If not set, all models in the JSON are plotted.')
    parser.add_argument('--rename_models', type=str, nargs='+', default=None,
                        help='Rename selected models for display. Must match --select_models count. '
                             'E.g.: --rename_models "Random init" "Pretrained"')
    parser.add_argument('--no_std', action='store_true',
                        help='Do not show error bars/std in data efficiency plots (cleaner for many models)')
    parser.add_argument('--knn_k_lineplot', type=int, default=None,
                        help='Generate a standalone line plot of k-NN accuracy and F1 across epochs '
                             'for this k value (e.g. --knn_k_lineplot 5). '
                             'The k must be one of the k values already in the JSON (e.g. 1,3,5,10,20). '
                             'Saved as {output}_knn_k{K}_lineplot.png')
    
    # ── Compute mode ─────────────────────────────────────────────
    parser.add_argument('--embeddings', type=str, nargs='+', default=None,
                        help='Path(s) to embeddings.npz file(s)')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Name(s) for each embeddings file (for display)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Single model name (shortcut for --model_names with one file)')
    parser.add_argument('--metadata', type=str, required=False, default=None,
                        help='Dataset annotation table (CSV/TSV/XLSX) mapping each source '
                             'dataset to organism / tissue / modality. Required in compute mode. '
                             'See metadata/dataset_metadata.csv for the expected schema.')
    parser.add_argument('--label_levels', type=str, nargs='+', default=['coarse'],
                        choices=list(VALID_LABEL_LEVELS),
                        help='Label granularity to evaluate. The number of classes at each '
                             'level depends on your metadata table: coarse = organism x coarse '
                             'tissue, fine = organism x fine tissue, organism = organism only, '
                             'modality = imaging modality. '
                             'Use multiple: --label_levels coarse organism modality')
    # Every flag below that takes a number has NO built-in default and is
    # required in compute mode (validated after parsing, since these are
    # irrelevant in --from_json mode). This is deliberate: a benchmark number
    # that changes with an undocumented hidden default is not reproducible, so
    # every hyperparameter that can move the reported accuracy must be a value
    # the reader chose and can see on the command line.
    parser.add_argument('--knn_k', type=int, default=None,
                        help='[REQUIRED in compute mode] Primary k for k-NN in data-efficiency '
                             'curves and summary tables. Use --knn_k 1 for TOP1 k-NN (as in Cryo-IEF).')
    parser.add_argument('--knn_k_values', type=int, nargs='+', default=None,
                        help='[REQUIRED in compute mode] k values to evaluate in the main k-NN '
                             'sweep, e.g. --knn_k_values 1 5 10 20. --knn_k is always included '
                             'even if omitted from this list.')
    parser.add_argument('--extra_de_knn_ks', type=int, nargs='+', default=None,
                        help='Additional k values to evaluate per fraction in data-efficiency curves '
                             '(stored as knn_k{k} in the JSON), e.g. --extra_de_knn_ks 5 10. '
                             'Optional; use with --knn_k_lineplot to plot those curves later.')
    parser.add_argument('--test_fraction', type=float, default=None,
                        help='[REQUIRED in compute mode] Fraction of samples held out for the test '
                             'split, e.g. --test_fraction 0.2.')
    parser.add_argument('--min_samples', type=int, default=None,
                        help='[REQUIRED in compute mode] Minimum samples a class must have to be '
                             'included in the benchmark.')
    parser.add_argument('--data_efficiency', action='store_true',
                        help='Run data-efficiency curves over --de_fractions, repeated '
                             '--de_repeats times per fraction for error bars (both become required '
                             'when this flag is set).')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='L2-normalize embeddings before classification (default: True). '
                             'This removes magnitude shortcuts that favor random init.')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable L2-normalization (not recommended)')
    parser.add_argument('--de_fractions', type=float, nargs='+', default=None,
                        help='[REQUIRED with --data_efficiency] Training-set fractions for the '
                             'data-efficiency curve, e.g. --de_fractions 0.01 0.1 0.5 1.0.')
    parser.add_argument('--de_repeats', type=int, default=None,
                        help='[REQUIRED with --data_efficiency] Subsampling repeats per fraction, '
                             'for error bars. At 100%% only one run is done regardless of this value.')
    parser.add_argument('--linear_max_iter', type=int, default=None,
                        help='[REQUIRED in compute mode] max_iter for the linear-probe '
                             'LogisticRegression solver.')
    parser.add_argument('--linear_C', type=float, default=None,
                        help='[REQUIRED in compute mode] Inverse L2 regularization strength for '
                             'the linear probe.')
    parser.add_argument('--group_split', action='store_true',
                        help='Hold out whole source datasets instead of splitting individual '
                             'volumes. Stricter: no source dataset appears in both train and '
                             'test, so the score measures generalization to unseen datasets '
                             'rather than to unseen crops of seen datasets. Requires each class '
                             'to be covered by at least two source datasets.')
    parser.add_argument('--seed', type=int, default=None,
                        help='[REQUIRED in compute mode] Random seed for the train/test split and '
                             'all data-efficiency subsampling.')
    parser.add_argument('--output', type=str, default='classification_results.json',
                        help='Output JSON file (compute mode) or output prefix for figures (from_json mode)')

    args = parser.parse_args()
    
    # ══════════════════════════════════════════════════════════════
    # MODE 1: Plot from pre-computed JSON files
    # ══════════════════════════════════════════════════════════════
    if args.from_json is not None:
        print("=" * 60)
        print("  PLOT-FROM-JSON MODE")
        print("=" * 60)
        
        all_output = _load_and_merge_jsons(args.from_json, args.from_json_names)
        
        # Filter models if --select_models is given
        if args.select_models is not None:
            for label_level in list(all_output.keys()):
                level_data = all_output[label_level]
                selected = {}
                for sel_pattern in args.select_models:
                    for model_name, results in level_data.items():
                        if sel_pattern.lower() in model_name.lower():
                            selected[model_name] = results
                            break
                if not selected:
                    print(f"  WARNING: No models matched --select_models {args.select_models}")
                    print(f"  Available: {list(level_data.keys())}")
                    selected = level_data
                all_output[label_level] = selected
        
        # Rename models if --rename_models is given
        if args.rename_models is not None:
            for label_level in list(all_output.keys()):
                level_data = all_output[label_level]
                old_names = list(level_data.keys())
                if len(args.rename_models) == len(old_names):
                    renamed = {}
                    for old, new in zip(old_names, args.rename_models):
                        renamed[new] = level_data[old]
                    all_output[label_level] = renamed
                else:
                    print(f"  WARNING: --rename_models count ({len(args.rename_models)}) "
                          f"!= model count ({len(old_names)}). Skipping rename.")
        
        # Print tables
        for label_level, level_results in all_output.items():
            print_comparison_table(level_results, label_level)
        
        # Generate figures
        fig_prefix = args.output.replace('.json', '').replace('.pdf', '').replace('.png', '')
        print(f"\nGenerating figures with prefix: {fig_prefix}")
        plot_results(all_output, output_prefix=fig_prefix, show_std=not args.no_std)
        if args.knn_k_lineplot is not None:
            plot_knn_k_lineplot(all_output, args.knn_k_lineplot, fig_prefix,
                                show_std=not args.no_std)

        # Print summary
        print(f"\n{'='*80}")
        print("PAPER-READY SUMMARY")
        print(f"{'='*80}")
        for level, level_results in all_output.items():
            n_cls = level_results[list(level_results.keys())[0]].get('num_classes', '?')
            print(f"\n[{level.upper()} labels] ({n_cls} classes)")
            for model_name, res in level_results.items():
                knn_acc = res.get('knn', {}).get(_get_knn_key(res), {}).get('accuracy', 0) * 100
                lp = res.get('linear_probe', {}).get('accuracy', 0) * 100
                k_used = res.get('knn_k', '?')
                print(f"  {model_name}: k-NN(k={k_used})={knn_acc:.1f}%, Linear={lp:.1f}%")
        
        return
    
    # ══════════════════════════════════════════════════════════════
    # MODE 2: Full computation from embeddings
    # ══════════════════════════════════════════════════════════════
    if args.embeddings is None:
        parser.error("Either --from_json or --embeddings is required.")
    if args.metadata is None:
        if not DEFAULT_METADATA_CSV.is_file():
            parser.error("--metadata is required when computing from embeddings "
                         f"(no default table at {DEFAULT_METADATA_CSV}).")
        args.metadata = str(DEFAULT_METADATA_CSV)

    # No numeric hyperparameter has a built-in default (see the argparse
    # section above), so every one of them must be checked here before any
    # computation runs. This is what actually enforces "[REQUIRED in compute
    # mode]": argparse itself can't express "required only in this mode",
    # since --from_json mode needs none of these.
    required_in_compute_mode = {
        '--knn_k': args.knn_k,
        '--knn_k_values': args.knn_k_values,
        '--test_fraction': args.test_fraction,
        '--min_samples': args.min_samples,
        '--seed': args.seed,
        '--linear_max_iter': args.linear_max_iter,
        '--linear_C': args.linear_C,
    }
    missing = [name for name, val in required_in_compute_mode.items() if val is None]
    if missing:
        parser.error(
            "The following flags have no built-in default and must be set explicitly "
            f"when computing from embeddings: {', '.join(missing)}"
        )
    if args.data_efficiency and (args.de_fractions is None or args.de_repeats is None):
        parser.error("--data_efficiency requires --de_fractions and --de_repeats to be set.")

    # Resolve model names
    if args.model_name and not args.model_names:
        model_names = [args.model_name]
    elif args.model_names:
        model_names = args.model_names
    else:
        model_names = [Path(p).stem for p in args.embeddings]
    
    assert len(model_names) == len(args.embeddings), \
        f"Number of model names ({len(model_names)}) must match embeddings files ({len(args.embeddings)})"
    
    # Load metadata
    print("Loading metadata...")
    metadata_rows = load_metadata(args.metadata)
    label_map = build_dataset_label_map(metadata_rows)
    fuzzy_lookup = build_fuzzy_lookup(label_map)
    print(f"Metadata: {len(label_map)} datasets loaded from {args.metadata}")
    
    # Run for each label level
    all_output = {}
    
    for label_level in args.label_levels:
        print(f"\n{'#'*80}")
        print(f"# Evaluating with {label_level.upper()} labels")
        print(f"{'#'*80}")
        
        level_results = {}
        
        for emb_path, model_name in zip(args.embeddings, model_names):
            print(f"\n--- {model_name} ---")
            print(f"Loading: {emb_path}")
            
            data = np.load(emb_path, allow_pickle=True)
            embeddings = data['embeddings']
            labels = data['labels']
            names = data.get('names', labels)  # fallback
            
            print(f"  Embeddings shape: {embeddings.shape}")
            print(f"  Unique raw labels: {len(np.unique(labels))}")
            
            do_normalize = args.normalize and not args.no_normalize
            
            results = evaluate_embeddings(
                embeddings=embeddings,
                labels_raw=labels,
                names=names,
                label_map=label_map,
                fuzzy_lookup=fuzzy_lookup,
                label_level=label_level,
                test_fraction=args.test_fraction,
                seed=args.seed,
                data_efficiency=args.data_efficiency,
                min_samples_per_class=args.min_samples,
                normalize=do_normalize,
                knn_k=args.knn_k,
                knn_k_values=args.knn_k_values,
                extra_knn_ks=args.extra_de_knn_ks,
                de_fractions=args.de_fractions,
                de_repeats=args.de_repeats,
                linear_max_iter=args.linear_max_iter,
                linear_C=args.linear_C,
                group_split=args.group_split,
            )
            
            level_results[model_name] = results
        
        # Print comparison table
        print_comparison_table(level_results, label_level)
        all_output[label_level] = level_results
    
    # Save results
    output_path = args.output
    
    # Make JSON-serializable (convert numpy types)
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(make_serializable(all_output), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Generate figures
    print(f"\nGenerating figures...")
    fig_prefix = output_path.replace('.json', '')
    plot_results(all_output, output_prefix=fig_prefix, show_std=not args.no_std)
    if args.knn_k_lineplot is not None:
        plot_knn_k_lineplot(all_output, args.knn_k_lineplot, fig_prefix,
                            show_std=not args.no_std)

    # Print quick summary for paper
    print(f"\n{'='*80}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*80}")
    for level, level_results in all_output.items():
        print(f"\n[{level.upper()} labels] ({level_results[list(level_results.keys())[0]].get('num_classes', '?')} classes)")
        for model_name, res in level_results.items():
            knn_acc = res.get('knn', {}).get(_get_knn_key(res), {}).get('accuracy', 0) * 100
            lp = res.get('linear_probe', {}).get('accuracy', 0) * 100
            k_used = res.get('knn_k', '?')
            print(f"  {model_name}: k-NN(k={k_used})={knn_acc:.1f}%, Linear={lp:.1f}%")


def _load_and_merge_jsons(json_paths: list, display_names: list = None) -> dict:
    """
    Load one or more pre-computed result JSON files and merge them into
    a single all_output dict suitable for plot_results().
    
    Each JSON file has structure:
        { "coarse": { "ModelName": { results... } }, "fine": { ... } }
    
    When multiple JSONs are provided, models from all files are merged
    under the same label levels. If display_names are provided, they
    override the model names found in each JSON (one name per file).
    
    This handles two common patterns:
      (a) Each JSON has one model → merge side by side for comparison
      (b) One JSON has multiple models → just load and plot
    """
    if display_names is not None:
        assert len(display_names) == len(json_paths), \
            f"--from_json_names ({len(display_names)}) must match --from_json ({len(json_paths)})"
    
    merged = {}  # label_level -> { model_name -> results }
    
    for file_idx, json_path in enumerate(json_paths):
        print(f"Loading: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Detect format: the JSON could be:
        #   (A) { "coarse": { "model": {...} }, "fine": {...} }  — standard output
        #   (B) { "model": {...} }  — single-level, if user saved only one level
        #   (C) direct results dict (no label level wrapper)
        
        # Check if top-level keys look like label levels
        top_keys = list(data.keys())
        is_level_keyed = any(k in VALID_LABEL_LEVELS for k in top_keys)

        if is_level_keyed:
            # Standard format (A)
            for label_level, level_data in data.items():
                if label_level not in VALID_LABEL_LEVELS:
                    continue
                if label_level not in merged:
                    merged[label_level] = {}
                
                for orig_model_name, results in level_data.items():
                    # Determine display name
                    if display_names is not None:
                        # One name per file: if this file has multiple models,
                        # append original name to distinguish
                        if len(level_data) == 1:
                            name = display_names[file_idx]
                        else:
                            name = f"{display_names[file_idx]} ({orig_model_name})"
                    else:
                        name = orig_model_name
                    
                    # Handle duplicate names by appending suffix
                    if name in merged[label_level]:
                        suffix = 2
                        while f"{name} ({suffix})" in merged[label_level]:
                            suffix += 1
                        name = f"{name} ({suffix})"
                    
                    merged[label_level][name] = results
        else:
            # Format (B) or (C): a single label level, not named by a top-level key.
            # Try to detect if it's model-keyed or results directly.
            sample_val = data[top_keys[0]]
            if isinstance(sample_val, dict) and any(
                k in sample_val for k in ['knn', 'linear_probe', 'num_classes']
            ):
                # It's model_name -> results. Recover the real label level from
                # the result dict itself (evaluate_embeddings always stores one)
                # instead of guessing -- a JSON saved with --label_levels fine
                # would otherwise get silently mislabelled as coarse.
                label_level = sample_val.get('label_level', 'coarse')
                if label_level not in merged:
                    merged[label_level] = {}
                
                for orig_model_name, results in data.items():
                    name = display_names[file_idx] if display_names and len(data) == 1 else orig_model_name
                    merged[label_level][name] = results
            else:
                print(f"  WARNING: Could not determine format of {json_path}, skipping.")
    
    # Report
    for level, models in merged.items():
        print(f"\n  [{level}] Models loaded: {list(models.keys())}")
    
    return merged


if __name__ == '__main__':
    main()
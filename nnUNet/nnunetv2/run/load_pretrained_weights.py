"""
Adaptive Pretrained Weight Loader for nnUNet (torch.compile compatible)
========================================================================

Drop-in replacement for nnunetv2/run/load_pretrained_weights.py

Handles:
  - torch.compile '_orig_mod.' prefix on target model keys
  - DDP 'module.' prefix on pretrained keys  
  - Conv kernel [3,3,3] → [1,3,3] via center Z-slice adaptation
  - Skips segmentation layers

Installation:
  cp load_pretrained_weights.py /path/to/nnUNet/nnunetv2/run/load_pretrained_weights.py
"""

import torch
from torch import nn


def load_pretrained_weights(network: nn.Module,
                            fname: str,
                            verbose: bool = False):
    """
    Load pretrained weights with adaptive kernel shape handling.
    Handles torch.compile prefix mismatches automatically.
    """
    saved_model = torch.load(fname, map_location=torch.device('cpu'))

    # ---- Extract pretrained state dict ----
    if 'network_weights' in saved_model:
        pretrained_dict = saved_model['network_weights']
    elif 'state_dict' in saved_model:
        pretrained_dict = saved_model['state_dict']
    elif 'model' in saved_model:
        pretrained_dict = saved_model['model']
    else:
        pretrained_dict = saved_model

    # ---- Strip prefixes from pretrained keys ----
    def strip_prefix(key):
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('_orig_mod.'):
            key = key[len('_orig_mod.'):]
        return key

    pretrained_clean = {strip_prefix(k): v for k, v in pretrained_dict.items()}

    # ---- Get target model state dict ----
    model_dict = network.state_dict()

    # ---- Build mapping: target key → stripped base key ----
    # The target model may have _orig_mod. prefix from torch.compile
    target_to_base = {}
    for k in model_dict.keys():
        base = strip_prefix(k)
        target_to_base[k] = base

    # ---- Keys to skip ----
    skip_strings = ['.seg_layers.', 'main_final', 'aux_final']

    # ---- Match and transfer ----
    loaded_exact = []
    loaded_adapted = []
    skipped_seg = []
    skipped_missing = []
    skipped_shape = []

    transfer_dict = {}

    for target_key, base_key in target_to_base.items():
        # Skip segmentation layers
        if any(s in base_key for s in skip_strings):
            skipped_seg.append(target_key)
            continue

        if base_key not in pretrained_clean:
            skipped_missing.append((target_key, base_key))
            continue

        pretrained_val = pretrained_clean[base_key]
        target_shape = model_dict[target_key].shape

        if pretrained_val.shape == target_shape:
            # Exact match
            transfer_dict[target_key] = pretrained_val
            loaded_exact.append((target_key, base_key))

        elif (pretrained_val.dim() == 5 and
              len(target_shape) == 5 and
              target_shape[0] == pretrained_val.shape[0] and
              target_shape[1] == pretrained_val.shape[1] and
              target_shape[3] == pretrained_val.shape[3] and
              target_shape[4] == pretrained_val.shape[4] and
              pretrained_val.shape[2] == 3 and
              target_shape[2] == 1):
            # Conv kernel [C_out, C_in, 3, 3, 3] → [C_out, C_in, 1, 3, 3]
            adapted = pretrained_val[:, :, 1:2, :, :]
            transfer_dict[target_key] = adapted
            loaded_adapted.append((target_key, base_key))

        else:
            skipped_shape.append((target_key, base_key,
                                  pretrained_val.shape, target_shape))

    # ---- Apply ----
    model_dict.update(transfer_dict)
    network.load_state_dict(model_dict)

    # ---- Report ----
    if verbose:
        print("\n" + "=" * 70)
        print("Adaptive pretrained weight loading")
        print("=" * 70)

        # Detect prefix situation
        sample_target = list(model_dict.keys())[0] if model_dict else ""
        sample_base = target_to_base.get(sample_target, "")
        if sample_target != sample_base:
            print(f"\n  Key prefix detected:")
            print(f"    Target model: '{sample_target[:30]}...'")
            print(f"    Base (stripped): '{sample_base[:30]}...'")

        print(f"\n  Loaded (exact match):     {len(loaded_exact)}")
        for tk, bk in loaded_exact[:3]:
            print(f"    {bk}: {tuple(pretrained_clean[bk].shape)}")
        if len(loaded_exact) > 3:
            print(f"    ... and {len(loaded_exact) - 3} more")

        print(f"\n  Loaded (kernel adapted):  {len(loaded_adapted)}")
        print(f"    Strategy: [3,3,3] → [1,3,3] center Z-slice")
        for tk, bk in loaded_adapted:
            ps = tuple(pretrained_clean[bk].shape)
            ts = tuple(model_dict[tk].shape)
            print(f"    {bk}: {ps} → {ts}")

        print(f"\n  Skipped (seg layers):     {len(skipped_seg)}")
        print(f"  Skipped (missing):        {len(skipped_missing)}")
        if skipped_missing:
            # Show first few to help debug
            for tk, bk in skipped_missing[:5]:
                print(f"    target='{tk}' base='{bk}'")
            if len(skipped_missing) > 5:
                print(f"    ... and {len(skipped_missing) - 5} more")

        if skipped_shape:
            print(f"\n  Skipped (shape mismatch): {len(skipped_shape)}")
            for tk, bk, ps, ts in skipped_shape:
                print(f"    {bk}: pretrained={ps} vs target={ts}")

        total = len(loaded_exact) + len(loaded_adapted)
        total_params = len(model_dict)
        pct = 100 * total / total_params if total_params > 0 else 0
        print(f"\n  TOTAL: {total}/{total_params} parameters loaded ({pct:.1f}%)")
        print("=" * 70 + "\n")
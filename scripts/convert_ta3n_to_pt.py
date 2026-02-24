"""
Convert TA3N video features (.t7 per-frame files + .txt list files) to DM-POT .pt format.

This script reads the HMDB_UCF feature files used by casual_ot and converts them
into the {samples, labels} .pt format expected by DM-POT.

Usage:
    python scripts/convert_ta3n_to_pt.py \
        --list_dir  /path/to/video_dataset/all_list \
        --feat_root /path/to/video_dataset \
        --output_dir /path/to/DM-POT/Datasets/HMDB_UCF_small \
        --variant small \
        --num_segments 16

The output will be:
    <output_dir>/train_ucf101.pt
    <output_dir>/test_ucf101.pt
    <output_dir>/train_hmdb51.pt
    <output_dir>/test_hmdb51.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm


def sample_indices(num_frames, num_segments):
    """
    Uniformly sample `num_segments` frame indices from a video with `num_frames` frames.
    Returns 1-indexed indices (matching TA3N img_XXXXX.t7 naming).
    """
    if num_frames > num_segments:
        average_duration = (num_frames - 1) // num_segments
        offsets = np.multiply(list(range(num_segments)), average_duration) + \
                  np.random.randint(average_duration, size=num_segments)
    elif num_frames == num_segments:
        offsets = np.arange(num_segments)
    else:
        # Repeat frames if video is shorter than num_segments
        offsets = np.sort(np.random.choice(num_frames, num_segments, replace=True))
    return offsets + 1  # 1-indexed


def load_video_features(feature_path, num_frames, num_segments):
    """
    Load and stack frame-level .t7 features for a single video.
    Returns tensor of shape (feature_dim, num_segments).
    """
    indices = sample_indices(num_frames, num_segments)
    frames = []
    for idx in indices:
        p = min(int(idx), num_frames)
        t7_path = os.path.join(feature_path, f'img_{p:05d}.t7')
        try:
            feat = torch.load(t7_path, weights_only=False)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            frames.append(feat.float())
        except FileNotFoundError:
            # Fallback to first frame
            fallback = os.path.join(feature_path, 'img_00001.t7')
            feat = torch.load(fallback, weights_only=False)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            frames.append(feat.float())

    # Stack: (num_segments, feat_dim) -> (feat_dim, num_segments)
    return torch.stack(frames).permute(1, 0)


def parse_list_file(list_dir, list_file, feat_root):
    """
    Parse a TA3N list file. Each line: <path> <num_frames> <label>
    Returns list of (feature_path, num_frames, label).
    """
    entries = []
    filepath = os.path.join(list_dir, list_file)
    if not os.path.exists(filepath):
        print(f"Warning: List file not found: {filepath}")
        return entries

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 3:
                continue
            rel_path = parts[0].replace('dataset/', '')
            feature_path = os.path.join(feat_root, rel_path)
            num_frames = int(parts[1])
            label = int(parts[2])
            entries.append((feature_path, num_frames, label))
    return entries


def convert_split(entries, num_segments):
    """
    Convert a list of (feature_path, num_frames, label) entries into
    {samples: Tensor, labels: Tensor} dict.
    """
    all_features = []
    all_labels = []
    skipped = 0

    for feature_path, num_frames, label in tqdm(entries, desc="Converting"):
        if num_frames < 1:
            skipped += 1
            continue
        try:
            feat = load_video_features(feature_path, num_frames, num_segments)
            all_features.append(feat)
            all_labels.append(label)
        except Exception as e:
            print(f"  Skipping {feature_path}: {e}")
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} videos due to errors")

    if len(all_features) == 0:
        print("  ERROR: No features loaded!")
        return None

    samples = torch.stack(all_features)  # (N, feat_dim, num_segments)
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Loaded {len(all_features)} videos -> samples shape: {samples.shape}, labels shape: {labels.shape}")
    return {"samples": samples, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Convert TA3N features to DM-POT .pt format")
    parser.add_argument('--list_dir', type=str, required=True,
                        help='Directory containing the .txt list files (e.g., video_dataset/all_list)')
    parser.add_argument('--feat_root', type=str, required=True,
                        help='Root directory containing frame-level .t7 features')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .pt files (e.g., Datasets/HMDB_UCF_small)')
    parser.add_argument('--variant', type=str, default='small', choices=['small', 'full'],
                        help='Which variant: small (5 classes) or full (12 classes)')
    parser.add_argument('--num_segments', type=int, default=16,
                        help='Number of temporal segments to sample per video (sequence_len)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Domain IDs
    domains = ['ucf101', 'hmdb51']

    # List file naming convention (from casual_ot dataloader)
    for domain in domains:
        for split, split_name in [('train', 'train'), ('val', 'test')]:
            list_file = f'list_{domain}_{split}_hmdb_ucf{"_small" if args.variant == "small" else ""}-feature.txt'

            print(f"\n--- Processing {domain} {split} (list: {list_file}) ---")
            entries = parse_list_file(args.list_dir, list_file, args.feat_root)

            if not entries:
                print(f"  No entries found. Trying alternate naming...")
                # Try alternate naming
                list_file = f'{split}_{domain}.txt'
                entries = parse_list_file(args.list_dir, list_file, args.feat_root)

            if not entries:
                print(f"  WARNING: Still no entries. Skipping.")
                continue

            print(f"  Found {len(entries)} entries")
            dataset = convert_split(entries, args.num_segments)

            if dataset is not None:
                out_path = os.path.join(args.output_dir, f'{split_name}_{domain}.pt')
                torch.save(dataset, out_path)
                print(f"  Saved to {out_path}")

    print("\nDone! Place the output directory in DM-POT's Datasets/ folder.")


if __name__ == '__main__':
    main()

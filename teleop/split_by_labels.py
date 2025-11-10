#!/usr/bin/env python3
"""
Script to organize episode frames into category folders based on labels.csv

Based on the timestamp ranges defined in labels.csv, this script:
1. Creates a 'labels' folder in the dataset directory
2. Inside 'labels', creates category folders: others, pickup_left, pickup_right, align_central_aperture, assemble, placement
3. For each episode, copies frames to appropriate category folders based on their frame numbers
4. Frame categorization:
   - frame < pickup_left → others
   - pickup_left ≤ frame < pickup_right → pickup_left
   - pickup_right ≤ frame < align_central_aperture → pickup_right
   - align_central_aperture ≤ frame < assemble → align_central_aperture
   - assemble ≤ frame < placement → assemble
   - placement ≤ frame < end → placement
"""

import os
import csv
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# Category names in order (excluding 'end')
CATEGORIES = [
    "others",
    "pickup_left",
    "pickup_right",
    "align_central_aperture",
    "assemble",
    "placement"
]


def read_labels_csv(csv_path):
    """Read labels.csv and return a dictionary of episode labels."""
    labels = {}
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Strip whitespace from fieldnames
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        
        for row in reader:
            # Strip whitespace from all values
            row = {k.strip(): v.strip() for k, v in row.items()}
            
            episode_num = int(row['episode'])
            labels[episode_num] = {
                'pickup_left': int(row['pickup_left']),
                'pickup_right': int(row['pickup_right']),
                'align_central_aperture': int(row['align_central_aperture']),
                'assemble': int(row['assemble']),
                'placement': int(row['placement']),
                'end': int(row['end'])
            }
    
    return labels


def get_category_for_frame(frame_num, episode_labels):
    """Determine which category a frame belongs to based on its number."""
    if frame_num < episode_labels['pickup_left']:
        return 'others'
    elif frame_num < episode_labels['pickup_right']:
        return 'pickup_left'
    elif frame_num < episode_labels['align_central_aperture']:
        return 'pickup_right'
    elif frame_num < episode_labels['assemble']:
        return 'align_central_aperture'
    elif frame_num < episode_labels['placement']:
        return 'assemble'
    elif frame_num < episode_labels['end']:
        return 'placement'
    else:
        # Frames >= end are ignored or can be treated as others
        return None


def process_episode(episode_path, episode_num, episode_labels, category_folders):
    """Process a single episode and copy frames to category folders."""
    if not episode_path.exists() or not episode_path.is_dir():
        print(f"  Warning: {episode_path} does not exist or is not a directory")
        return
    
    # Get all image files
    image_files = sorted(episode_path.glob("*.jpg")) + sorted(episode_path.glob("*.png"))
    
    if not image_files:
        print(f"  Warning: No images found in {episode_path.name}")
        return
    
    print(f"  Processing {episode_path.name}: {len(image_files)} frames")
    
    # Count frames per category
    category_counts = defaultdict(int)
    
    for img_file in image_files:
        # Extract frame number from filename (e.g., "000123.jpg" -> 123)
        frame_num = int(img_file.stem)
        
        # Determine category
        category = get_category_for_frame(frame_num, episode_labels)
        
        if category is None:
            # Frame is beyond 'end', skip it
            continue
        
        # Copy file to category folder with episode prefix
        dest_filename = f"{episode_path.name}_{img_file.name}"
        dest_path = category_folders[category] / dest_filename
        
        try:
            shutil.copy2(img_file, dest_path)
            category_counts[category] += 1
        except Exception as e:
            print(f"    Error copying {img_file.name} to {category}: {e}")
    
    # Print summary
    summary = ", ".join([f"{cat}: {count}" for cat, count in sorted(category_counts.items())])
    print(f"    Copied: {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize episode frames into category folders based on labels.csv"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Dataset directory containing episode folders and labels.csv"
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Path to labels.csv file (default: <dataset_dir>/labels.csv)"
    )
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return
    
    # Determine labels.csv path
    if args.labels_csv:
        labels_csv_path = Path(args.labels_csv)
    else:
        labels_csv_path = dataset_dir / "labels.csv"
    
    if not labels_csv_path.exists():
        print(f"Error: labels.csv not found at {labels_csv_path}")
        print("Please specify the path with --labels-csv")
        return
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Labels file: {labels_csv_path}")
    print("=" * 60)
    
    # Read labels
    print("Reading labels.csv...")
    labels = read_labels_csv(labels_csv_path)
    print(f"Found labels for {len(labels)} episodes")
    print()
    
    # Create labels folder and category folders inside it
    print("Creating category folders...")
    labels_folder = dataset_dir / "labels"
    labels_folder.mkdir(exist_ok=True)
    print(f"Created labels/ folder")
    
    category_folders = {}
    for category in CATEGORIES:
        folder_path = labels_folder / category
        folder_path.mkdir(exist_ok=True)
        category_folders[category] = folder_path
        print(f"  Created: labels/{category}/")
    print()
    
    # Find all episode folders
    episode_folders = sorted([
        d for d in dataset_dir.iterdir() 
        if d.is_dir() and d.name.startswith("episode_")
    ])
    
    if not episode_folders:
        print("Error: No episode folders found!")
        return
    
    print(f"Found {len(episode_folders)} episode folders")
    print("=" * 60)
    print()
    
    # Process each episode
    processed = 0
    skipped = 0
    
    for episode_path in episode_folders:
        # Extract episode number from folder name (e.g., "episode_0000" -> 0)
        episode_num = int(episode_path.name.split("_")[1])
        
        if episode_num not in labels:
            print(f"Warning: No labels found for {episode_path.name}, skipping...")
            skipped += 1
            continue
        
        process_episode(episode_path, episode_num, labels[episode_num], category_folders)
        processed += 1
    
    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Processed: {processed} episodes")
    print(f"  Skipped: {skipped} episodes")
    print()
    
    # Print summary of each category
    print("Category summary:")
    for category in CATEGORIES:
        folder_path = category_folders[category]
        num_files = len(list(folder_path.glob("*.jpg"))) + len(list(folder_path.glob("*.png")))
        print(f"  labels/{category}: {num_files} frames")


if __name__ == "__main__":
    main()


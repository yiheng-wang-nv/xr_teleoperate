#!/usr/bin/env python3
"""
Script to process and clean up episode folders in install_trocar_from_tray directory.
Step 1: Process images
  - Concatenates _color_0.jpg, _color_1.jpg, and _color_2.jpg into a single image (left to right)
  - Saves as <frame_name>.jpg
Step 2: Clean up folders
  - Moves all images from colors/ folder to parent episode folder
  - Deletes audios/, colors/, and depths/ folders
  - Deletes all .json files
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image

def process_images_in_episode(episode_path):
    """Process all images in a single episode folder."""
    colors_path = episode_path / "colors"
    
    if not colors_path.exists():
        print(f"  Warning: {colors_path} does not exist, skipping image processing...")
        return
    
    # Get all unique frame names
    all_images = list(colors_path.glob("*_color_*.jpg"))
    frame_names = set()
    
    for img_path in all_images:
        # Extract frame name (e.g., "000000" from "000000_color_0.jpg")
        filename = img_path.stem  # Remove .jpg
        frame_name = filename.rsplit("_color_", 1)[0]
        frame_names.add(frame_name)
    
    if not frame_names:
        print(f"  No images to process")
        return
    
    print(f"  Processing images: Found {len(frame_names)} frames")
    
    processed = 0
    errors = 0
    
    for frame_name in sorted(frame_names):
        try:
            img_0_path = colors_path / f"{frame_name}_color_0.jpg"
            img_1_path = colors_path / f"{frame_name}_color_1.jpg"
            img_2_path = colors_path / f"{frame_name}_color_2.jpg"
            output_path = colors_path / f"{frame_name}.jpg"
            
            # Check if all three required images exist
            if not img_0_path.exists() or not img_1_path.exists() or not img_2_path.exists():
                print(f"    Warning: Missing images for frame {frame_name}, skipping...")
                continue
            
            # Load images
            img_0 = Image.open(img_0_path)
            img_1 = Image.open(img_1_path)
            img_2 = Image.open(img_2_path)
            
            # Get dimensions
            width_0 = img_0.width
            width_1 = img_1.width
            width_2 = img_2.width
            total_width = width_0 + width_1 + width_2
            height = max(img_0.height, img_1.height, img_2.height)
            
            # Create new image with combined width
            combined = Image.new('RGB', (total_width, height))
            
            # Paste images (left to right: img_0, img_1, img_2)
            combined.paste(img_0, (0, 0))
            combined.paste(img_1, (width_0, 0))
            combined.paste(img_2, (width_0 + width_1, 0))
            
            # Save combined image
            combined.save(output_path, quality=95)
            
            # Close images
            img_0.close()
            img_1.close()
            img_2.close()
            combined.close()
            
            # Delete the original images
            img_0_path.unlink()
            img_1_path.unlink()
            img_2_path.unlink()
            
            processed += 1
            
        except Exception as e:
            print(f"    Error processing frame {frame_name}: {e}")
            errors += 1
    
    print(f"  Image processing completed: {processed} frames processed, {errors} errors")

def cleanup_episode(episode_path):
    """Clean up a single episode folder."""
    colors_path = episode_path / "colors"
    audios_path = episode_path / "audios"
    depths_path = episode_path / "depths"
    
    # Step 1: Move all images from colors/ to episode root
    if colors_path.exists() and colors_path.is_dir():
        image_files = list(colors_path.glob("*.jpg")) + list(colors_path.glob("*.png"))
        
        if image_files:
            print(f"  Moving {len(image_files)} images from colors/ to {episode_path.name}/")
            
            for img_file in image_files:
                dest_path = episode_path / img_file.name
                try:
                    shutil.move(str(img_file), str(dest_path))
                except Exception as e:
                    print(f"    Error moving {img_file.name}: {e}")
    
    # Step 2: Delete audios/, colors/, and depths/ folders
    folders_to_delete = [audios_path, colors_path, depths_path]
    
    for folder in folders_to_delete:
        if folder.exists() and folder.is_dir():
            try:
                shutil.rmtree(folder)
                print(f"  Deleted {folder.name}/ folder")
            except Exception as e:
                print(f"    Error deleting {folder.name}/: {e}")
    
    # Step 3: Delete all .json files in the episode folder
    json_files = list(episode_path.glob("*.json"))
    
    if json_files:
        for json_file in json_files:
            try:
                json_file.unlink()
                print(f"  Deleted {json_file.name}")
            except Exception as e:
                print(f"    Error deleting {json_file.name}: {e}")

def process_episode(episode_path):
    """Process a single episode: process images, then clean up."""
    print(f"\nProcessing {episode_path.name}...")
    
    # Step 1: Process images (concatenate _color_0 and _color_2)
    process_images_in_episode(episode_path)
    
    # Step 2: Clean up folders and files
    cleanup_episode(episode_path)
    
    print(f"  {episode_path.name} completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Process and clean up episode folders with images."
    )
    parser.add_argument(
        "base_path",
        type=str,
        help="Base directory containing episode folders (e.g., /path/to/install_trocar_from_tray)"
    )
    
    args = parser.parse_args()
    base_path = Path(args.base_path)
    
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist!")
        return
    
    # Find all episode folders
    episode_folders = sorted([
        d for d in base_path.iterdir() 
        if d.is_dir() and d.name.startswith("episode_")
    ])
    
    if not episode_folders:
        print(f"Error: No episode folders found in {base_path}")
        return
    
    print("Starting image processing and cleanup...")
    print(f"Found {len(episode_folders)} episode folders")
    print("=" * 60)
    
    # Process all episode folders
    for episode_path in episode_folders:
        process_episode(episode_path)
    
    print("\n" + "=" * 60)
    print(f"All {len(episode_folders)} episodes processed and cleaned up!")

if __name__ == "__main__":
    main()

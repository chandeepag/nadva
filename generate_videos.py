import os
import cv2
import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# ============================
# Configuration and Paths
# ============================
IMAGE_ROOT = r"C:\Users\mrpre\OneDrive\Desktop\chandeepa-fyp\dataset\Samples"  # Root folder containing instance directories
OUTPUT_DIR = r"C:\Users\mrpre\OneDrive\Desktop\chandeepa-fyp\output"  # Output directory for generated files
ANNOTATION_FILE = os.path.join(OUTPUT_DIR, "image_annotations.json")  # Path to save generated annotations

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Check for GPU Availability
# ============================
if torch.cuda.is_available():
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# ============================
# Function to Generate Annotations
# ============================
def create_annotations():
    """
    Generate image sequence annotations for each instance in the dataset.
    The annotations are stored in a JSON file in the output directory.
    """
    annotations = []  # List to store annotations
    instance_folders = [f for f in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, f))]

    if not instance_folders:
        print("Error: No valid instance folders found in Samples/!")
        return

    # Iterate over each instance folder
    for instance_id, instance in enumerate(instance_folders):
        instance_token = f"instance_{instance_id:04d}"  # Unique instance identifier
        instance_path = os.path.join(IMAGE_ROOT, instance)

        # Collect all image files (sorted to maintain order)
        image_files = sorted([img for img in os.listdir(instance_path) if img.endswith((".jpg", ".png"))])

        if not image_files:
            print(f"Warning: No images found in {instance_path}. Skipping...")
            continue

        # Generate annotations for each image in the instance folder
        for frame_id, image in enumerate(image_files):
            sample_annotation_token = f"{instance_token}_frame_{frame_id:04d}"
            annotations.append({
                "instance_token": instance_token,
                "sample_annotation_token": sample_annotation_token,
                "filename": os.path.join(instance, image),  # Path relative to 'Samples/' directory
                "visibility_token": "3"  # Default visibility token
            })

    # Save annotations to JSON file
    if annotations:
        with open(ANNOTATION_FILE, "w") as f:
            json.dump(annotations, f, indent=4)
        print(f"Annotations saved at {ANNOTATION_FILE}")
    else:
        print("Error: No annotations were generated. Check dataset structure!")

# ============================
# Function to Load Annotations
# ============================
def load_annotations():
    """
    Load existing annotations from JSON file or generate them if not found.
    Returns:
        list: List of annotation dictionaries.
    """
    if not os.path.exists(ANNOTATION_FILE):
        print("Annotations not found. Generating now...")
        create_annotations()
    
    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

    if not annotations:
        print("Error: Annotations file is empty. Regenerating...")
        create_annotations()
        with open(ANNOTATION_FILE, "r") as f:
            annotations = json.load(f)

    return annotations

# ============================
# Function to Group Annotations by Instance
# ============================
def group_annotations_by_instance(annotations):
    """
    Organize annotations by instance token.
    Args:
        annotations (list): List of annotation dictionaries.
    Returns:
        dict: Dictionary mapping instance tokens to their corresponding annotations.
    """
    instance_dict = defaultdict(list)
    for ann in annotations:
        instance_dict[ann["instance_token"]].append(ann)
    return instance_dict

# ============================
# Function to Sort Annotations
# ============================
def sort_annotations(annotations):
    """
    Sort annotations in chronological order based on sample annotation tokens.
    Args:
        annotations (list): List of annotation dictionaries.
    Returns:
        list: Sorted list of annotations.
    """
    return sorted(annotations, key=lambda x: x["sample_annotation_token"])

# ============================
# Function to Convert Image Sequences to Videos
# ============================
def images_to_videos(instance_token, annotations, frame_rate=12, duplication_factor=3):
    """
    Convert an image sequence into a video, duplicating frames for smoother playback.

    Args:
        instance_token (str): Unique identifier for the instance.
        annotations (list): List of image annotations for the instance.
        frame_rate (int): Desired frame rate of the output video.
        duplication_factor (int): Number of times each frame is duplicated to slow down playback.
    """
    images = []
    
    # Collect valid image paths
    for ann in annotations:
        img_path = os.path.join(IMAGE_ROOT, ann["filename"])
        if os.path.exists(img_path):
            images.append(img_path)
    
    if not images:
        print(f"No valid images found for instance {instance_token}")
        return
    
    # Define output video path
    output_video_path = os.path.join(OUTPUT_DIR, f"{instance_token}.mp4")
    
    # Load first image to get frame dimensions
    first_image = cv2.imread(images[0])

    if first_image is None:
        print(f"Error: Unable to load {images[0]}. Skipping instance {instance_token}.")
        return

    height, width, _ = first_image.shape  # Get image dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define video codec
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Process each image
    for img_path in images:
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Skipping unreadable image {img_path}")
            continue

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        # Duplicate frames to improve smoothness
        for _ in range(duplication_factor):
            video.write(frame)
    
    video.release()  # Save the video
    print(f"Video saved: {output_video_path}")

# ============================
# Main Processing Function
# ============================
def process_all_instances():
    """
    Load annotations, group them by instance, and convert each instance's image sequence into a video.
    """
    annotations = load_annotations()  # Load or generate annotations
    instance_dict = group_annotations_by_instance(annotations)  # Group annotations by instance

    # Process each instance
    for instance_token, anns in tqdm(instance_dict.items(), desc="Processing instances"):
        sorted_anns = sort_annotations(anns)  # Sort images chronologically
        images_to_videos(instance_token, sorted_anns)  # Convert images to video

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    process_all_instances()

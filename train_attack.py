import os
import cv2
import time
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

# =============================
# Configuration parameters
# =============================
DATASET_ROOT = r"/workspace/"
CAMERA_FOLDER = "CAM_FRONT"
CAMERA_PATH = os.path.join(DATASET_ROOT, CAMERA_FOLDER)
MODEL_SAVE_PATH = "pixel_spaced_attack.pth"
BATCH_SIZE = 1
EPOCHS = 100
FRAMES_PER_SEQUENCE = 10
TARGET_CLASSES = [0, 2, 5, 7]  # car, bus, truck, person
EPSILON = 5.0  # perturbation strength

# =============================
# Set Device (GPU or CPU)
# =============================
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
    return device

device = get_device()

# =============================
# Load YOLOv8 Model for Detection
# =============================
yolo_model = YOLO("yolov8s.pt").to(device)

# =============================
# Pixel-Spaced-Attack GAN Architecture
# =============================
class PixelGAN_Generator(nn.Module):
    def __init__(self):
        super(PixelGAN_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

generator = PixelGAN_Generator().to(device).float()
optimizer = optim.Adam(generator.parameters(), lr=0.0005)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# =============================
# Compute Detection Suppression Loss
# =============================
def compute_detection_suppression_loss(original_results, attacked_results):
    original_confidences = []
    attacked_confidences = []

    for result in original_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                if int(cls.item()) in TARGET_CLASSES:
                    original_confidences.append(conf)

    for result in attacked_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                if int(cls.item()) in TARGET_CLASSES:
                    attacked_confidences.append(conf)

    if not attacked_confidences:
        return torch.tensor(0.0, device=device)

    attack_loss = torch.stack(attacked_confidences).mean()
    return attack_loss

# ======================================
# Load and Cache All Images in Memory
# ======================================
cached_images = {}

def preload_images(cam_path):
    print("Preloading all images to RAM...")
    for f in os.listdir(cam_path):
        if f.endswith((".jpg", ".png")):
            img_path = os.path.join(cam_path, f)
            img = cv2.imread(img_path)
            if img is not None:
                cached_images[img_path] = img
    print(f"Preloaded {len(cached_images)} images.")

preload_images(CAMERA_PATH)

# ======================================
# Prepare Sequences of Frames for Training
# ======================================
def load_sequences(cam_path, frames_per_seq):
    all_images = [f for f in os.listdir(cam_path) if f.endswith((".jpg", ".png"))]
    if not all_images:
        print("No images found in CAM_FRONT folder!")
        return []

    all_images.sort(key=lambda x: int(x.split("__")[-1].split(".")[0]))
    sequences = []
    for i in range(0, len(all_images) - frames_per_seq + 1, frames_per_seq):
        seq = all_images[i:i + frames_per_seq]
        sequences.append([os.path.join(cam_path, f) for f in seq])

    print(f"Loaded {len(sequences)} CAM_FRONT sequences of {frames_per_seq} frames.")
    return sequences

sequences = load_sequences(CAMERA_PATH, FRAMES_PER_SEQUENCE)
print("Starting training loop with", len(sequences), "sequences.")


# ======================================
# Training Loop
# ======================================
start_time = time.time()

for epoch in range(EPOCHS):
    total_loss = 0
    np.random.shuffle(sequences)
    epoch_start = time.time()

    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        if len(batch_seqs) < BATCH_SIZE:
            continue

        for seq in batch_seqs:
            batch_start = time.time()
            batch_frames = []
            for img_path in seq:
                frame = cached_images.get(img_path, None)
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 640))
                batch_frames.append(transform(frame))

            if len(batch_frames) == 0:
                continue

            original_images = torch.stack(batch_frames).to(device).float()
            optimizer.zero_grad()

            # Generate adversarial perturbations
            perturbations = generator(original_images)
            adversarial_images = torch.clamp(original_images + EPSILON * perturbations, 0, 1)

            try:
                # Run YOLO on original and adversarial inputs
                with torch.no_grad():
                    original_results = yolo_model(original_images * 255.0)
                    attacked_results = yolo_model(adversarial_images * 255.0)

                # Compute detection suppression loss
                detection_loss = compute_detection_suppression_loss(original_results, attacked_results)
                perturbation_loss = torch.mean(torch.abs(perturbations))
                print(f"[Epoch {epoch+1}] Avg perturbation strength: {perturbation_loss.item():.6f}")

                loss = 5.0 * detection_loss

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5)
                    optimizer.step()
                    total_loss += loss.item()

                    print(f"[Epoch {epoch+1}] Step {i} | Loss: {loss.item():.6f}")
                    with open("training_log.txt", "a") as log:
                        log.write(f"Epoch {epoch+1} | Loss: {loss.item():.6f} | Perturbation: {perturbation_loss.item():.6f}\n")

            except Exception as e:
                print(f"Training step failed: {e}")
                continue

            print(f"[Epoch {epoch+1}] Batch time: {(time.time() - batch_start)*1000:.2f} ms")

            del original_images, adversarial_images, perturbations, batch_frames
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Epoch {epoch + 1} completed - Total Loss: {total_loss:.4f} | Time: {(time.time() - epoch_start) / 60:.2f} minutes")

    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f"pixel_epoch{epoch+1}.pth")
        print(f"Saved checkpoint: pixel_epoch{epoch+1}.pth")

# ======================================
# Save Final Trained Generator Model
# ======================================
torch.save(generator.state_dict(), MODEL_SAVE_PATH)
print(f"Pixel-Spaced-GAN model saved at {MODEL_SAVE_PATH}")

total_minutes = (time.time() - start_time) / 60
print(f"Total training time: {total_minutes:.2f} minutes")
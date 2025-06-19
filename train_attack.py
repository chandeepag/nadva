import os
import cv2
import time
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

# =============================
# Configuration
# =============================
CAMERA_FOLDER = "CAM_FRONT"
DATASET_ROOT = r"E:\\chandeepa-fyp\\dataset\\samples"
CAMERA_PATH = os.path.join(DATASET_ROOT, CAMERA_FOLDER)
MODEL_SAVE_PATH = "msfd_gan_attack.pth"
BATCH_SIZE = 1
EPOCHS = 2
FRAMES_PER_SEQUENCE = 10
TARGET_CLASSES = [0, 2, 5, 7]  # car, bus, truck, person
EPSILON = 0.1  # Maximum perturbation strength

# =============================
# Device Setup
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
# Model Setup
# =============================
yolo_model = YOLO("yolov8s.pt").to(device)

# =============================
# Pixel-Space GAN Generator
# =============================
class MSFD_Generator(nn.Module):
    def __init__(self):
        super(MSFD_Generator, self).__init__()
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
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

generator = MSFD_Generator().to(device).float()
optimizer = optim.Adam(generator.parameters(), lr=0.0005)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# =============================
# Loss Function
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

# =============================
# Preload All Images
# =============================
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

# =============================
# Load Sequences
# =============================
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

# =============================
# Training Loop
# =============================
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

            perturbations = generator(original_images)
            adversarial_images = original_images + EPSILON * perturbations
            adversarial_images = torch.clamp(adversarial_images, 0, 1)

            try:
                with torch.no_grad():
                    original_results = yolo_model(original_images * 255.0)
                with torch.no_grad():
                    attacked_results = yolo_model(adversarial_images * 255.0)

                detection_loss = compute_detection_suppression_loss(original_results, attacked_results)
                perturbation_loss = torch.mean(torch.abs(perturbations))
                loss = detection_loss + 0.1 * perturbation_loss

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5)
                    optimizer.step()
                    total_loss += loss.item()

                    print(f"[Epoch {epoch+1}] Step {i} | Loss: {loss.item():.6f}")

            except Exception as e:
                print(f"⚠️ Training step failed: {e}")
                continue

            del original_images, adversarial_images, perturbations, batch_frames
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Epoch {epoch + 1} completed - Total Loss: {total_loss:.4f} | Time: {(time.time() - epoch_start) / 60:.2f} minutes")
    torch.save(generator.state_dict(), f"msfd_epoch{epoch+1}.pth")

# =============================
# Save Final Model
# =============================
torch.save(generator.state_dict(), MODEL_SAVE_PATH)
print(f"MSfd-GAN model saved at {MODEL_SAVE_PATH}")

total_minutes = (time.time() - start_time) / 60
print(f"Total training time: {total_minutes:.2f} minutes")

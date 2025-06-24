import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn

# Path to input video for testing the attack
VIDEO_INPUT_PATH = r"/workspace/VIDEO/instance_0003.mp4"
EPSILON = 5.0  # Maximum strength of adversarial perturbation

# Select device: Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained YOLOv8 object detection model
yolo_model = YOLO("yolov8s.pt").to(device)

# ------------------------------------------------------------------------------
# Pixel-Space GAN Generator Definition
# ------------------------------------------------------------------------------
class PixelGAN_Generator(nn.Module):
    """
    Generator model that creates pixel-wise perturbations to fool object detectors.
    The network follows a simple encoder-decoder architecture.
    """
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
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # 3 output channels (RGB)
            nn.Tanh()  # Outputs are scaled to [-1, 1] for pixel-space perturbations
        )

    def forward(self, x):
        return self.model(x)

# Load the trained generator weights
gan_generator = PixelGAN_Generator().to(device)
gan_generator.load_state_dict(torch.load("/workspace/pixel_spaced_attack.pth", map_location=device))
gan_generator.eval()

# ------------------------------------------------------------------------------
# Adversarial Attack Inference and Visualization
# ------------------------------------------------------------------------------
def process_video(video_path):
    """
    Applies the trained GAN-based pixel-space attack on a video, 
    compares detection results on original vs. perturbed frames, 
    and reports attack effectiveness.
    """
    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    original_detections = 0
    attacked_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Preprocess input frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        frame_tensor = torch.tensor(frame_resized, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Generate adversarial perturbation
        with torch.no_grad():
            perturbations = gan_generator(frame_tensor)
            adversarial_image = frame_tensor + EPSILON * perturbations
            adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # Run YOLO on both original and perturbed images
        results_original = yolo_model(frame_tensor)
        original_detections += len(results_original[0].boxes) if results_original[0].boxes is not None else 0

        results_attacked = yolo_model(adversarial_image)
        attacked_detections += len(results_attacked[0].boxes) if results_attacked[0].boxes is not None else 0

        # Visualization setup
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        adv_image_np = adversarial_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_image_bgr = cv2.cvtColor((adv_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Draw detections: Green = original, Red = attacked
        for result in results_original:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for result in results_attacked:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(adv_image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Add labels and detection counts
        cv2.putText(frame_bgr, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(adv_image_bgr, "Adversarial", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        orig_count = len(results_original[0].boxes) if results_original[0].boxes is not None else 0
        attack_count = len(results_attacked[0].boxes) if results_attacked[0].boxes is not None else 0

        cv2.putText(frame_bgr, f"Detections: {orig_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(adv_image_bgr, f"Detections: {attack_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display side-by-side output
        combined_output = np.hstack((frame_bgr, adv_image_bgr))
        combined_resized = cv2.resize(combined_output, (1280, 720))
        cv2.imshow("Pixel-Space Attack vs. Original Detection", combined_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Summary report
    print("\n" + "="*50)
    print("ATTACK EFFECTIVENESS METRICS")
    print("="*50)
    print(f"Total frames processed: {total_frames}")
    print(f"Original detections: {original_detections}")
    print(f"Attacked detections: {attacked_detections}")
    
    if original_detections > 0:
        suppression_rate = ((original_detections - attacked_detections) / original_detections) * 100
        print(f"Detection suppression rate: {suppression_rate:.2f}%")
    else:
        print("No detections found in original video")

    print("="*50)

# Run the attack demo
if __name__ == "__main__":
    process_video(VIDEO_INPUT_PATH)

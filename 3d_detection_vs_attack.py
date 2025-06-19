import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import torch.nn as nn

# Define the path to the input video
VIDEO_INPUT_PATH = r"E:\chandeepa-fyp\detected\instance_0003.mp4"
EPSILON = 0.1  # Attack strength

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLOv8 model onto the selected device
yolo_model = YOLO("yolov8s.pt").to(device)

# =============================
# Pixel-Space GAN Generator
# =============================
class MSFD_Generator(nn.Module):
    """
    GAN Generator that takes RGB images and outputs pixel-space perturbations
    Input: (batch, 3, 640, 640) - RGB images
    Output: (batch, 3, 640, 640) - RGB perturbations
    """
    def __init__(self):
        super(MSFD_Generator, self).__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Decoder
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output 3 channels (RGB)
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Load the trained MSFD-GAN model
gan_generator = MSFD_Generator().to(device)
gan_generator.load_state_dict(torch.load("msfd_gan_attack.pth", map_location=device))
gan_generator.eval()  # Set the model to evaluation mode

# =============================
# Process video with actual pixel-space attack
# =============================
def process_video(video_path):
    """
    Processes the input video by applying pixel-space adversarial attack 
    and displays the original vs. attacked detection side by side.
    
    Args:
        video_path: Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)  # Open the video file
    
    # Metrics tracking
    total_frames = 0
    original_detections = 0
    attacked_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames are available

        total_frames += 1
        
        # Convert frame to RGB (OpenCV uses BGR format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to 640x640 (YOLO's default input size)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        
        # Convert image to tensor and normalize pixel values (0-1)
        frame_tensor = torch.tensor(frame_resized, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

        # =============================
        # Generate adversarial image using pixel-space perturbations
        # =============================
        with torch.no_grad():
            # Generate perturbations using trained GAN
            perturbations = gan_generator(frame_tensor)
            
            # Apply perturbations to create adversarial image
            adversarial_image = frame_tensor + EPSILON * perturbations
            
            # Clamp to valid image range [0, 1]
            adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # =============================
        # Run YOLO on ORIGINAL image
        # =============================
        results_original = yolo_model(frame_tensor * 255.0)
        original_detections += len(results_original[0].boxes) if results_original[0].boxes is not None else 0

        # =============================
        # FIXED: Run YOLO on ADVERSARIAL image
        # =============================
        results_attacked = yolo_model(adversarial_image * 255.0)
        attacked_detections += len(results_attacked[0].boxes) if results_attacked[0].boxes is not None else 0

        # =============================
        # Visualization
        # =============================
        # Convert back to OpenCV BGR format for visualization
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # Convert adversarial image back to numpy for visualization
        adv_image_np = adversarial_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_image_bgr = cv2.cvtColor((adv_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Draw YOLO detections on the original frame (GREEN)
        for result in results_original:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for normal detection

        # Draw YOLO detections on the attacked frame (RED)
        for result in results_attacked:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(adv_image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for attacked detection

        # Add text labels
        cv2.putText(frame_bgr, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(adv_image_bgr, "Adversarial", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add detection counts
        orig_count = len(results_original[0].boxes) if results_original[0].boxes is not None else 0
        attack_count = len(results_attacked[0].boxes) if results_attacked[0].boxes is not None else 0
        
        cv2.putText(frame_bgr, f"Detections: {orig_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(adv_image_bgr, f"Detections: {attack_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Concatenate original and attacked frames side by side
        combined_output = np.hstack((frame_bgr, adv_image_bgr))
        
        # Resize for better display
        combined_resized = cv2.resize(combined_output, (1280, 720))

        # Show the visualization
        cv2.imshow("MSFD Attack vs Normal Detection", combined_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print attack effectiveness metrics
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

# Entry point: Process the video
if __name__ == "__main__":
    process_video(VIDEO_INPUT_PATH)
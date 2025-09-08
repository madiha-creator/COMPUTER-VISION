import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import torch

print("OpenCV version:", cv.__version__)
print("NumPy version:", np.__version__)

# =============================
# 1. Load images from a folder
# =============================
folder_path = "C:\\Users\\ms864\\python practice\\COMPUTER VISION PROJECT\\Update820\\test\\images"
images = []

# 🔹 Ensure all images have same size
target_size = (224, 224)

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, file_name)
        img = cv.imread(img_path)

        if img is not None:
            # 🔹 Resize to same shape
            img = cv.resize(img, target_size)

            # 🔹 Normalize pixel values to [0,1]
            img_normalized = img.astype(np.float32) / 255.0

            images.append(img_normalized)
            print(f"Loaded {file_name} - shape: {img.shape}, dtype: {img_normalized.dtype}, "
                  f"min: {img_normalized.min()}, max: {img_normalized.max()}")
        else:
            print(f"Warning: Could not read {img_path}")

print(f"✅ Total images loaded & normalized: {len(images)}")

# Example: display the first normalized image
if images:
    cv.imshow("First Image (Normalized)", images[0])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Convert to grayscale and normalize
    gray = cv.cvtColor((images[0] * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
    gray_normalized = gray.astype(np.float32) / 255.0
    cv.imshow("Gray Normalized", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Convert to HSV and normalize
    hsv = cv.cvtColor((images[0] * 255).astype(np.uint8), cv.COLOR_BGR2HSV)
    hsv_normalized = hsv.astype(np.float32) / 255.0
    cv.imshow("HSV Normalized", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Convert list to numpy array
images = np.array(images)

# Dummy labels (all zeros, replace later if you have real classes)
labels = np.zeros(len(images))

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, shuffle=True
)

# =============================
# 2. Load YOLOv5 model
# =============================
print("⏳ Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("✅ YOLOv5 model loaded!")

# =============================
# 3. Run YOLO on a video file
# =============================
cap = cv.VideoCapture(r'E:\DS Material\Computer Vision\video.mp4')  # change path to your video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw results
    annotated_frame = results.render()[0]

    cv.imshow('YOLOv5 Video Detection', annotated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# =============================
# 4. Run YOLO on webcam
# =============================
cap = cv.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break

    # Run YOLOv5 inference on webcam frame
    results = model(frame)

    # Render predictions (boxes, labels, confidences)
    annotated_frame = results.render()[0]

    # Show annotated frame
    cv.imshow("YOLOv5 Webcam Detection", annotated_frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



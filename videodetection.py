import cv2
import torch
import time


# Load YOLOv5 model (pre-trained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video source (could be a file or webcam)
video_path = r"C:\Users\Krishna\downloaded_video3.mp4"  # replace with your video path or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video resolution: {frame_width}x{frame_height}")
print(f"Video FPS: {fps}")

# For measuring performance
frame_times = []

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame)
    
    # Render results on frame
    results.render()
    # `results.render()` modifies frame in-place
    
    # Display the frame
    cv2.imshow('Object Detection', results.ims[0])
    
    # Measure FPS
    end_time = time.time()
    frame_time = end_time - start_time
    frame_times.append(frame_time)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average FPS
average_fps = 1 / (sum(frame_times) / len(frame_times))
print(f"Average FPS: {average_fps:.2f}")

cap.release()
cv2.destroyAllWindows()
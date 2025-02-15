from ultralytics import YOLO
from playsound import playsound  # Install via: pip install playsound
import time


model = YOLO('dectect_yolov8.pt')

# Set the target object you want to trigger the audio for and the MP3 file path
target_class = "Fall-Detected"  # Change as needed (e.g., "chair")
mp3_file = "alert.mp3"  # Replace with the path to your MP3 file

# Start processing the webcam stream (source=0 for default camera)
# 'stream=True' makes the predict call yield results frame by frame.
for results in model.predict(source=0, show=True, stream=True):
    # Iterate over each detected box in the current frame
    for box in results.boxes:
        # Get the class id and then map to its name via the names dictionary
        class_id = int(box.cls.item())
        detected_class = results.names.get(class_id)

        # If the detected class matches our target, play the MP3 alert
        if detected_class == target_class:
            print(f"Detected: {detected_class}")
            playsound(mp3_file)

            # Optional: Pause briefly to avoid multiple triggers on consecutive frames
            time.sleep(1)

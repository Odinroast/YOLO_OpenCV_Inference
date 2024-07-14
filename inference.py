# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import YOLO Library
import ultralytics
# Library for paths
import os

# Library to load tensorflow h5 CNN
import pickle

# Load tensorflow model
with open("model.pkl", "rb") as f:
    CNN =  pickle.load(f)

# Load the YOLOV8m model
model = ultralytics.YOLO("best.pt")

# Create a camera object
cap = cv2.VideoCapture(0)

# Counter to keep track of data collection
Results = {0:'5.2 Kilo Ohms', 1:"200 Kilo Ohms", 2: "390 Kilo Ohms", 3: "560 Ohms"}

# While loop to capture the frames
while True:
    # Wait to give the camera time to start
    cv2.waitKey(20)

    # Read the frame from the camera
    ret, frame = cap.read()

    # Resize the frame to model expected size
    frame = cv2.resize(frame, (640, 480))

    # Get predictions from the model
    results = model.predict(frame)

    # Get bounding boxes from the predictions
    for r in results:
        try:
            # Get x1, y1, x2, y2 coordinates of the bounding box for a predicted resistor if there is a prediction
            x1, y1, x2, y2 = [int(x) for x in r.boxes.xyxy[0].tolist()]
            # Crop the image
            cropped = frame[y1:y2, x1:x2]
            # Resize the cropped image to 100x100
            cropped = cv2.resize(cropped, (100, 100))
            cropped = np.expand_dims(cropped, axis=0)
            out = CNN.predict(cropped)[0].tolist()
            print(Results[out.index(max(out))])
            # Show the cropped image of the resistor
            cv2.imshow("Resistor", cropped[0])
            # Wait after showing the cropped image
            cv2.waitKey(1000)

        except IndexError:
            print("No resistor detected")
            pass

    # Options to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
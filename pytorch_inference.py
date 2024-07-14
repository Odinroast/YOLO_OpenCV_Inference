# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
# Import YOLO Library
import ultralytics
# Library for paths
import os

# Library to load tensorflow h5 CNN
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load tensorflow model
CNN = torch.jit.load('CNN.pt').to(device)
# Set to evaluation mode
CNN.eval()

# Define the data transformations for the induvidual image frames
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Load the YOLOV8m model
model = ultralytics.YOLO("best.pt")

# Create a camera object
cap = cv2.VideoCapture(0)

# Counter to keep track of data collection
Results = {0:'2.1Ohms', 1:"2.7MegaOhms", 2: "100KOhms", 3: "300Ohms"}

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
            # Convert the cropped image from numpy array to PIL image
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
           # Apply the transformations
            cropped_tensor = data_transforms['test'](cropped_pil).unsqueeze(0)  # Add batch dimension
            # Perform inference
            with torch.no_grad():
                out = CNN(cropped_tensor)
                predicted_class = torch.argmax(out, dim=1).item()
                print(Results[predicted_class])
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
# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import YOLO Library
import ultralytics
# Library for paths
import os

# Load the YOLOV8m model
model = ultralytics.YOLO("best.pt")

reults = model.predict(source='0', show = True)
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2

from ultralytics import YOLO

Tk().withdraw()
filename = askopenfilename()
model = YOLO("plate_license_mt.pt")
results = model.predict(show=True,  source= filename, save=True )
# for r in results:
#     boxes = r.boxes  # Boxes object for bbox outputs
#     masks = r.masks  # Masks object for segment masks outputs
#     probs = r.probs  # Class probabilities for classification outputs
# def show_frame():
#     cv2.imshow("show", frame)
#     cv2.waitKey(1)
#
#
# for result, frame in results:
#     show_frame()

cv2.waitKey(0)

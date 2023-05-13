from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2

from ultralytics import YOLO

Tk().withdraw()
filename = askopenfilename()
model = YOLO("plate_license_mt.pt")
results = model.predict(show=True,  source= filename)
cv2.waitKey(0)

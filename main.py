from tkinter import Tk
from tkinter.filedialog import askopenfilename

from ultralytics import YOLO

Tk().withdraw()
filename = askopenfilename() # sh
model = YOLO("plate_license_mt.pt")
results = model.predict(show=True,  source= filename)

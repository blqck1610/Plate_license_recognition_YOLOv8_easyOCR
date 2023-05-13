import cv2
from ultralytics import  YOLO


model = YOLO("plate_license_mt.pt")
results = model.predict(show=True,  source= 0) #"C:\\Users\\USER\\Downloads\\test.MOV")
cv2.waitKey(0)
import cv2
from ultralytics import  YOLO


model = YOLO("plate_license_mt.pt")
results = model.predict(show=True,  source="D:\\photo\\trung-bien-so-dep-nhieu-xe-o-to-doi-gia-vai-ty-dong-gay-sot.jpg")
cv2.waitKey(0)
import numpy as np
import cv2

# OpenCV는 BGR임 
# 파란색
color = [255,0,0]
pixel = np.uint8([[color]])

hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
hsv = hsv[0][0]

print("bgr: ",color)
print("hsv: ",hsv)
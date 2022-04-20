#### Importing all the required libraries
import cv2 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Reading the images
img = cv2.imread("Citizenship_cards/new/20.jpg")
cv2.imshow("Original",img)
copy_img = img.copy()

#### Converting BGR to RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

lower = np.array([150,150,150])
higher = np.array([250,250,250])

mask = cv2.inRange(img,lower,higher)

contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cont_img = cv2.drawContours(img,contours,-1,255,3)

#### Extract Maximum Contour
c = max(contours, key = cv2.contourArea)

#### Getting coordinates
x,y,w,h = cv2.boundingRect(c)

cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),1)
plt.imshow(img)

cropped_image = copy_img[y:y+h, x:x+w]
# cv2.imwrite("Cropped20.jpg",cropped_image)

cv2.imshow("Cropped",cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

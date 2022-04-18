import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
reader = easyocr.Reader(["en"], gpu = False)
import pandas as pd

#If your runnning to single file 
image = cv2.imread('output/output60.jpg')
bound = reader.readtext(image)
for i in bound:
    text = i[1]
    print(text)


cv2.waitKey(0)
cv2.destroyAllwindows()
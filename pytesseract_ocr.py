### Using Tesseract OCR engine

import cv2
import pytesseract
import glob
def ocr_core(img):
    text = pytesseract.image_to_string(img, config ='-l eng --oem 3 --psm 11' )
    return text

path = glob.glob("output/*.jpg")
for img in path:
    img_name = img.split('.')[0]
    img_array = cv2.imread(img)
    text = ocr_core(img)
    with open(img_name + ".txt", 'w') as f:
        f.write(text)
        f.write('\n')


    



















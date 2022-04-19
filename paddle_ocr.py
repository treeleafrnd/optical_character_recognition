from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import cv2
import os
import glob

### Setup model
ocr_model = PaddleOCR(lang = "en")

img_path = '/content/drive/MyDrive/OCR/drive_data/output/output57.jpg'
print(img_path)
# img_path = glob.glob("/content/drive/MyDrive/OCR/drive_data/output/*.jpg")

result = ocr_model.ocr(img_path)
for i in result:
  text = (i[1][0])
  print(text)
  with open("/content/drive/MyDrive/OCR/drive_data/paddle_ocr_text/output57.txt", "a") as o:
    o.write(text)
    o.write('\n')
  


# final_result = []
# for img in img_path:
#     img_name = img.split('.')[0]
#     img_array = cv2.imread(img)
#     result = ocr_model.ocr(img)
#     final_result.append(result)




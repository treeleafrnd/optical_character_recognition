This is the task of Optical Character Recognition that automate the data extraction from printed or written text from a scanned document or image file and then converting the text into a machine-readable form where we have created own datasets for implementation into OCR model. 
Dataset and Output link: https://drive.google.com/drive/folders/10w_Gg4HGCKYTlCCCVXc98nmIWOKl-dhF?usp=sharing 
Steps of the task are:-

1. Creating own datasets, here the datasets are aligned images which are to be aligned. 
link for the dataset:- https://drive.google.com/drive/folders/1oe_UDLYMxJH74HqbQWvzmvdBJ-em_pkw?usp=sharing

2. Image Alignment:-
At first images are read by using opencv library. 
After reading the image, the image is resized into (600,600) pixels. 
Then we convert the resized image into gray scale image for applying edge detection. 
After that canny edge detector is used to detect the edges in the images. 
In input data or images most of them have horizontal lines so I apply hough transformation to detect the horizontal lines inside the images. 
Now after that we get lines and use those lines to calculate the slope.
Here alogorithm detected many lines in which some of them may be undesired lines so I use median such that 50% of the data comes from median value and use them. 
Then by using the slope I calculate angle of inclination and finally rotate the images in a certain degree.
link for the file is :- https://github.com/treeleafrnd/Optical_character_recognition/blob/master/function_image_alignment.py
link for the aligned image is:- https://drive.google.com/drive/folders/1aUBpdhBLoNbKWqdLq5td6Uxk-ocO2jbG?usp=sharing

3. Extracted text from aligned images using Tesseract OCR engine:-
file link:- https://github.com/treeleafrnd/Optical_character_recognition/blob/master/pytesseract_ocr.py
Extracted text files link:- https://drive.google.com/drive/folders/1fZ_eeaVdkV0DvTeyNbEw4jT03zpB4YeS?usp=sharing

4. Extracted text from aligned images using EasyOCR engine:-
file link:- https://github.com/treeleafrnd/Optical_character_recognition/blob/master/easy_ocr.py
Extracted text files link:- https://drive.google.com/drive/folders/12hM5vMVMIKYKZJ9a3QlnXAK6MtOG9D9O?usp=sharing

5. Extracted text from aligned images using PaddleOCR engine:-
file link:- https://github.com/treeleafrnd/Optical_character_recognition/blob/master/paddle_ocr.py
Extracted text files link:-https://drive.google.com/drive/folders/1nJjgqRqwpZispaYFxTlxQ_eMZJ2KsDX1?usp=sharing

6. Generated evaluation report by calculating sentence Error Rate by taking reference text as a actual text and extracted text as a text extracted
from OCR engines:-
Here Actual text is extracted from google drive
link for actual text is:- https://drive.google.com/drive/folders/137-KUkqMs8hG2twOCeBoCaDZ_M-A7PtP?usp=sharing
report link:- https://drive.google.com/file/d/11aRodsZv8ATjdiLipRVdImf9iDxaox9F/view?usp=sharing

7. Image alignment using homography approaches:-
file's link:- https://drive.google.com/drive/folders/1OwtBEVDvUvZ0L7ZNXBApGoQS7hXYECCw
input and output link:-https://drive.google.com/drive/folders/1XM8IfLs20fvMp2YI0vup6Xv0vG-GtfJQ?usp=sharing 

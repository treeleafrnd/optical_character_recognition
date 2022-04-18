This is the task of Optical Character Recognition that automate the data extraction from printed or written text from a scanned document or image file and then converting the text into a machine-readable form where we have created own datasets for implementation into OCR model.
Dataset and Output link: https://drive.google.com/drive/folders/10w_Gg4HGCKYTlCCCVXc98nmIWOKl-dhF?usp=sharing
    For image Alignment:-
    At first images are read by using opencv library. After reading the image, the image is resized into (600,600) pixels. 
    Then we convert the resized image into gray scale image for applying edge detection. 
    After that canny edge detector is used to detect the edges in the images. 
    In input data or images most of them have horizontal lines so I apply hough transformation to detect the horizontal lines inside the images. 
    Now after that we get lines and use those lines to calculate the slope.
    Here alogorithm detected many lines in which some of them may be undesired lines so I use median such that 50% of the data comes from median value and use them. 
    Then by using the slope I calculate angle of inclination and finally rotate the images in a certain degree.
    For text extraction
Here we have extracted a text from aligned images by using three different OCR engine i.e TesseractOCR, EasyOCR and PaddleOCR.

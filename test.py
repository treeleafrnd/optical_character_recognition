
import glob
import cv2
from cv2 import Laplacian
import numpy as np
import glob
import os


widthImg=600
heightImg =800




def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(15,15),1)
    sobelx = cv2.Sobel(src=imgBlur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
     
    # Calculation of Sobely
    sobely = cv2.Sobel(imgBlur,cv2.CV_64F,0,1,ksize=5)
     
    # Calculation of Laplacian
    laplacian = cv2.Laplacian(imgBlur,cv2.CV_64F)
    #cv2.convertScaleAbs(Laplacian)
    outputImg8U = cv2.convertScaleAbs(laplacian, alpha=(255.0/65535.0))
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow("laplacian",laplacian)
    kernel = np.ones((3,3))
    imgDial = cv2.dilate(outputImg8U,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=2)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.1*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour,biggest, -1, (255, 0, 0), 20)
    cv2.imshow("imageoutline",imgContour)
    return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
     

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped


#loading i/o
path = glob.glob("./input/*.jpg")
path2=glob.glob("./output")
    
image = []
count = 1
for images in path:
    
        
    img= cv2.imread(images)
    img = cv2.resize(img,(widthImg,heightImg))

    imgContour = img.copy()
    imgThres = preProcessing(img) 
    
    
    biggest = getContours(imgThres)
    if biggest.size !=0:
        imgWarped=getWarp(img,biggest)
        imageArray = ([imgContour, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
        cv2.imwrite("sobel/frame%d.jpg" % count, imgWarped)
        count+=1
    else:
        cv2.imwrite(f"unaligned/{os.path.basename(images)}",img)
        
    
    


    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
       
   
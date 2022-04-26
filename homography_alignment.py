##### Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

##### Reading the images 

def read_img(img):
    img1_color = cv2.imread(img) # Image to be aligned.
    img2_color = cv2.imread("Citizenship_images/Citizen/suman/reference.jpg") # Reference image.
    ##### Converting to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
    return img1_color, img2_color, img1, img2,height, width

def detect_kp(img1,img2):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    orb_detector = cv2.ORB_create(5000)
    #### Computing keypoints and descriptors                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    return kp1,d1,kp2,d2

def detect_matches(kp1, d1, kp2, d2, img1_color, img2_color):
    ## Matching features between the two images here we create a Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    # Matching the two sets of descriptors.
    matches = matcher.match(d1, d2)
    # Sorting matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    #### Remove not so good matches
    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]
    im_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, None)
    # cv2.imshow("Matches",im_matches)
    # Taking the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    return no_of_matches, matches,im_matches

def detect_homography(no_of_matches, matches):
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Finding the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography

def transformed(homography, height, width):
    transformed_img = cv2.warpPerspective(img1_color,homography, (width, height))
    return transformed_img

def save_output(transformed_img,im_matches, image_resize,transformed_img_resize):
    path = "Citizenship_images/Citizen/suman/orb"
    cv2.imwrite(os.path.join(path,"align5.jpg"),transformed_img)
    cv2.imwrite(os.path.join(path,"matched5.jpg"),im_matches)
    cv2.imshow("Matched",im_matches_resize)
    cv2.imshow("Aligned",transformed_img_resize)
    cv2.imshow("original",image_resize)
    
image = "Citizenship_images/Citizen/suman/5.jpg"
img1_color, img2_color, img1, img2,height, width = read_img(image)
image_resize = cv2.resize(img1_color, (800,800))
kp1,d1,kp2,d2 = detect_kp(img1,img2)
no_of_matches, matches, im_matches = detect_matches(kp1, d1, kp2, d2, img1_color, img2_color)
homography = detect_homography(no_of_matches, matches)
transformed_img = transformed(homography, height, width)
transformed_img_resize = cv2.resize(transformed_img,(700,700))
im_matches_resize = cv2.resize(im_matches, (1000,800))
save_output(transformed_img, im_matches, image_resize, transformed_img_resize)


cv2.waitKey(0)
cv2.destroyAllWindows()
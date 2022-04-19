##### Importing Libraries
import cv2
import numpy as np

##### Reading the images 
img1_color = cv2.imread("Grace_alignment/2.jpg") # Image to be aligned.
img2_color = cv2.imread("Grace_alignment/reference.jpg") # Reference image.

##### Converting to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape

##### Creating ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)

#### Computing keypoints and descriptors
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

## Matching features between the two images here we create a Brute Force matcher with Hamming distance
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Matching the two sets of descriptors.
matches = matcher.match(d1, d2)

# Sorting matches on the basis of their Hamming distance.
matches.sort(key = lambda x: x.distance)

# Taking the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

# Finding the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

# Using this matrix to transform the colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,homography, (width, height))
transformed_img_resize = cv2.resize(transformed_img,(600,600))
# Saving the output
cv2.imwrite('output2.jpg', transformed_img)
cv2.imshow("Aligned Image",transformed_img_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()
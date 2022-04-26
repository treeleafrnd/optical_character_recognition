import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def detect_kp_des(imguser,imgcom):
    #Detect keypoints and compute keypointer descriptors
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0,edgeThreshold=300)
    kpts1, descs1 = sift.detectAndCompute(imguser,None)
    kpts2, descs2 = sift.detectAndCompute(imgcom,None)
    return kpts1,descs1,kpts2,descs2

def match_des(descs1,descs2):
    ##### Here we have used flann based matcher
    FLANN_BASED_MATCHER = 1
    index_params = dict(algorithm=FLANN_BASED_MATCHER, trees=10)
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(descs1, descs2, 2)
    #Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)
    return matches

def draw_matches(imguser,kp1,imgcom, kp2, matches):
    im_matches = cv2.drawMatches(imguser, kpts1, imgcom, kpts2, matches, None)
    match_draw = cv2.imread("Matches",im_matches)
    return match_draw


def ratio_test(matches):
    #Ratio test, to get good matches.
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    return src_pts, dst_pts,good

def homography(src_pts, dst_pts,imguser,imgcom):
    #### finding homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    h,w = imguser.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    canvas = imgcom.copy()
    cv2.polylines(canvas,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    return canvas,M
def draw_match(canvas,imguser,kpts1,kpts2,good):
    matched = cv2.drawMatches(imguser,kpts1,imgcom,kpts2,good,None)
    return matched
def crop_img(imguser,M,imgcom):
    #Crop the matched region 
    h,w = imguser.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
    found = cv2.warpPerspective(imgcom,perspectiveM,(w,h),borderMode=cv2.BORDER_REPLICATE)
    return found
def save_output(found,imgcom,matched):
    path = "Citizenship_images/Citizen/suman/sift"
    # cv2.imwrite(os.path.join(path,"output9.jpg"),found)
    # cv2.imwrite(os.path.join(path,"matched9.jpg"),matched)
    
imguser = cv2.imread("Citizenship_images/Citizen/suman/reference.jpg")
imgcom = cv2.imread("Citizenship_images/Citizen/suman/1.jpg")
imguser = cv2.resize(imguser, (700,700))
imgcom = cv2.resize(imgcom, (700,700))
cv2.imshow("Original",imgcom)
# path = glob.glob("Citizenship_images/driving_license/uncrop/1.jpg")
# for img in path:
#     imgname = img.split('.')[0]
#     imgcom = cv2.imread(img)
kpts1,descs1,kpts2,descs2 = detect_kp_des(imguser,imgcom)
matches = match_des(descs1,descs2)
src_pts, dst_pts, good = ratio_test(matches)
canvas, M = homography(src_pts, dst_pts,imguser,imgcom)
matched = draw_match(canvas, imguser, kpts1, kpts2, good)
found = crop_img(imguser, M, imgcom)
save_output(found, imgcom,matched)
cv2.imshow("Matches",matched)
cv2.imshow("Cropped",found)

cv2.waitKey(0)
cv2.destroyAllWindows()
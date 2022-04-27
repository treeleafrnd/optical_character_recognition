import cv2
import numpy as np

### Initializes the ORB Feature Detector object
MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures= 5000)

#### Preparing the FLANN Based Matcher
index_params = dict(algorithm = 1, trees = 3)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params, search_params)



#### Functions for loading input image
def load_input():
    input_img = cv2.imread("/home/chhabilal/Desktop/treeleaf/OCR_project/grace/input/reference.jpg")
    input_img = cv2.resize(input_img,(600,600))
    gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ##### finding keypoints ans Descriptors with ORB
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return gray_image, keypoints, descriptors

##### Function for Computing Matches between the train and query descriptors
def compute_matches(descriptors_input, descriptors_output):
    if (len(descriptors_output)!=0 and len(descriptors_input)!=0):
        matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output, np.float32), k =2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])
        return good
    else:
        return None


#### Main Working Logic
if __name__=='__main__':


### Getting information from the input image
    input_image, input_keypoints, input_descriptors = load_input()

    #### making camera ready
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    while(ret):
        ret, frame = cap.read()

##### Condition check for error escaping
        if(len(input_keypoints)<MIN_MATCHES):
            continue

##### Resizing input_image for fast computation
        frame = cv2.resize(frame, (700,700))
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##### Computing and Matching the keypoints of Input image and query Image
        output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
        matches = compute_matches(input_descriptors, output_descriptors)

        if (matches!= None):
            output_final = cv2.drawMatchesKnn(input_image, input_keypoints, 
                                                    frame, output_keypoints, 
                                                    matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


            cv2.imshow("Final Output",output_final)
        else:
            cv2.imshow("Final Output", frame)
        
        key = cv2.waitKey(5)
        if(key==27):
            break


# DataFlair background removal 

# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp


# initialize mediapipe 
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# store background images in a list
image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

# create videocapture object to access the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    # flip the frame to horizontal direction
    frame = cv2.flip(frame, 1)
    height , width, channel = frame.shape

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the result 
    results = selfie_segmentation.process(RGB)

    # extract segmented mask
    mask = results.segmentation_mask

    # it returns true or false where the condition applies in the mask
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.6

    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, frame, bg_image)

    # show outputs
    #cv2.imshow("mask", mask)
    cv2.imshow("Output", output_image)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # if 'd' key is pressed then change the background image
    elif key == ord('n'):

        if image_index != len(images)-1:
            image_index += 1
        else:
            image_index = 0
        bg_image = cv2.imread(image_path+'/'+images[image_index])


# release the capture object and close all active windows 
cap.release()
cv2.destroyAllWindows()




#Importing required libraries
import cv2
import numpy as np

#Reading image
img = cv2.imread("dushyantprofile.png")
cv2.imshow("Image", img)

#Detecting edges of the input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur (gray, 5)
edges = cv2.adaptiveThreshold (gray, 255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)

#Cartoonifying the image
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

# Step-5: Final Output (Cartoon Image)
cv2.imshow("Image", img)
cv2.imshow("Edges",edges)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

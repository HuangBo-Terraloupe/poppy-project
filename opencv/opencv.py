import numpy as np
import cv2

class ShapeDetector:
    def __init__(self):
        pass
 
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
 
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape


sd = ShapeDetector()
img = cv2.imread('111_1024.jpg', cv2.IMREAD_COLOR)
ratio = 1.

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([90,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# eroding + dilating
kernel5 = np.ones((10,10),np.uint8)
kernel10 = np.ones((10,10),np.uint8)
mask = cv2.erode(mask,kernel10,iterations = 1)
mask = cv2.dilate(mask,kernel10,iterations = 2)
mask = cv2.erode(mask,kernel10,iterations = 1)


# find contours
(cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = []
# loop over our contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
 
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    x,y,w,h = cv2.boundingRect(c)
    areas.append(w*h)

largest_id = areas.index(max(areas))

cv2.drawContours(img, [cnts[largest_id]], -1, (60, 255, 255), 2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
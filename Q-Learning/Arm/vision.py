"""
@author: Huang Bo
"""


import cv2
import numpy as np


BENCH_WIDTH = 29.5
BENCH_HEIGHT = 20.5
BACKGROUND_COLOR = "b"
OBJECT_COLOR = "g"


class Color_segmentation:

    def __init__(self, image, bentch_mark_x, bentch_mark_y, coordinates_offset):
        self.image = image
        self.bentch_mark_x = bentch_mark_x
        self.bentch_mark_y = bentch_mark_y
        self.coordinates_offset = coordinates_offset

    def detect_biggest_color_region(self, color):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        if color == "b":
            lower_range = np.array([90, 50, 50], dtype=np.uint8)
            upper_range = np.array([120, 255, 255], dtype=np.uint8)

        elif color == "y":
            lower_range = np.array([10, 120, 100], dtype=np.uint8)
            upper_range = np.array([60, 255, 255], dtype=np.uint8)

        elif color == "g":
            lower_range = np.array([10, 120, 100], dtype=np.uint8)
            upper_range = np.array([60, 255, 255], dtype=np.uint8)

        elif color == "r":
            lower_range = np.array([160, 50, 50], dtype=np.uint8)
            upper_range = np.array([200, 255, 255], dtype=np.uint8)

        else:
            print "please input the right color"

        mask = cv2.inRange(hsv_image, lower_range, upper_range)



        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=8)
        mask = cv2.erode(mask, kernel, iterations=6)

        if color=="r":
            cv2.imshow('mask', mask)
            cv2.waitKey(1)



        (cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        contours_info = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            areas.append(w * h)
            contours_info.append([x,y,w,h])

        if len(areas)>=1:

            largest_id = areas.index(max(areas))
            cv2.drawContours(self.image, [cnts[largest_id]], -1, (0, 255, 0), 2)
            # cv2.imshow('image', image)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()
            return contours_info[largest_id]
        else:
            print "Warning: did not detect an object"
            return None

    def bentch_mark(self, contours_info):
        w = contours_info[2]
        h = contours_info[3]

        ratio_x = float(self.bentch_mark_x) / float(w)
        ratio_y = float(self.bentch_mark_y) / float(h)

        return ratio_x , ratio_y

    def coordinate_transform(self, contours_info, ratio_x, ratio_y, original_x, original_y):

        x = contours_info[0] - original_x + contours_info[2]*0.5
        y = contours_info[1] - original_y + contours_info[3]*0.5
        x_real = x * ratio_x
        y_real = y * ratio_y

        return x_real, y_real


def getGoalPosition(image):
    detector = Color_segmentation(image=image, bentch_mark_x=29.5, bentch_mark_y=20.5, coordinates_offset=0)

    contours_info_b = detector.detect_biggest_color_region(BACKGROUND_COLOR)
    if contours_info_b:
        original_x, original_y = contours_info_b[0], contours_info_b[1]
        ratio_x, ratio_y = detector.bentch_mark(contours_info_b)
        contours_info_r = detector.detect_biggest_color_region(OBJECT_COLOR)

        if contours_info_r:
            x, y = detector.coordinate_transform(contours_info_r, ratio_x, ratio_y, original_x, original_y)
            y -= BENCH_HEIGHT
            x -= BENCH_WIDTH/2.
            x, y = -y, -x

            return np.array([x,y])

    return None



if __name__ == '__main__':
    
    camera = cv2.VideoCapture(1)
    
    while 1:
        _, image = camera.read()
        print getGoalPosition(image)

        cv2.imshow('image', image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    camera.release()
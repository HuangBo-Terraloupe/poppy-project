import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Color_segmentation:
    def __init__(self, image, box_length):
        self.image = image
        self.box_length = box_length

    def detect_biggest_color_region(self, color):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        if color == "b":
            lower_range = np.array([90, 50, 50], dtype=np.uint8)
            upper_range = np.array([130, 255, 255], dtype=np.uint8)

        elif color == "g":
            lower_range = np.array([50, 100, 100], dtype=np.uint8)
            upper_range = np.array([70, 255, 255], dtype=np.uint8)

        elif color == "r":
            lower_range = np.array([0, 70, 50], dtype=np.uint8)
            upper_range = np.array([10, 255, 255], dtype=np.uint8)

        else:
            print "please input the right color"

        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        kernel10 = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel10, iterations=1)
        mask = cv2.dilate(mask, kernel10, iterations=2)
        mask = cv2.erode(mask, kernel10, iterations=1)

        (cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        contours_info = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            areas.append(w * h)
            contours_info.append([x,y,w,h])

        largest_id = areas.index(max(areas))
        cv2.drawContours(image, [cnts[largest_id]], -1, (0, 0, 0), 2)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return contours_info[largest_id]

    def bentch_mark(self, contours_info):
        y = contours_info[0]
        x = contours_info[1]
        w = contours_info[2]
        h = contours_info[3]

        ratio = float(self.box_length) / float(w)
        x_real = x * ratio
        y_real = y * ratio
        h_real = h * ratio

        x_translation = -(x_real + h_real)
        y_translation = (y_real + self.box_length*0.5)

        return ratio, x_translation, y_translation

    def coordinate_transform(self, contours_info, ratio, x_translation, y_translation, alpha=-np.pi*0.5):
        x = contours_info[1] + 0.5 * contours_info[3]
        y = contours_info[0] + 0.5 * contours_info[2]
        x = x * ratio
        y = y * ratio
        tm = np.matrix([[np.cos(alpha), -np.sin(alpha), x_translation],
                        [np.sin(alpha), np.cos(alpha), y_translation],
                        [0, 0, 1]])
        original = np.matrix([[x], [y], [1]])
        return np.dot(tm, original)


if __name__ == '__main__':
    image = cv2.imread("/home/huangbo/Desktop/1111.jpg")
    detector = Color_segmentation(image, 15)

    contours_info = detector.detect_biggest_color_region("b")
    #print contours_info

    ratio, x_translation, y_translation = detector.bentch_mark(contours_info)

    contours_info = detector.detect_biggest_color_region("g")
    result = detector.coordinate_transform(contours_info, ratio, x_translation, y_translation)

    print result[0,0]
    print result[1,0]



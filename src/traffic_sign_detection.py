import argparse
import cv2
import math
import numpy as np
import traffic_sign_detection as traffic

def find_shapes(img, img_to_show):
    find_circle(img, img_to_show, "red")
    find_circle(img, img_to_show, "blue")
    find_triangle(img, img_to_show)
    find_square(img, img_to_show)
    find_stop(img, img_to_show)

def find_circle(img, img_to_show, color):
    
    # Segment image according to the color received as argument
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "red":
        img_red1 = cv2.inRange(img_hsv, (0, 180, 70), (10, 255, 255))
        img_red2 = cv2.inRange(img_hsv, (170, 90, 70), (180, 255, 255))
        img_color = img_red1 + img_red2
    else:
        img_color = cv2.inRange(img_hsv, (100, 150, 70), (140, 255, 255))

    result_color = cv2.bitwise_and(img, img, mask=img_color)
    img_gray = cv2.cvtColor(result_color, cv2.COLOR_BGR2GRAY)
    
    # Smoothing of the image
    simple_blur = cv2.blur(img_gray, (9, 9))
    gaussian_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_equalized = cv2.equalizeHist(cv2.absdiff(gaussian_blur, simple_blur))

    # Detection of circles
    circles = cv2.HoughCircles(img_equalized, cv2.HOUGH_GRADIENT, 1.2, max(img.shape[0], img.shape[1]) / 7, param1=200, param2=100, minRadius=0, maxRadius=0)

    # Draw circles on the image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img_to_show, (x, y), r, (0, 255, 0), 4)
            cv2.putText(img_to_show, color + " circle", ((int)(x-r/2), y-r+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 105, 0), 2)
            print(color + " circle")

def find_triangle(image, img_to_show):

    # Smoothing of the image
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
   
    # Converting to HSV color space in order to segment the image according to colors
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Range for upper range of red
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    
    mask1 = mask1 + mask2

    result_red = cv2.bitwise_and(image, image, mask=mask1)
    
    gray = cv2.cvtColor(result_red, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    
    # Detect contours using Canny Edge Detector
    canny = cv2.Canny(thresh, 100, 200)
    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyse contours to detect triangles
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 3 and cv2.isContourConvex(approx) and image.shape[0] * image.shape[1] / math.fabs(cv2.contourArea(approx)) < 10000:
                x = approx.ravel()[0]
                y = approx.ravel()[1] + 2
                cv2.drawContours(img_to_show, [c], -1, (0, 255, 0), 2)
                cv2.putText(img_to_show, "triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 105, 0), 2)
                print("red triangle")

def find_square(image, img_to_show):

    # Smoothing of the image
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
   
    # Color segmentation of the image
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    img_blue = cv2.inRange(hsv, (100, 120, 70), (140, 255, 255))
    result_blue = cv2.bitwise_and(image, image, mask=img_blue)

    gray = cv2.cvtColor(result_blue, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    # Detect contours using Canny Edge Detector
    canny = cv2.Canny(thresh, 100, 200)
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyse contours to detect squares / rectangles
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # A square is convex, has four contours and it can't be too small compared to the size of the image
        if len(approx) == 4 and cv2.isContourConvex(approx) and image.shape[0] * image.shape[1] / math.fabs(cv2.contourArea(approx)) < 10000:
            maxCosine = 0
            i = 2
            while i < 5:
                cosine = math.fabs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
                maxCosine = max(cosine, maxCosine)
                i += 1
            # We assume that the angles between the contours can vary between ~70 and ~110 degrees.
            if maxCosine < 0.3:
                x = approx.ravel()[0]
                y = approx.ravel()[1] - 5
                cv2.drawContours(img_to_show, [c], -1, (0, 255, 0), 2)
                cv2.putText(img_to_show, "square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 105, 0), 2)
                print("blue square")

def find_stop(img, img_to_show):
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    img_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    img_red1 = cv2.inRange(img_hsv, (0, 120, 70), (10, 255, 255))
    img_red2 = cv2.inRange(img_hsv, (170, 120, 70), (180, 255, 255))
    img_color = img_red1 + img_red2

    result_color = cv2.bitwise_and(img, img, mask=img_color)
    img_gray = cv2.cvtColor(result_color, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    canny = cv2.Canny(thresh, 100, 200)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 8 and cv2.isContourConvex(approx) and img.shape[0] * img.shape[1] / math.fabs(cv2.contourArea(approx)) < 10000:
            maxCosine = 0
            i = 2
            while i < 9:
                cosine = math.fabs(angle(approx[i % 8], approx[i - 2], approx[i - 1]))
                maxCosine = max(cosine, maxCosine)
                i += 1
            # We assume that the angles between the contours can vary between ~30 and ~60 degrees.
            if maxCosine > 0.5 and maxCosine < 0.9:
                x = approx.ravel()[0]
                y = approx.ravel()[1] - 5
                cv2.drawContours(img_to_show, [cnt], 0, (0, 255, 0), 6)
                cv2.putText(img_to_show, "STOP", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 105, 0), 2)
                print("STOP")

# Auxiliary function to determine angles between lines
def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]

    if (dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10 <= 0:
        return math.sqrt(2) / 2
    return (dx1*dx2 + dy1*dy2)/math.sqrt((pow(dx1, 2) + pow(dy1, 2))*(pow(dx2, 2) + pow(dy2, 2)) + 1e-10)
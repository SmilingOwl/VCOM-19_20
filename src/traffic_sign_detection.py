import cv2
import numpy as np
import imutils as imutils
from pyimagesearch.shapedetector import ShapeDetector

def find_circle(img, img_to_show, color):
    
    # Obtain black/white image with the wanted colors, according to the string received in color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "red":
        img_red1 = cv2.inRange(img_hsv, (0, 70, 70), (10, 255, 255))
        img_red2 = cv2.inRange(img_hsv, (170, 70, 70), (180, 255, 255))
        img_color = img_red1 + img_red2
    else:
        img_color = cv2.inRange(img_hsv, (100, 70, 0), (140, 255, 255))

    # Apply Canny Edge Detector to obtain edges, blur image and obtain circles through HoughCircles
    img_canny = cv2.Canny(img_color, 100, 200)
    img_blur = cv2.GaussianBlur(img_canny, (5, 5), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1.5, img_blur.shape[0]/8, param1=200, param2=100, minRadius=0, maxRadius=0)

    # Obtain biggest circle
    max_radius = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if r > max_radius:
                circle_x = x
                circle_y = y
                max_radius = r
    if max_radius > -1:
        cv2.circle(img_to_show, (circle_x, circle_y), max_radius, (0, 255, 0), 4)
        print(color + " circle")

def find_triangle(img):
    
    width = 400
    height = 400
    dim = (width, height)
    
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", image)
    cv2.waitKey(0)

    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(height)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
   
    # converting to HSV color space
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    
    mask1 = mask1 + mask2

    result_red = cv2.bitwise_and(image, image, mask=mask1)
    cv2.imshow("result", result_red)
    cv2.waitKey(0)

    gray = cv2.cvtColor(result_red, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 20,255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Grey", thresh)
    cv2.waitKey(0)

    # finding contours
    canny = cv2.Canny(thresh, 100, 200)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
        M = cv2.moments(c)
        if(M["m00"]==0): # this is a line
            shape = "line" 
            print(shape)
        else: 
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
            
            print(shape)
            # multiply the contour (x, y)-coordinates by the resize ratio
            c = c.astype("float")
            c *= ratio
            c =  c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 2)

            cv2.imshow("Image", image)
            cv2.waitKey(0)
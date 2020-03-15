import cv2
import argparse
import numpy as np

def find_circle(img, img_to_show, color):
    # Obtain black/white image with the wanted colors, according to the string received in color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "red":
        img_red1 = cv2.inRange(img_hsv, (0, 120, 70), (10, 255, 255))
        img_red2 = cv2.inRange(img_hsv, (170, 120, 70), (180, 255, 255))
        img_color = img_red1 + img_red2
    else:
        img_color = cv2.inRange(img_hsv, (100, 150, 0), (140, 255, 255))

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

def main():
    # Get image from arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    args = vars(ap.parse_args())

    # Read image
    img = cv2.imread(args["image"], 1) # create image
    img_to_show = img.copy()

    # Find circle
    find_circle(img, img_to_show, "red")
    find_circle(img, img_to_show, "blue")

    # Show result
    cv2.imshow('Circle', img_to_show)
    cv2.waitKey(0) # so that the window doesn't close right away
    cv2.destroyAllWindows() # to destroy all open windows


main()
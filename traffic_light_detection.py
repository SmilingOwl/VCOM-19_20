import cv2
import argparse
import numpy as np

def find_biggest_circle(img):
    img_canny = cv2.Canny(img, 100, 200)
    img_blur = cv2.GaussianBlur(img_canny, (5,5), 0)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img_blur.shape[0], param1=200, param2=100, minRadius=0, maxRadius=0)

    max_radius = -1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if r > max_radius:
                circle_x = x
                circle_y = y
                max_radius = r

    cv2.circle(img, (circle_x, circle_y), max_radius, (0, 255, 0), 4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image")
    args = vars(ap.parse_args()) # get image from arguments

    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # create a normal window
    img = cv2.imread(args["image"], 1) # create image

    find_biggest_circle(img)

    cv2.imshow('Circle', img)
    cv2.waitKey(0) # so that the window doesn't close right away
    cv2.destroyAllWindows() # to destroy all open windows


main()
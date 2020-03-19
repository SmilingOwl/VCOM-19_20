import sys
import cv2
import image
import argparse
import numpy as np
import traffic_sign_detection as traffic

parser = argparse.ArgumentParser(description="Traffic sign detection")
parser.add_argument('method' ,help='\'camera\' or \'file\'', type=str)
parser.add_argument('-i', '--image', dest='path', default='img.jpg',type=str)
img = None

args = parser.parse_args()
arg = args.method

if arg == 'camera':
    img = image.CaptureCameraImage()

elif arg == 'file':
    print('Opening image ' + args.path)
    img = image.ReadImageFile(args.path)
    img_to_show = img.copy()

    # Find circle
    traffic.find_circle(img, img_to_show, "red")
    traffic.find_circle(img, img_to_show, "blue")

    # Show result
    cv2.imshow('Circle', img_to_show)
    cv2.waitKey(0) # so that the window doesn't close right away
    cv2.destroyAllWindows() # to destroy all open windows

if img is None:
    print('Image not found')
    quit()


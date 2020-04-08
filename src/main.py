import sys
import cv2
import argparse
import numpy as np
import traffic_sign_detection as traffic

parser = argparse.ArgumentParser(description="Traffic sign detection")
parser.add_argument('method' ,help='\'camera\' or \'file\'', type=str)
parser.add_argument('-i', '--image', dest='path', default='img.jpg',type=str)
img = None

# parse all the arguments received
args = parser.parse_args()
arg = args.method

# get images from camera
if arg == 'camera':
    cap = cv2.VideoCapture(0)
    img = None
    while(True):
        ret, frame = cap.read()
        frame_to_show = frame.copy()
       
        # Find shapes
        traffic.find_shapes(frame, frame) 
       
        cv2.imshow('Image', frame)
        key = cv2.waitKey(1)
        esc_key = 27
        if key == esc_key:
            break

    cap.release()
    cv2.destroyAllWindows() # to destroy all open windows

# open preacquired image
elif arg == 'file':
    print('Opening image ' + args.path)
    img = cv2.imread(args.path)
    if img is None:
        print('Image not found')
        quit()
    img_to_show = img.copy()

    # Find shapes
    traffic.find_shapes(img, img_to_show)

    # Show result
    cv2.imshow('Image', img_to_show)
    cv2.waitKey(0) # so that the window doesn't close right away
    cv2.destroyAllWindows() # to destroy all open windows
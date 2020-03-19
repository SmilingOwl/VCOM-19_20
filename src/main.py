import sys
import cv2
import image
import argparse

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

if img is None:
    print('Image not found')
    quit()

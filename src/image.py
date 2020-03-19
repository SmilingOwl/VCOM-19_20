import cv2

# Returns image from camera when Space is pressed, or None when Esc is pressed

def CaptureCameraImage():
    cap = cv2.VideoCapture(0)
    img = None
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        esc_key = 27
        if key == esc_key:
            break
    cap.release()
    cv2.destroyAllWindows()
    return img
    
# Reads image from input file
def ReadImageFile(path):
    img = cv2.imread(path)
    return img
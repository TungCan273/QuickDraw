#pylint: disable=E1101

import cv2
import numpy as np
from keras.models import load_model
import os
import cv2
import numpy as np
import torch
from LoadData import load_data
from PIL import Image


CLASSES = ['drums', 'alarmclock', 'apple', 'backpack', 'barn', 
               'bed', 'bowtie', 'candle', 'door', 'envelope', 
               'fish', 'guitar', 'icecream', 'mountain', 'star', 
               'tent', 'toothbrush', 'wristwatch']

WHITE_RGB=(255,255,255)

_, _, _, _, classes = load_data('data')
classes = [c.replace('full_numpy_bitmap_', ' ').replace(' ', '') for c in classes]
coords = []
clicked = False

def main():
    # Load model
    model = load_model("model/QuickDraw.h5")
    image = np.zeros((480, 640, 3), dtype=np.float32)
    cv2.namedWindow("Canvas")
    global ix, iy, is_drawing
    is_drawing = False

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing == True:
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y
    
    cv2.setMouseCallback('Canvas', paint_draw)
    """ 
   while (1):
        cv2.imshow('Canvas', 255-image)
        key = cv2.waitKey(10)
        if key == 27:
            #image = cv2.resize(image, (28,28)) 
            cv2.imwrite("painted.png", image)
            break
    """
    check_point = False
    while (1):
        cv2.imshow('Canvas', 255-image)
        key = cv2.waitKey(1)
        #frame = np.zeros((480,640,3), dtype=np.float32)
        #cv2.imshow("Draw", image)
        if key == 27:
            cv2.imwrite("painted.png", image)
            break
        elif key == ord('c'):
            pass 
        elif key == ord('s'):
            cv2.imwrite("painted.png", image)
            del_contour()
            IMG()

    cv2.destroyAllWindows()

def IMG():
    model = load_model("model/QuickDraw.h5")
    convert_8bit()
    image = cv2.imread("painted.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28,28))
    image = np.array(image, dtype=np.float32)[None, :, :]
    logits = model.predict(image)[0]
    ans = classes[np.argmax(logits)]
    print(ans)
    
    frame = np.ones((480,480,3))
    conf = int(max(logits)*10000)/100
    print("Accuary: {}%".format(comf))
    print(ans)
    
    if conf > 98.0:
        cv2.putText(frame, "You are drawing " + ans, (10,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
    else:
        cv2.putText(frame, "Maybe " + ans, (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
    print(ans, conf)
    cv2.imshow("Draw", frame)
    cv2.waitKey(0)

def convert_8bit():
    """
    Convert_8bit: String, String -> void.
    Converts the input image file into 8bit depth.
    im = Image.open(src)
    if not im.mode == "P":
        im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
        im2.save(dest)
        """

    img_path = "painted.png"
    img = Image.open(img_path)
    # Convert the image to 8-bit mode
    quantized = img.quantize(colors=256)
    # Save the quantized image
    quantized.save("painted.png")

def del_contour():
    # Read the image and convert it to grayscale
    image = cv2.imread('painted.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to convert the image to black and white
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find the contours of the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to remove the excess white space
    image = image[y:y+h, x:x+w]

    # Save the image and resize this from (480,640) to (28,28)
    image = cv2.resize(image,(28,28))
    cv2.imwrite('painted.png', image)

if __name__ == '__main__':
    main()
    #convert_8bit()
    #del_contour()
    #IMG()


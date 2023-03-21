#pylint: disable=E1101

import cv2
import numpy as np
from keras.models import load_model
import os
from PIL import ImageFont, ImageDraw, Image
import torch

drawing = False  # true if mouse is pressed
model = load_model('model/QuickDraw.h5')


def main():
    emojis = get_emojis()
    
    # Initialize deque for storing detected points and canvas for drawing 
    point = deque(maxlen = 512)
    image = np.zeros((480,640,3), dtype=np.uint8)
     
    # mouse callback function
    def paint_draw(event, x, y, flags, param):
        global ix, iy, drawing, mode
        color = (255, 255, 255)
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(image, (ix, iy), (x, y),color, 5)
                ix = x
                iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(image, (ix, iy), (x, y), color, 5)
            ix = x
            iy = y
        return x, y


    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Canvas', paint_draw)

    while (1):
        cv2.imshow('Canvas', 255-image)
        k = cv2.waitKey(10) & 0xFF
        if k == ord(" "):  
            #cv2.imwrite("painted_image.jpg", image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(image)
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(ys)
            max_x = np.max(xs)
            image = image[min_y:max_y, min_x:max_x]

            image = cv2.resize(image, (28,28))
            image = np.array(image,dtype=np.float32)[None,None,:,:]
            image = torch.from_numpy(image)
            logits = model(image)
            print(torch.max(logits[0]))
            image = overlay(image, emojis[pred_class], 400, 250, 100, 100)
        if k == 27: # Escape KEY and stop when use ESC
            break

def get_emojis():
    folder_emo = 'lb/'
    emojis = []
    for emoji in range(len(os.listdir(folder_emo))):
        print("You are drawing "+emoji)
        emojis.append(cv2.imread(folder_emo+str(emoji) + '.npy', -1))
    return emojis

def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji,(w,h))
    try:
        image[y:y + h, x:x + w] = transparent(image[y:y + h, x:x + x], emoji)
    except:
        pass
    return image

def trainsparent(face_img, overlay_t_img):
     # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

#cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

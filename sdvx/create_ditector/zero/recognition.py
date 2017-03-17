import os
import dlib
import cv2
from scipy import ndimage
import pyocr, pyocr.builders
import sys
from PIL import Image
import numpy as np

tools = pyocr.get_available_tools()
if len(tools) == 0:
    sys.exit('Not Found OCR tools')


class PerCharacter(object):

    def cut(self, img, img_hist_x):
        chars = []
        in_char = False
        x_start = None
        last_x = None

        for x in range(len(img_hist_x)):
            if in_char:
                if img_hist_x[x] > 0:
                    continue
                else:
                    char = (x_start, x - 1)
                    if char[1] - char[0] > 2:
                        chars.append((x_start, x - 1))
                    in_char = False
            else:
                if img_hist_x[x] > 0:
                    x_start = x
                    in_char = True
                else:
                    continue

        return chars

    def cutmax(self, img, img_hist_y):
        max_char = None
        in_char = False
        y_start = None
        last_y = None
        threshold = img_hist_y.mean()
        print("threshold"  +str(threshold))

        for y in range(len(img_hist_y) + 1):
            if y == len(img_hist_y):
                hist = 0
            else:
                hist = img_hist_y[y]

            if in_char:
                if hist > threshold:
                    continue
                else:
                    char = (y_start, y - 1)
                    if char[1] - char[0] > 2:
                        if max_char == None or (char[1] - char[0] > max_char[1] - max_char[0]):
                            max_char = char
                        in_char = False
            else:
                if hist > threshold:
                    y_start = y
                    in_char = True
                else:
                    continue

        return max_char

    def __init__(self):
        pass


def cut_out(img, header_rect):
    WIDTH_RATIO = 1.92
    HEIGHT_RATIO = 1.85
    MARGIN_RATIO = 0.02

    # width & height of score block
    width = (header_rect.right() - header_rect.left()) * WIDTH_RATIO
    height = (header_rect.bottom() - header_rect.top()) * HEIGHT_RATIO
    # margin
    m_width = width * MARGIN_RATIO
    m_height = height * MARGIN_RATIO
    # calc score block
    left = int(header_rect.left())
    right = int(left + width + m_width*2)
    top = int(header_rect.bottom())
    bottom = int(top + height + m_height*2)

    print("H_top" + str(header_rect.top()))
    print("H_bottom" + str(header_rect.bottom()))
    print("H_left" + str(header_rect.left()))
    print("H_right" + str(header_rect.right()))

    print("top" + str(top))
    print("bottom" + str(bottom))
    print("left" + str(left))
    print("right" + str(right))

    ret_img = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
    binalize(ret_img[top:bottom, left:right])
    return ret_img

def NOT_WHITE_STRING(img_hsv):
    # pixelValue = img_hsv[20, 20]
    # print('pixelValue = ' + str(pixelValue))
    white_mask_s = cv2.inRange(img_hsv[:, :, 1], 0, 200)
    white_mask_v = cv2.inRange(img_hsv[:, :, 2], 160, 256)
    white_mask = white_mask_s & white_mask_v
    not_white_mask = cv2.bitwise_not(white_mask)
    return not_white_mask

def binalize(img):
    # clear B
    r = img.copy()

    # r_blur = cv2.GaussianBlur(r,(3,3),0)
    not_white = NOT_WHITE_STRING(cv2.cvtColor(r, cv2.COLOR_BGR2HSV))

    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    blur = gray

    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,21,0)

    kernel = np.ones((3,3),np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th_adapt = cv2.morphologyEx(th_adapt, cv2.MORPH_OPEN, kernel)
    not_white = cv2.morphologyEx(not_white, cv2.MORPH_CLOSE, kernel)

    total = th & not_white
    cv2.imshow("th", th)
    cv2.imshow("not_white", not_white)
    cv2.imshow("total", total)
    return

    # Detect each character box
    pc = PerCharacter()
    hist_y = np.sum(total / 255, axis=1)
    box_y = pc.cutmax(total, hist_y)

    height, width = img.shape[:2]
    number_img = total[box_y[0]:box_y[1], 0:width]
    cv2.imshow("numbers",number_img)

    image, contours, hierarchy = cv2.findContours(total,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    bgr_total = cv2.cvtColor(total, cv2.COLOR_GRAY2BGR)
    contourImg = bgr_total.copy()

    color = np.random.randint(0,255,(100,3))
    cv2.drawContours(contourImg, contours, -1, (0,255,0), 3)
    cv2.imshow("contour", contourImg)

    approxes = []
    for contour in contours:
        epsilon = 0.01*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        area = cv2.contourArea(approx)

        if(area > 20):
            approxes.append(approx)
            x,y,w,h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            print("aspect"  +str(aspect_ratio))
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                cv2.rectangle(bgr_total,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.imshow("approx", bgr_total)
                # cv2.waitKey(0)


    cv2.imshow("approx", bgr_total)


    # hist_x = np.sum(number_img / 255, axis=0)
    # ch_boxes_x = pc.cut(th, hist_x)
    # bgr_th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    # for xr in ch_boxes_x:
    #     cv2.rectangle(bgr_th, (xr[0], box_y[0]), (xr[1], box_y[1]), (0, 0, 255), 2)

    cv2.imshow("binalize",th)
    cv2.imshow("adapt",th_adapt)

    # Opencv Mat -> PIL Image
    PIL_th=Image.fromarray(th)
    PIL_adapt=Image.fromarray(th_adapt)

    text = tools[0].image_to_string(
            PIL_th,
            builder=pyocr.builders.DigitBuilder()
        )
    text = text.encode('utf_8').decode('utf_8')
    print("th : " + text)

    text = tools[0].image_to_string(
            PIL_adapt,
            builder=pyocr.builders.DigitBuilder()
        )
    text = text.encode('utf_8').decode('utf_8')
    print("adapt : " + text)

    return 

def detection():
    IMG_DIR = './experiment_img/'
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    detector = dlib.simple_object_detector("detector.svm")
    files = os.listdir(IMG_DIR)

    for file in files:
        ftitle, fext = os.path.splitext(file)
        if fext != '.jpg':
            continue

        # Load image
        abspath = os.path.abspath(IMG_DIR + file)
        print(abspath)
        img = cv2.imread(abspath)
        if img is None:
            print('Failed to read image')
            break

        dets = detector(img)
        if len(dets) != 0 :
            for d in dets:
                # cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
                img = cut_out(img, d)

            # Show image
            cv2.imshow("img",img)
            cv2.waitKey(0)
            continue

        # Failed to Detecting -> rotate
        rIntr = 15
        rStart = -30
        rEnd = 30
        for r in range(rStart, rEnd+1, rIntr):
            rotate_img = ndimage.rotate(img, r)
            dets = detector(rotate_img)
            if len(dets) != 0 :
                for d in dets:
                    # cv2.rectangle(rotate_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
                    rotate_img = cut_out(rotate_img, d)
                # Show image
                cv2.imshow("img",rotate_img)
                cv2.waitKey(0)
                break
    # end for
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    detection()

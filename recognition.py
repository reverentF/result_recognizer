import os
import dlib
import cv2
from scipy import ndimage
import pyocr, pyocr.builders
import sys
from PIL import Image
import numpy as np
import math
import time
import random

tools = pyocr.get_available_tools()
if len(tools) == 0:
    sys.exit('Not Found OCR tools')

class PerCharacter(object):

    def divideByXHist(self, img):
        chars = []
        in_char = False
        x_start = None

        hist_x = np.sum(img / 255, axis=0)
        threshold = int(hist_x.mean() * 0.30)
        # threshold = int(hist_x.mean() * 0.50)

        for x in range(len(hist_x) + 1):
            if x == len(hist_x):
                hist = 0
            else:
                hist = hist_x[x]

            if in_char:
                if hist > threshold:
                    continue
                else:
                    char = (x_start, x - 1)
                    if char[1] - char[0] > 2:
                        chars.append((x_start, x - 1))
                    in_char = False
            else:
                if hist > threshold:
                    x_start = x
                    in_char = True
                else:
                    continue

        return chars

    def cutoutMaxYRange(self, img, threshold_rate=0.10):
        # Y軸方向ヒストグラムを基に領域を分割 -> 最大部分を返す
        max_char = None
        in_char = False
        y_start = None
        AS_NOISE = 5 # この値以下の領域はノイズとして処理

        hist_y = np.sum(img / 255, axis=1)
        threshold = hist_y.mean() * threshold_rate

        for y in range(len(hist_y) + 1):
            if y == len(hist_y):
                # 画像の下端に到達
                hist = 0
            else:
                hist = hist_y[y]

            if in_char:
                if hist > threshold:
                    continue
                else:
                    char = (y_start, y - 1)
                    if char[1] - char[0] > AS_NOISE:
                        # ノイズより大きければ文字領域として確定
                        if max_char is None :
                            max_char = char
                        elif char[0] - max_char[1] < AS_NOISE:
                            #ノイズで文字領域が分断されたとき対応
                            max_char = (max_char[0], char[1])
                        elif char[1] - char[0] > max_char[1] - max_char[0]:
                            max_char = char
                        in_char = False
            else:
                if hist > threshold:
                    y_start = y
                    in_char = True
                else:
                    continue
        if max_char is None:
            return max_char, img
        return max_char, img[max_char[0]:max_char[1]+1, :]

    def cutBlackRange(self, img, from_top=True, threshold_rate = 0.0):
        # 上端または下端の何もない(黒い)領域を削除

        hist_y = np.sum(img / 255, axis=1)
        threshold = hist_y.mean() * threshold_rate
        height = img.shape[0]

        for i in range(height):
            if from_top:
                y = i
            else:
                y = (height-1) - i

            hist = hist_y[y]
            if hist <= threshold:
                continue
            else:
                if from_top:
                    return img[y:height, :]
                else:
                    return img[0:y+1, :]

        return img

    # 画像上部の不要な枠部分の削除
    def deleteTopFrame(self, img, hist_y):
        end_frame_y = 0
        threshold = img_hist_y.mean() * 0.3

        for y in range(len(img_hist_y)):
            if(img_hist_y[y] < threshold):
                end_frame_y = y
                break

        height = img.shape[0]
        if end_frame_y >= height:
            return img
        else:
            return img[end_frame_y:height, :]

    # 画像左部の不要な枠部分の削除
    def deleteLeftFrame(self, img, img_hist_x):
        end_frame_x = 0
        threshold = img_hist_x.mean() * 0.3

        for x in range(len(img_hist_x)):
            if(img_hist_x[x] < threshold):
                end_frame_x = x
                break

        width = img.shape[1]
        if end_frame_x >= width:
            return img
        else:
            return img[:, end_frame_x:width]

    def __init__(self):
        pass

def train():
    print("Training Starts . . .")
    TRAIN_DATA_DIR = "./training/number/"

    samples = None
    labels = []
    # for 0 - 9
    for character_i in range(0, 10):
        img_dir = TRAIN_DATA_DIR + str(character_i) + "/"
        files = os.listdir(img_dir)
        for file in files:
            ftitle, fext = os.path.splitext(file)
            if fext != '.jpg' and fext != '.png':
                continue

            # Load image
            abspath = os.path.abspath(img_dir + file)
            img = cv2.imread(abspath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print('Failed to read image')
                break

            sample = img.reshape((1, -1))
            label = ord('0') + int(character_i)

            if samples is None:
                samples = np.empty((0, img.shape[0] * img.shape[1]))
            samples = np.append(samples, sample, 0).astype(np.float32)
            labels.append(label)

    labels = np.array(labels, np.float32)
    labels = labels.reshape((labels.size, 1)).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
    print("complete.")

    return knn

def detection():
    # IMG_DIR = './img/result/train/'
    # IMG_DIR = './img/result/test/'
    IMG_DIR = './img/eamu/'
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)

    # score
    detector_s = Detector("detector_score.svm", 2.0, 2.5) # 1.92
    # hi_score
    detector_h = Detector("detector_hiscore.svm", 2.0, 2.5) # 1.40

    # knn
    knn = train()

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

        # Correct rotation
        arg = get_degree(img)
        rotate_img = ndimage.rotate(img, arg)

        character_imgs = detectCharacters(detector_s, rotate_img)
        if len(character_imgs) == 0:
            character_imgs = detectCharacters(detector_h, rotate_img)
        if len(character_imgs) == 0:
            print("Failed to Detect Characters.")
            continue

        if 0:
            # 学習データ収集
            retval = 0
            for ch_img in character_imgs:
                SAVE_DIR = "training/number/raw"
                # SAVE_DIR = "training/number_adaptive/raw"
                cv2.imwrite('%s/%s.%s.png' %
                        (SAVE_DIR, retval, time.time()), ch_img)
            # cv2.waitKey(0)
        elif 1:
            # kNN
            ch_string=""
            for ch_img in character_imgs:
                sample = ch_img.reshape((1, ch_img.shape[0] * ch_img.shape[1]))
                sample = np.array(sample, np.float32)

                k = 3
                retval, results, neigh_resp, dists = knn.findNearest(sample, k)
                d = chr(int(results.ravel()))
                ch_string += str(d)
            print("result:" + ch_string)
            cv2.waitKey(0)
    # end for
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    detection()

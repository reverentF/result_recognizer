import cv2
from scipy import ndimage
import pyocr, pyocr.builders
import sys
from PIL import Image
import numpy as np

class UtilImage():
    # 適応的 モアレに弱い
    def binalizeByAdaptive(img):
        r = img.copy()

        # R, G値のみ取り出しグレースケール化
        green = r[:,:,1]
        red = r[:,:,2]
        redGreen = cv2.addWeighted(red, 0.5, green, 0.5, 0)

        # binalize
        th_red = cv2.adaptiveThreshold(redGreen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,21,25)
        
        # cleaning noise by opening
        kernel = np.ones((1,1),np.uint8)
        th_red = cv2.morphologyEx(th_red, cv2.MORPH_OPEN, kernel)

        cv2.imshow("binalize", th_red)
        # cv2.waitKey(0)

        return th_red

    # Pタイル法で2値化
    # P : 文字の面積 / 全体の面積 (文字の面積が全体に占める割合)
    def binalizeByPTile(img, p):
        l_img = img.copy()

        # R値のみ取り出しグレースケール化
        green = l_img[:,:,1]
        red = l_img[:,:,2]
        redGreen = cv2.addWeighted(red, 0.5, green, 0.5, 0)

        # ヒストグラム生成
        hist = cv2.calcHist([redGreen],[0],None,[256],[0,256])
        # P-タイル法
        height, width = redGreen.shape[:2]
        N = height*width # all
        sum_n = 0
        for index, val in enumerate(hist):
            sum_n += val
            if sum_n / N > p:
                ret,binalized_img = cv2.threshold(redGreen,index,255,cv2.THRESH_BINARY_INV)
                break

        kernel = np.ones((2,2),np.uint8)
        binalized_img = cv2.morphologyEx(binalized_img, cv2.MORPH_OPEN, kernel)

        return binalized_img

    def binalize(img):
        r = img.copy()

        # R, G値のみ取り出しグレースケール化
        green = r[:,:,1]
        red = r[:,:,2]
        redGreen = cv2.addWeighted(red, 0.5, green, 0.5, 0)

        # binalize
        ret,th_red = cv2.threshold(redGreen,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # cleaning noise by opening
        kernel = np.ones((2,2),np.uint8)
        th_red = cv2.morphologyEx(th_red, cv2.MORPH_OPEN, kernel)

        return th_red

    # 画像の傾き検出
    # @return 水平からの傾き角度
    def get_degree(img):
        # TODO : マジックナンバー
        l_img = img.copy()
        gray_image = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image,50,150,apertureSize = 3)
        minLineLength = 200
        maxLineGap = 30
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

        sum_arg = 0;
        count = 0;
        for line in lines:
            for x1,y1,x2,y2 in line:
                arg = math.degrees(math.atan2((y2-y1), (x2-x1)))
                HORIZONTAL = 0
                DIFF = 20 # 許容誤差 -> -20 - +20 を本来の水平線と考える
                if arg > HORIZONTAL - DIFF and arg < HORIZONTAL + DIFF : 
                    sum_arg += arg;
                    count += 1

        if count == 0:
            return HORIZONTAL
        else:
            return (sum_arg / count) - HORIZONTAL;

    def getMaxBlob(bin_img):
        # 一定面積以上のブロブを返す
        labelnum, labelimg, contours, gocs = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        if len(contours) == 1:
            return np.zeros(bin_img.shape[:2],np.uint8)
        # ラベル:0 は黒色領域
        contours_white = np.delete(contours, 0, axis=0)
        max_size = contours_white.max(axis=0)[4]
        # 平均より大きいブロブ
        # threshold = contours_white.mean(axis=0)[4] * 0.8
        threshold = max_size * 0.15
        # print(contours_white)
        # print(threshold)

        accept_labels = []
        for label in range(1, labelnum):
            x,y,w,h,size = contours[label]
            if size >= threshold:
                accept_labels.append(label)

        # サイズが閾値を超えたブロブのみ残した画像を返す
        result_img = np.zeros(bin_img.shape[:2],np.uint8)
        for label in accept_labels:
            result_img[labelimg == label] = 255

            return result_img

    def resize(character_img):
        RESIZED_WIDTH = 35
        RESIZED_HEIGHT = 30

        height, width = character_img.shape[:2]
        if height == 0 or width == 0:
            print("Invalid Image height(width) in resize()")
            return None
        else:
            resized_img = cv2.resize(character_img, (RESIZED_WIDTH, RESIZED_HEIGHT), 
                                                interpolation=cv2.INTER_NEAREST)
        return resized_im


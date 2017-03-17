import cv2
from scipy import ndimage
import pyocr, pyocr.builders
import sys
from PIL import Image
import numpy as np

class CharactorRecognizer():
    # 不要部分を削除し数字部分の画像を取り出す
    # return --
    #  number_img : 数字部分2値画像
    #  数字部分の座標値
    def cutoutNumberImg(gray_img):
        l_img = gray_img.copy()

        # 上下左右を白縁取り
        height, width = l_img.shape[:2]
        l_img[0:15, :] = 255
        l_img[:, 0] = 255
        l_img[:, width-1] = 255
        l_img[height-1, :] = 255

        # 白枠で縁取られた面積最大の領域を探す
        image, contours, hierarchy = cv2.findContours(l_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        contourImg = cv2.cvtColor(l_img, cv2.COLOR_GRAY2BGR)

        inner_contours = []
        for index, contour in enumerate(contours):
            if hierarchy[0][index][2] == -1:
                inner_contours.append(contour)

        approxes = []
        max_box = None
        for contour in inner_contours:
            # 矩形補完
            epsilon = 0.01*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            area = cv2.contourArea(approx)

            if max_box is None or cv2.contourArea(max_box) < cv2.contourArea(approx):
                max_box = approx
            if(area > 20):
                approxes.append(approx)
                x,y,w,h = cv2.boundingRect(contour)

        x,y,w,h = cv2.boundingRect(max_box)
        x += 1
        y += 1
        w -= 1
        h -= 1
        number_img = l_img[y:y+h, x:x+w]

        return number_img, x, y, w, h

    def cutoutCharactersByWatershed(gray_number_img):
        l_number_img = gray_number_img.copy()

        pc = PerCharacter()
        # Y軸方向ヒストグラム探索
        box_y, l_number_img = pc.cutoutMaxYRange(l_number_img)

        # watershed
        distance = cv2.distanceTransform(l_number_img, cv2.DIST_L2, 5)
        cv2.normalize(distance, distance,0,1,cv2.NORM_MINMAX)
        cv2.imshow("distance", distance)
        distance2 = distance * 255
        distance2 = np.uint8(distance2)
        
        ret,distance2 = cv2.threshold(distance2, 80, 255, cv2.THRESH_BINARY)
        # ret,distance2 = cv2.threshold(distance2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cleaning noise by opening
        kernel = np.ones((2,2),np.uint8)
        distance2 = cv2.morphologyEx(distance2, cv2.MORPH_OPEN, kernel, iterations = 1)
        cv2.imshow("Peaks", distance2);
        sure_fg = distance2 #getMaxBlob(distance2)
        cv2.imshow("Blobs", sure_fg);
        # Marker labelling
        unknown = cv2.subtract(l_number_img,sure_fg)
        nLabels, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        testimg = cv2.cvtColor(l_number_img, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(testimg,markers)
        colors = []
        for i in range(1, nLabels + 2):
            colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

        height, width = testimg.shape[:2]
        for y in range(0, height):
            for x in range(0, width):
                if markers[y, x] > 0:
                    testimg[y, x] = colors[markers[y, x]]
                else:
                    testimg[y, x] = [0, 0, 0]
        cv2.imshow("testimg", testimg)
        cv2.waitKey(0)
        return 

    # Detect and cutout each character imag
    def cutoutCharacters(gray_number_img):
        l_number_img = gray_number_img.copy()

        pc = PerCharacter()
        # Y軸方向ヒストグラム探索
        box_y, l_number_img = pc.cutoutMaxYRange(l_number_img)

        # Divide characters by x_histgram
        boxes_x = pc.divideByXHist(l_number_img)

        # アスペクト比から文字候補絞り込み
        img_height, img_width = l_number_img.shape[:2]
        character_boxes_x = []
        #watershed で分割に変えた方がよさそう
        for xr in boxes_x:
            aspect = (xr[1] - xr[0]) / img_height
            # cv2.rectangle(hist_img, (xr[0], 0), (xr[1], img_height), (255,0,0), 1)
            if aspect > 0.5:
                # box(xstart, xend)
                character_boxes_x.append(xr)

        # for debug
        hist_img = cv2.cvtColor(l_number_img, cv2.COLOR_GRAY2BGR)

        # 文字候補から外れた文字を補完
        FONT_SIZE_RATE = 0.83
        character_boxes = []
        arr_width = []
        arr_margin = []

        for index, xr in enumerate(character_boxes_x):
            if len(character_boxes) >= 8:
                break
            if index < len(character_boxes_x) - 1:
                next_xs = character_boxes_x[index+1][0]
            else:
                next_xs = img_width

            # 注目boxは文字として確定
            ch_box = (xr[0], 0, xr[1], img_height) # xs, ys, xe, ye
            character_boxes.append(ch_box)
            cv2.rectangle(hist_img, (ch_box[0], ch_box[1]), (ch_box[2], ch_box[3]), (0,0,255), 1)

            xs = xr[0]
            xe = xr[1]
            w = xe - xs
            between = (next_xs - xe)

            arr_width.append(w)
            if between < w:
                arr_margin.append(between)
            else:
                if len(arr_margin) > 0:
                    arr_margin.append(int(np.average(arr_margin)))
                else:
                    # 初回かつbetween < w
                    # o x x x x ... o のとき
                    arr_margin.append(int(w * 0.2))

            ave_w = int(np.average(arr_width))
            ave_m = int(np.average(arr_margin))

            # フォントサイズ変化対応
            if(len(character_boxes) == 5):
                ave_w = int(ave_w * FONT_SIZE_RATE)
            elif(len(character_boxes) > 5):
                ave_w  = int((np.average(arr_width[:5]) * FONT_SIZE_RATE + np.average(arr_width[5:]))/2)

            if(len(character_boxes) == 6):
                ave_m = int(ave_m * FONT_SIZE_RATE)
            elif(len(character_boxes) > 6):
                ave_m  = int((np.average(arr_margin[:6]) * FONT_SIZE_RATE + np.average(arr_margin[6:]))/2)

            while len(character_boxes) < 8 and between > ave_w:
                # 次のboxとの間に抜けがないか調べる
                ch_box = (ave_m + xe, 0, ave_m + xe + ave_w, img_height) # xs, ys, xe, ye
                character_boxes.append(ch_box) # xs, ys, xe, ye
                cv2.rectangle(hist_img, (ch_box[0], ch_box[1]), (ch_box[2], ch_box[3]), (255,0,0), 1)
                xe += ave_m + ave_w
                between -= ave_w
                arr_width.append(ave_w)
                arr_margin.append(ave_m)
                if(len(character_boxes) == 5):
                    ave_w  = int(ave_w * FONT_SIZE_RATE)
                if(len(character_boxes) == 6):
                    ave_m  = int(ave_m * FONT_SIZE_RATE)

        cv2.imshow("hpist", hist_img)

        character_imgs = []
        for xs, ys, xe, ye in character_boxes:
            character_imgs.append(l_number_img[ys:ye, xs:xe].copy())

        return character_imgs, character_boxes

    def detectCharacters(detector:Detector, img):
        character_imgs = []

        dets = detector.detectHeader(img)
        if len(dets) == 0:
            return character_imgs
        else:
            retval = 0
            for d in dets:
                score_characters = detector.cutoutContents(img, d)
                bin_characters = binalize(score_characters)
                # bin_characters = binalizeByAdaptive(score_characters)
                number_img, x, y, w, h = cutoutNumberImg(bin_characters)
                characters, ch_boxes = cutoutCharacters(number_img)
                cutoutCharactersByWatershed(number_img)
                return character_imgs
                cv2.imshow("threshold", number_img)

                # ----- test_adaptive
                bgr_number_img = score_characters[y:y+h, x:x+w]
                clear_bin_number_img = binalizeByAdaptive(bgr_number_img)
                # clear_bin_number_img = getMaxBlob(clear_bin_number_img)
                cv2.imshow("adaptive", clear_bin_number_img)
                kernel = np.ones((2,2),np.uint8)
                mask = cv2.morphologyEx(number_img, cv2.MORPH_GRADIENT, kernel)

                # 検出結果をもとにadaptiveの方をカット
                clear_characters = []
                height, width = clear_bin_number_img.shape[:2]
                for xs, ys, xe, ye in ch_boxes:
                    clear_characters.append(clear_bin_number_img[0:height, xs:xe].copy())

                # characters = clear_characters # test
                # -----

                pc = PerCharacter()
                for character in characters:
                    ch_img = getMaxBlob(character)
                    box_y, ch_img = pc.cutoutMaxYRange(ch_img, threshold_rate=0.00)
                    ch_img = resize(ch_img)
                    if ch_img is not None:
                        character_imgs.append(ch_img)

                for i in range(len(character_imgs)):
                    # cv2.imshow("ci"+str(i), character_imgs[i])
                    if character_imgs[i] is None:
                        print("None")
                    else:
                        print("Not None")
                cv2.waitKey(0)

        return character_imgs


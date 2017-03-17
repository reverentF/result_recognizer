import dlib
import numpy as np

# dlibで作成したditectorを基に画像中から指定領域を切り抜く
class Detector(object):
    def __init__(self, detector_file_name:str, width_rate:float, height_rate:float):
        """
        @params
            ditector_file_name : dlibで作成した検出器へのパス
            box_size_rate(width, height) : headerのwidth/heightを1としたときのcontentsのboxの比率
            relative_pos(x,y) : headerのboxの左上座標を(0,0)としたときのcontentsの相対位置
                                -> contentsが上下左右どこかで基準にすべき点が変わるのでどうしよう
        """
        self.detector_file_name  = detector_file_name
        self.width_rate  = width_rate
        self.height_rate  = height_rate
        self.detector = dlib.simple_object_detector(detector_file_name)
        pass


    def getContents(self, img):
        """
        @return boolean succeed_flg, np.array content_img
        """
        dets = self.__detectHeader(img)
        if len(dets) == 0:
            # header is not found
            return False, np.zeros((1,1), np.uint8)
        else:
            content_imgs = []
            for d in dets:
                content_img = self.__cutoutContents(img, d)
                content_imgs.append(content_img)
            # TODO : ヘッダが複数見つかった場合、最もそれらしいものを返すようにする
            #      : 現状は最初にみつかったもの
            return content_imgs[0]

    # ------- private functions -------

    def __detectHeader(self, img):
        """
            detectorで示されるheaderを画像中から検出
        """
        dets = self.detector(img)
        return dets

    # TODO refact
    def __cutoutContents(self, img, header_rect):
        """
            headerを手掛かりにしてcontentsを探す
        """
        WIDTH_RATIO = self.width_rate
        HEIGHT_RATIO = self.height_rate
        MARGIN_RATIO = 0.02

        # TODO : 様々な場合に対応できるように位置を可変にする 

        # width & height of score block
        width = (header_rect.right() - header_rect.left()) * WIDTH_RATIO
        height = (header_rect.bottom() - header_rect.top()) * HEIGHT_RATIO
        # margin
        m_width = width * MARGIN_RATIO
        m_height = height * MARGIN_RATIO
        # calc score block
        left = int(header_rect.left() + m_width)
        right = int(left + width)
        top = int(header_rect.bottom())
        bottom = int(top + height)

        ret_img = img[top:bottom, left:right].copy()

        return ret_img
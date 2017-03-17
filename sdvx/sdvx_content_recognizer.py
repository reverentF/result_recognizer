import result_recognizer
import detector
import util_image

class SDVXResultRecognizer(ResultRecognizer):
    def __init__:
        pass

    # TODO : 親クラスのメソッドにする
    def recognize(self, img, test=false):
        self.createDetectors()
        
        for detector in self.detectors:
            # 読み取り対象画像の切り抜き
            content_img = detector.getContents
            # 2値化
            # 個別の文字に分割
            # 認識

        if test:
            self.saveCutoutImages()
            return
        else:
            #json形式で返すイメージ
            return result
        pass

    def createDetectors(self):
        # 
        self.detectors['score'] = Detector("./ditectors/detector_score.svm", 2.0, 2.5)
        self.detectors['hiscore'] = Detector("./ditectors/detector_hiscore.svm", 2.0, 2.5)
        # self.detectors['music_nm'] = hoge
        pass

    def binalize(self, img):
        return UtilImage.binalize(img)

    def 



if __name__ == '__main__':
    recognizer = SDVXResultRecognizer()
    recognizer.recognize()

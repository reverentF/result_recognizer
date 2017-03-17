import os
import dlib
import cv2
from scipy import ndimage

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
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
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
                cv2.rectangle(rotate_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
                    # Show image
            cv2.imshow("img",rotate_img)
            cv2.waitKey(0)
            break

cv2.destroyAllWindows()
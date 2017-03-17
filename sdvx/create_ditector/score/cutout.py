import os
import dlib
import cv2
from scipy import ndimage


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
    left = int(header_rect.left() - m_width)
    right = int(left + width + m_width*2)
    top = int(header_rect.bottom() - m_height)
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
    return ret_img

# cut_out()

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
                cv2.rectangle(rotate_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
                rotate_img = cut_out(rotate_img, d)
            # Show image
            cv2.imshow("img",rotate_img)
            cv2.waitKey(0)
            break
# end for
cv2.destroyAllWindows()

if __name__ == '__main__':
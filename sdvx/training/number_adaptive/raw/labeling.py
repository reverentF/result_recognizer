import os
import cv2
import sys
import shutil

if __name__ == '__main__':
    IMG_DIR = './'
    DST_DIR = '../'
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)

    files = os.listdir(IMG_DIR)
    beforepath = None
    for file in files:
        ftitle, fext = os.path.splitext(file)
        if fext != '.jpg' and fext != '.png':
            continue

        # Load image
        abspath = os.path.abspath(IMG_DIR + file)
        print(abspath)
        img = cv2.imread(abspath)
        if img is None:
            print('Failed to read image')
            break

        while 1:
            cv2.imshow("img", img)
            key = cv2.waitKey(0)
            ch = key - ord('0')
            if ch >= 0 and ch <= 9:
                tmppath = DST_DIR + "/" + str(ch) + "/" + file
                shutil.move(abspath, tmppath)
                beforepath = tmppath
                break
            elif key == ord('b'):
                # back
                beforefile = os.path.basename(beforepath)
                targetpath = os.path.abspath(beforepath)
                shutil.move(targetpath,os.path.abspath(IMG_DIR + beforefile))
                break
            elif key == ord('d'):
                # delete
                tmppath = DST_DIR + "/del/" + file
                shutil.move(abspath, tmppath)
                beforepath = tmppath
                break
            else:
                print("input is 0-9")
        # end while
    # end for
    cv2.destroyAllWindows()
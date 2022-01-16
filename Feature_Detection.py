import cv2
import numpy as np


def feature_detection(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

    corners = cv2.dilate(corners, None)

    show_img = np.copy(img)
    show_img[corners>0.1*corners.max()]=[0,0,255]

    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

    #_show_img = cv2.resize(show_img, dsize = (0,0), fx=0.6, fy=0.6, interpolation = cv2.INTER_LINEAR)
    
    cv2.imshow('Harris corner detector', show_img )
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

    _image = cv2.imread(file, 0)
    cv2.GaussianBlur(_image, (3,3), 0)
    work_image = cv2.Canny(_image, 50, 80)
    #_work_image = cv2.resize(work_image, dsize = (0,0), fx=0.6, fy=0.6, interpolation = cv2.INTER_LINEAR)

    cv2.imshow("Work", work_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

feature_detection('./stitching/s1.jpg')
feature_detection('./stitching/boat1.jpg')
feature_detection('./stitching/budapest1.jpg')
feature_detection('./stitching/newspaper1.jpg')



import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb(img):
    orb =  cv2.ORB_create()
    orb.setMaxFeatures(200)

    keyPoints = orb.detect(img, None)
    keyPoints, descrptors = orb.compute(img, keyPoints)

    show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("ORB descriptors", show_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def ransac(img0, img1):
    detector = cv2.ORB_create(100)
    kps0, fea0 = detector.detectAndCompute(img0, None)
    kps1, fea1 = detector.detectAndCompute(img1, None)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
    matches = matcher.match(fea0, fea1)

    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

    plt.figure()
    plt.subplot(211)
    plt.axis('off')
    plt.title('all matches')
    dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
    plt.imshow(dbg_img[:,:, [2,1,0]])
    plt.subplot(212)
    plt.axis('off')
    plt.title('filtered matches')
    dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, [m for i, m in enumerate(matches) if mask[i]], None)
    plt.imshow(dbg_img[:, :, [2, 1, 0]])
    plt.tight_layout()
    plt.show()

def warping(img):
    height, width, channel = img.shape

    srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
    dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    dst = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

img0 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('./stitching/newspaper2.jpg', cv2.IMREAD_COLOR)

orb(img0)
orb(img1)
ransac(img0, img1)
warping(img0)



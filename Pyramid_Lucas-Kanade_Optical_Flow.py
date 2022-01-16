import cv2

img0 = cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_GRAYSCALE)

pts = cv2.goodFeaturesToTrack(img0, 500, 0.01, 5)
#pts = pts.reshape(-1, 1, 2)

pt2, status, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, pts, None, winSize=(15, 15), maxLevel=5, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03)
        )

dst = cv2.addWeighted(img0, 0.3, img1, 0.3, 0)

good_pts = pts[status == 1]

for i in range(pt2.shape[0]):
    if status[i,0] == 0: 
        continue
        
    cv2.circle(dst, tuple(pts[i, 0]), 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(dst, tuple(pt2[i, 0]), 4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(dst, tuple(pts[i, 0]), tuple(pt2[i, 0]), (0, 255, 0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()


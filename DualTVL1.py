import cv2
import numpy as np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # 입력 영상의 컬러 영상 변환
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 직선 그리기
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)
    
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        
    return vis

img0 = cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_GRAYSCALE)

optical_flow = cv2.DualTVL1OpticalFlow_create()
flow = optical_flow.calc(img0, img1, None)

dst = cv2.addWeighted(img0, 0.3, img1, 0.3, 0)

cv2.imshow('dst', draw_flow(dst, flow))
cv2.waitKey()
cv2.destroyAllWindows()



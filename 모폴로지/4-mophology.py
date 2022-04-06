import cv2
import numpy as np

image = cv2.imread('image_Peppers512rgb.png',cv2.IMREAD_GRAYSCALE) 

#이진화 진행
th, threshold_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

#3x3 매트릭스 생성
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], np.uint8)


#Erosion, Dilation, Opening, Closing  1회씩 실행
dilation_image = cv2.dilate(threshold_image, kernel, iterations=1) 
erosion_image = cv2.erode(threshold_image, kernel, iterations=1)
opening_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)
closing_image =  cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

cv2.imshow("threshold", threshold_image)
cv2.imshow("Dilation", dilation_image)
cv2.imshow("Erosion", erosion_image)
cv2.imshow("Opening", opening_image)
cv2.imshow("Closing", closing_image)

cv2.waitKey()
cv2.destroyAllWindows()

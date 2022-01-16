#필요한모듈 import
import cv2
import numpy as np
import matplotlib.pyplot as plt

#이미지를 그레이스케일로 불러오기
grey = cv2.imread('Lena.png',0)
cv2.imshow('original grey', grey)
cv2.waitKey()

hist, bins = np.histogram(grey, 256, [0,255])
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

#그레이스케일 이미지 히스토그램 평탄화
grey_eq = cv2.equalizeHist(grey)

#평탄화 된 이미지의 히스토그램 계산, 그래프표시
hist, bins = np.histogram(grey_eq, 256, [0,255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

#평탄화된 이미지 화면표시
cv2.imshow('equalized grey', grey_eq)
cv2.waitKey()

#BGR로 이미지 불러와서 HSV 색 공간으로 변환
color = cv2.imread('Lena.png')
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

#V 채널 평탄화 후 다시 BGR 색 공간으로 변화
hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('original color', color)

#평탄화된 풀컬러 이미지 화면 표
cv2.imshow('equalized color', color_eq)
cv2.waitKey()
cv2.destroyAllWindows()


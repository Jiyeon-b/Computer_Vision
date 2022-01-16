#필요한모듈 import
import cv2
import numpy as np
import matplotlib.pyplot as plt

#이미지를 불러오고 [0,1]범위의 부동소수점 데이터 타입으로 변환
image = cv2.imread('Lena.png').astype(np.float32)/255

#각 픽셀에 임의의 값을 더해 노이즈를 생성하고 화면에 표시
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0,1)
plt.imshow(noised[:,:,[2,1,0]])
plt.xlabel('noised')
plt.show()

#노이즈가 있는 이미지에 양방향 필터 적용 1
bilat = cv2.bilateralFilter(noised, -1, 0.3, 10) 
#입력이미지, 윈도우 크기diameter, 색상SigmaColor, 공간시그마SigmaSpace 값을 매개변수로 받음
#윈도우 크기가 음수면 공간시그마 값을 사용해 계산
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaColor 0.3')
plt.show()

#노이즈가 있는 이미지에 양방향 필터 적용 2
bilat = cv2.bilateralFilter(noised, -1, 0.5, 10) 
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaColor 0.5')
plt.show()


#노이즈가 있는 이미지에 양방향 필터 적용 3
bilat = cv2.bilateralFilter(noised, -1, 2, 10) 
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaColor 2')
plt.show()

#노이즈가 있는 이미지에 양방향 필터 적용 4
bilat = cv2.bilateralFilter(noised, -1, 0.3, 15)
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaSpace 15')
plt.show()

#노이즈가 있는 이미지에 양방향 필터 적용 5
bilat = cv2.bilateralFilter(noised, -1, 0.5, 20) 
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaSpace 20')
plt.show()


#노이즈가 있는 이미지에 양방향 필터 적용 6
bilat = cv2.bilateralFilter(noised, -1, 2, 40) 
plt.imshow(bilat[:, :, [2,1,0]])
plt.xlabel('bilateral Filter SigmaSpace 40')
plt.show()

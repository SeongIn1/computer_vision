import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 그레이스케일 변환 [cite: 15]
# cv.imread()를 사용하여 이미지를 불러옵니다. [cite: 19]
img1 = cv.imread('edgeDetectionImage.jpg') 

# cv.cvtColor()를 사용하여 컬러 이미지를 흑백(그레이스케일)으로 변환합니다. [cite: 20]
gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 

# 2. Sobel 에지 검출 [cite: 16]
# cv.Sobel()을 사용하여 x축 방향의 에지를 검출합니다. 데이터 타입은 cv.CV_64F, x방향 1, y방향 0입니다. [cite: 21]
# 힌트에 따라 ksize는 3으로 설정했습니다. [cite: 27]
sobel_x = cv.Sobel(gray_img1, cv.CV_64F, 1, 0, ksize=3) 

# cv.Sobel()을 사용하여 y축 방향의 에지를 검출합니다. 데이터 타입은 cv.CV_64F, x방향 0, y방향 1입니다. [cite: 21]
# 마찬가지로 ksize는 3으로 설정했습니다. [cite: 27]
sobel_y = cv.Sobel(gray_img1, cv.CV_64F, 0, 1, ksize=3) 

# 3. 에지 강도 계산 및 형변환
# cv.magnitude()를 사용하여 x축과 y축 결과를 합친 에지 강도를 계산합니다. [cite: 22]
magnitude = cv.magnitude(sobel_x, sobel_y) 

# cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 화면에 표시할 수 있는 uint8 형식으로 변환합니다. [cite: 28]
magnitude_uint8 = cv.convertScaleAbs(magnitude) 

# 4. 시각화
# Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화합니다. [cite: 23]
plt.figure(figsize=(10, 5)) 

# 첫 번째 영역: 원본 이미지 시각화
plt.subplot(1, 2, 1) 
# OpenCV는 BGR을 사용하므로, Matplotlib에서 올바른 색상으로 보려면 RGB로 변환해야 합니다.
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB)) 
plt.title('Original Image') 
plt.axis('off') # 축 눈금 숨김

# 두 번째 영역: 검출된 에지 강도 이미지 시각화
plt.subplot(1, 2, 2) 
# plt.imshow()에서 cmap='gray'를 사용하여 흑백으로 시각화합니다. [cite: 29]
plt.imshow(magnitude_uint8, cmap='gray') 
plt.title('Sobel Edge Magnitude') 
plt.axis('off') # 축 눈금 숨김

plt.tight_layout() # 이미지 겹침 방지
plt.show() # 결과물 창 띄우기
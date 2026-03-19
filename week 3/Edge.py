import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 복사본 생성
# cv.imread()를 사용하여 원본 이미지(dabo.jpg)를 불러옵니다.
img2 = cv.imread('dabo.jpg') 

# 검출된 직선을 원본 이미지에 그리기 위해 복사본을 생성합니다. [cite: 42, 46]
result_img2 = img2.copy() 

# 2. Canny 에지 검출
# cv.Canny()를 사용하여 에지 맵을 생성합니다. [cite: 44]
# 힌트에 따라 threshold1은 100, threshold2는 200으로 설정합니다. [cite: 49]
edges = cv.Canny(img2, 100, 200) 

# 3. 허프 변환을 이용한 직선 검출
# cv.HoughLinesP()를 사용하여 에지 맵에서 직선을 검출합니다. [cite: 41, 45]
# 힌트에 따라 rho, theta, threshold, minLineLength, maxLineGap 값을 설정합니다. (직선 검출 성능 개선을 위해 이미지를 보며 수치 조정이 필요할 수 있습니다.) [cite: 50]
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10) 

# 4. 검출된 직선 원본 이미지에 그리기
# 직선이 하나라도 검출된 경우에만 그리기 작업을 수행합니다.
if lines is not None: 
    # 검출된 모든 직선의 배열에 대해 반복합니다.
    for line in lines: 
        # 각 직선의 시작점(x1, y1)과 끝점(x2, y2) 좌표를 추출합니다.
        x1, y1, x2, y2 = line[0] 
        
        # cv.line()을 사용하여 검출된 직선을 결과 이미지 위에 그립니다. [cite: 46]
        # 힌트에 따라 선의 색상은 (0, 0, 255)인 빨간색으로, 두께는 2로 설정합니다. [cite: 51]
        cv.line(result_img2, (x1, y1), (x2, y2), (0, 0, 255), 2) 

# 5. 시각화
# Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화합니다. [cite: 47]
plt.figure(figsize=(10, 5)) # 전체 출력 창의 크기를 가로 10, 세로 5로 설정합니다.

# 첫 번째 영역: 원본 이미지
plt.subplot(1, 2, 1) # 1행 2열로 나눈 공간 중 첫 번째 공간을 지정합니다.
# OpenCV는 이미지를 BGR로 읽어오므로, Matplotlib에 맞게 RGB로 색상 공간을 변환하여 출력합니다.
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)) 
plt.title('Original Image') # 첫 번째 이미지의 제목을 설정합니다.
plt.axis('off') # x축, y축 눈금을 숨깁니다.

# 두 번째 영역: 허프 변환으로 검출된 직선 결과 이미지
plt.subplot(1, 2, 2) # 1행 2열로 나눈 공간 중 두 번째 공간을 지정합니다.
# 결과 이미지 또한 RGB로 색상 공간을 변환하여 화면에 출력합니다.
plt.imshow(cv.cvtColor(result_img2, cv.COLOR_BGR2RGB)) 
plt.title('Hough Lines Detected') # 두 번째 이미지의 제목을 설정합니다.
plt.axis('off') # x축, y축 눈금을 숨깁니다.

plt.tight_layout() # 시각화된 이미지들이 서로 겹치지 않도록 여백을 조절합니다.
plt.show() # 최종 결과물을 화면에 띄웁니다.
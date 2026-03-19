
# [컴퓨터비전] L03. Edge and Region 실습 과제

본 레포지토리는 OpenCV와 Matplotlib를 활용하여 이미지의 에지(Edge)를 검출하고, 허프 변환으로 직선을 찾으며, GrabCut을 이용해 객체와 배경을 분할하는 세 가지 실습 과제의 결과물을 담고 있습니다.

---

## 01. 소벨(Sobel) 에지 검출 및 결과 시각화

### 📌 과제 설명
`edgeDetectionImage` 이미지를 그레이스케일로 변환한 뒤, `cv.Sobel()` 필터를 사용하여 x축과 y축 방향의 에지를 각각 검출합니다. 이후 두 에지 결과를 바탕으로 전체 에지 강도를 계산하여 원본 이미지와 나란히 시각화합니다.

### 🖼️ 중간 및 최종 결과물

* **최종 결과 (Sobel Edge Magnitude):**
 <img width="994" height="559" alt="1번 과제" src="https://github.com/user-attachments/assets/6bfa5851-3c45-4186-add4-d7e1cc6146d7" />


### 💻 소스 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 그레이스케일 변환
# cv.imread()를 사용하여 원본 이미지를 불러옵니다.
img1 = cv.imread('edgeDetectionImage.jpg') 

# cv.cvtColor()를 사용하여 컬러 이미지를 흑백(그레이스케일)으로 변환합니다.
gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 

# 2. Sobel 에지 검출
# cv.Sobel()을 사용하여 x축 방향의 에지를 검출합니다. (타입: CV_64F, x방향: 1, y방향: 0, 커널 크기: 3)
sobel_x = cv.Sobel(gray_img1, cv.CV_64F, 1, 0, ksize=3) 

# cv.Sobel()을 사용하여 y축 방향의 에지를 검출합니다. (타입: CV_64F, x방향: 0, y방향: 1, 커널 크기: 3)
sobel_y = cv.Sobel(gray_img1, cv.CV_64F, 0, 1, ksize=3) 

# 3. 에지 강도 계산 및 형변환
# cv.magnitude()를 사용하여 x축과 y축 결과를 합친 에지 강도를 계산합니다.
magnitude = cv.magnitude(sobel_x, sobel_y) 

# cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 화면에 표시할 수 있는 uint8 형식으로 변환합니다.
magnitude_uint8 = cv.convertScaleAbs(magnitude) 

# 4. 시각화
# Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화합니다.
plt.figure(figsize=(10, 5)) 

# 첫 번째 영역: 원본 이미지 시각화
plt.subplot(1, 2, 1) 
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB)) # BGR을 RGB로 변환하여 출력
plt.title('Original Image') 
plt.axis('off') # 축 눈금 숨김

# 두 번째 영역: 검출된 에지 강도 이미지 시각화
plt.subplot(1, 2, 2) 
plt.imshow(magnitude_uint8, cmap='gray') # cmap='gray'를 사용하여 흑백으로 시각화
plt.title('Sobel Edge Magnitude') 
plt.axis('off') # 축 눈금 숨김

plt.tight_layout() # 이미지 겹침 방지
plt.show() # 결과물 창 띄우기
```

### 💡 핵심 함수 정리
* `cv.cvtColor(src, code)`: 이미지의 색상 공간을 변환합니다. (예: BGR에서 그레이스케일 또는 RGB로 변환)
* `cv.Sobel(src, ddepth, dx, dy, ksize)`: 소벨 마스크를 적용하여 이미지의 에지(윤곽선) 방향 미분값을 계산합니다.
* `cv.magnitude(x, y)`: x축과 y축 방향의 에지 기울기를 바탕으로 2D 벡터의 전체 크기(에지 강도)를 계산합니다.
* `cv.convertScaleAbs(src)`: 실수형으로 계산된 배열의 절대값을 취한 뒤, 이미지 시각화를 위해 uint8(0~255) 타입으로 변환합니다.

---

## 02. 캐니(Canny) 에지 및 허프(Hough) 변환을 이용한 직선 검출

### 📌 과제 설명
`dabo` 이미지에 캐니(Canny) 에지 검출을 적용하여 에지 맵을 생성한 후, 허프(Hough) 변환 알고리즘을 사용하여 이미지 내의 직선 성분을 추출합니다. 검출된 직선은 원본 이미지 위에 빨간색으로 표시하여 시각화합니다.

### 🖼️ 중간 및 최종 결과물

* **최종 결과 (Hough Lines Detected):**
  <img width="997" height="554" alt="2번 과제" src="https://github.com/user-attachments/assets/cfca0fb2-c65f-4e1f-8887-b7a6ab654f6e" />


### 💻 소스 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 복사본 생성
# cv.imread()를 사용하여 원본 이미지(dabo.jpg)를 불러옵니다.
img2 = cv.imread('dabo.jpg') 

# 검출된 직선을 원본 이미지에 그리기 위해 복사본을 생성합니다.
result_img2 = img2.copy() 

# 2. Canny 에지 검출
# cv.Canny()를 사용하여 에지 맵을 생성합니다. (threshold1=100, threshold2=200 설정)
edges = cv.Canny(img2, 100, 200) 

# 3. 허프 변환을 이용한 직선 검출
# cv.HoughLinesP()를 사용하여 에지 맵에서 직선을 검출합니다.
# 직선 검출 성능 개선을 위해 rho, theta, threshold, minLineLength, maxLineGap 값을 조정합니다.
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10) 

# 4. 검출된 직선 원본 이미지에 그리기
if lines is not None: # 직선이 하나라도 검출된 경우에만 실행
    for line in lines: # 검출된 모든 직선의 배열에 대해 반복
        x1, y1, x2, y2 = line[0] # 각 직선의 시작점과 끝점 좌표 추출
        
        # cv.line()을 사용하여 결과 이미지 위에 선 색상은 빨간색(0, 0, 255), 두께는 2로 직선을 그립니다.
        cv.line(result_img2, (x1, y1), (x2, y2), (0, 0, 255), 2) 

# 5. 시각화
# Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화합니다.
plt.figure(figsize=(10, 5)) 

# 첫 번째 영역: 원본 이미지
plt.subplot(1, 2, 1) 
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)) 
plt.title('Original Image') 
plt.axis('off') 

# 두 번째 영역: 허프 변환으로 검출된 직선 결과 이미지
plt.subplot(1, 2, 2) 
plt.imshow(cv.cvtColor(result_img2, cv.COLOR_BGR2RGB)) 
plt.title('Hough Lines Detected') 
plt.axis('off') 

plt.tight_layout() 
plt.show() 
```

### 💡 핵심 함수 정리
* `cv.Canny(image, threshold1, threshold2)`: 노이즈 제거와 그레디언트 계산 과정을 거쳐 이미지에서 명확한 에지(윤곽선) 맵을 추출합니다.
* `cv.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)`: 확률적 허프 변환(Probabilistic Hough Transform)을 적용하여 에지 맵에서 시작점과 끝점이 있는 선분(직선)을 검출합니다.
* `cv.line(img, pt1, pt2, color, thickness)`: 지정된 이미지(img) 위에 두 점(pt1, pt2)을 잇는 특정 색상과 두께의 직선을 그립니다.

---

## 03. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

### 📌 과제 설명
`coffee cup` 이미지에서 사용자가 마우스로 직접 지정한 사각형 영역을 바탕으로 `GrabCut` 알고리즘을 수행하여 객체(커피잔)를 추출합니다. 내부의 어두운 커피 내용물이 배경으로 인식되는 문제를 해결하기 위해, 윤곽선 검출(Contour)을 추가로 적용하여 마스크를 완벽하게 보완한 후 배경을 제거합니다.

### 🖼️ 중간 및 최종 결과물

* **최종 결과 (Background Removed):**
  <img width="1499" height="559" alt="3번 과제" src="https://github.com/user-attachments/assets/b385691c-c92e-4d07-ba85-624cdd7a6278" />


### 💻 소스 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 초기화
# 원본 이미지를 불러옵니다.
img3 = cv.imread('coffee cup.JPG') 

# 원본 이미지와 동일한 크기의 마스크를 0으로 초기화하여 생성합니다.
mask = np.zeros(img3.shape[:2], np.uint8) 

# cv.grabCut()에서 사용할 bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화합니다.
bgdModel = np.zeros((1, 65), np.float64) 
fgdModel = np.zeros((1, 65), np.float64) 

# ★ 대화식 분할: 사용자가 마우스로 직접 객체 영역(사각형)을 지정합니다. (초기 사각형 영역 설정)
rect = cv.selectROI('Select Target', img3, showCrosshair=True, fromCenter=False)
cv.destroyWindow('Select Target') # 영역 선택 후 창 닫기

# 2. GrabCut 알고리즘 수행
# 사용자가 지정한 rect 영역을 바탕으로 cv.grabCut() 알고리즘을 수행합니다.
cv.grabCut(img3, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT) 

# 3. 마스크 처리 및 보완
# 마스크 값(cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD) 중 배경(0)이나 아마도 배경(2)은 0으로, 전경은 1로 변경합니다.
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') 

# --- 문제 해결 알고리즘: 윤곽선을 찾아 뚫린 구멍(커피 내용물) 메우기 ---
# cv.findContours를 통해 가장 바깥쪽 외곽선(커피잔 테두리)만 찾습니다.
contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 찾은 외곽선 내부를 1(전경)로 꽉 채워 마스크의 결함을 보완합니다.
cv.drawContours(mask2, contours, -1, 1, thickness=cv.FILLED)
# --------------------------------------------------------

# 4. 배경 제거 적용
# 마스크 값을 원본 이미지에 곱하여 원본 이미지에서 배경을 완벽하게 제거합니다.
img3_nobg = img3 * mask2[:, :, np.newaxis] 

# 5. 시각화
# matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화합니다.
plt.figure(figsize=(15, 5)) 

# 첫 번째 영역: 원본 이미지
plt.subplot(1, 3, 1) 
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title('Original Image') 
plt.axis('off') 

# 두 번째 영역: 객체 추출 결과를 마스크 형태로 시각화
plt.subplot(1, 3, 2) 
plt.imshow(mask2, cmap='gray')
plt.title('Mask') 
plt.axis('off') 

# 세 번째 영역: 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력
plt.subplot(1, 3, 3) 
plt.imshow(cv.cvtColor(img3_nobg, cv.COLOR_BGR2RGB))
plt.title('Background Removed') 
plt.axis('off') 

plt.tight_layout() 
plt.show() 
```

### 💡 핵심 함수 정리
* `cv.selectROI(windowName, img, ...)`: 이미지를 띄워 사용자가 마우스를 이용해 관심 영역(ROI, Region of Interest)을 사각형 형태로 직접 지정할 수 있게 해줍니다.
* `cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)`: 지정된 사각형 영역 또는 마스크를 기반으로 그래프 컷 알고리즘을 반복 수행하여 전경(객체)과 배경을 분리합니다.
* `cv.findContours(image, mode, method)`: 이진화된 마스크 이미지에서 객체의 윤곽선(Contour)을 찾아 좌표 형태로 반환합니다.
* `cv.drawContours(image, contours, contourIdx, color, thickness)`: 찾은 윤곽선을 이미지 위에 그리거나, 두께를 `cv.FILLED`(-1)로 설정하여 윤곽선 내부 영역을 특정 색상(또는 값)으로 꽉 채웁니다.

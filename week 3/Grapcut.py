import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 초기화
img3 = cv.imread('coffee cup.JPG') 
mask = np.zeros(img3.shape[:2], np.uint8) 

bgdModel = np.zeros((1, 65), np.float64) 
fgdModel = np.zeros((1, 65), np.float64) 

# 영역 지정: 마우스로 넉넉하게 커피잔 전체 영역을 지정합니다.
rect = cv.selectROI('Select Target', img3, showCrosshair=True, fromCenter=False)
cv.destroyWindow('Select Target') 

# 2. GrabCut 알고리즘 수행
cv.grabCut(img3, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT) 

# 3. 마스크 이진화 (배경은 0, 전경은 1)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') 

# --- 확실한 해결책: 윤곽선(Contour)을 찾아 내부 꽉 채우기 ---
# cv.RETR_EXTERNAL을 사용하면 가장 바깥쪽 외곽선(커피잔 테두리)만 찾습니다.
contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 찾은 외곽선의 내부를 1(전경)로 꽉 채워줍니다. (커피 내용물 구멍이 완벽히 메워집니다)
cv.drawContours(mask2, contours, -1, 1, thickness=cv.FILLED)
# --------------------------------------------------------

# 완성된 마스크를 원본 이미지에 적용하여 배경 제거
img3_nobg = img3 * mask2[:, :, np.newaxis] 

# 4. 시각화
plt.figure(figsize=(15, 5)) 

plt.subplot(1, 3, 1) 
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title('Original Image') 
plt.axis('off') 

plt.subplot(1, 3, 2) 
plt.imshow(mask2, cmap='gray')
plt.title('Mask') 
plt.axis('off') 

plt.subplot(1, 3, 3) 
plt.imshow(cv.cvtColor(img3_nobg, cv.COLOR_BGR2RGB))
plt.title('Background Removed') 
plt.axis('off') 

plt.tight_layout() 
plt.show()
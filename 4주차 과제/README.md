# [컴퓨터비전] L04. Local Feature 실습 과제

본 레포지토리는 OpenCV와 Matplotlib를 활용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘으로 이미지의 특징점을 검출하고, 두 이미지 간의 특징점을 매칭하며, 호모그래피(Homography)를 이용해 파노라마 이미지로 정합하는 세 가지 실습 과제의 결과물을 담고 있습니다.

---

## 01. SIFT를 이용한 특징점 검출 및 시각화

### 📌 과제 설명

`mot_color70.jpg` 이미지를 그레이스케일로 변환한 뒤, `cv.SIFT_create()`를 사용하여 크기와 회전에 불변하는 SIFT 특징점을 검출합니다. 특징점이 너무 많아지는 것을 방지하기 위해 `nfeatures` 값을 조정하여 개수를 제한하고, `cv.drawKeypoints()`의 플래그를 활용해 특징점의 방향과 크기를 원본 이미지와 나란히 시각화합니다.

### 🖼️ 중간 및 최종 결과물

* **최종 결과 (SIFT Keypoints Detected):**
  <img width="1395" height="754" alt="1번 과제" src="https://github.com/user-attachments/assets/3d2202a3-81bb-4508-ae23-23794efc761d" />


### 💻 소스 코드

```python
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 색상 변환
# cv.imread()를 사용하여 원본 이미지를 불러옵니다.
img = cv.imread('mot_color70.jpg') 

# matplotlib 출력을 위해 BGR 색상 공간을 RGB로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

# SIFT 특징점 추출은 명암(Intensity) 정보를 사용하므로 그레이스케일로 변환합니다.
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

# 2. SIFT 객체 생성
# cv.SIFT_create()를 사용하여 SIFT 객체를 생성하되, nfeatures=300으로 특징점 개수를 제한합니다.
sift = cv.SIFT_create(nfeatures=300) 

# 3. 특징점 검출 및 디스크립터 계산
# detectAndCompute()를 사용하여 흑백 이미지에서 특징점(keypoints)을 검출합니다.
keypoints, descriptors = sift.detectAndCompute(img_gray, None) 

# 4. 특징점 시각화
# cv.drawKeypoints()를 사용하여 검출된 특징점을 이미지 위에 그립니다.
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 플래그로 특징점의 크기와 방향을 함께 표시합니다.
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
) 

# 5. 시각화
# Matplotlib를 사용하여 원본 이미지와 특징점 시각화 이미지를 나란히 출력합니다.
plt.figure(figsize=(14, 7)) 

# 첫 번째 영역: 원본 이미지 시각화
plt.subplot(1, 2, 1) 
plt.imshow(img_rgb) 
plt.title('Original Image') 
plt.axis('off') # 축 눈금 숨김

# 두 번째 영역: 특징점이 시각화된 결과 이미지 출력
plt.subplot(1, 2, 2) 
plt.imshow(img_with_keypoints) 
plt.title(f'SIFT Keypoints (nfeatures={len(keypoints)})') 
plt.axis('off') # 축 눈금 숨김

plt.tight_layout() # 이미지 겹침 방지
plt.show() # 결과물 창 띄우기
```

### 💡 핵심 함수 정리

* `cv.SIFT_create(nfeatures, ...)`: SIFT 알고리즘 객체를 생성합니다. `nfeatures` 파라미터로 유지할 최상위 특징점의 개수를 제한할 수 있습니다.
* `sift.detectAndCompute(image, mask)`: 입력 이미지에서 특징점(Keypoints)을 찾고, 각 특징점에 대한 128차원의 디스크립터(Descriptors)를 계산하여 반환합니다.
* `cv.drawKeypoints(image, keypoints, outImage, color, flags)`: 검출된 특징점을 이미지 위에 시각적으로 그려줍니다. 플래그 설정을 통해 단순한 점이나, 크기 및 방향을 포함한 원 형태로 표현할 수 있습니다.

---

## 02. SIFT를 이용한 두 영상 간 특징점 매칭

### 📌 과제 설명

`mot_color70.jpg`와 `mot_color80.jpg` 두 장의 이미지에서 각각 SIFT 특징점을 추출한 뒤, `cv.BFMatcher`의 K-NN(K-Nearest Neighbors) 알고리즘을 사용하여 특징점들을 매칭합니다. 매칭의 정확도를 높이기 위해 최근접 이웃 거리 비율(Lowe's Ratio Test)을 적용하여 오매칭(Outlier)을 걸러내고 시각화합니다.

### 🖼️ 중간 및 최종 결과물

* **최종 결과 (SIFT Feature Matching):**
<img width="1594" height="856" alt="2번 과제" src="https://github.com/user-attachments/assets/a3897998-078e-41cf-9ab9-b1d2f04875ab" />


### 💻 소스 코드

```python
import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기 및 변환
# 원본 이미지 두 장을 각각 불러옵니다.
img1 = cv.imread('mot_color70.jpg') 
img2 = cv.imread('mot_color80.jpg') 

# 시각화를 위한 RGB 변환 및 특징점 추출을 위한 그레이스케일 변환을 수행합니다.
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점, 디스크립터 추출
sift = cv.SIFT_create() 
kp1, des1 = sift.detectAndCompute(img1_gray, None) # 첫 번째 이미지의 특징점과 디스크립터 추출
kp2, des2 = sift.detectAndCompute(img2_gray, None) # 두 번째 이미지의 특징점과 디스크립터 추출

# 3. 특징점 매칭 (BFMatcher & knnMatch)
# SIFT 디스크립터 거리를 계산하기 위해 cv.NORM_L2를 사용하는 BFMatcher를 생성합니다.
bf = cv.BFMatcher(cv.NORM_L2) 

# knnMatch를 사용하여 각 특징점당 가장 가까운 이웃 2개(k=2)를 찾습니다.
matches = bf.knnMatch(des1, des2, k=2) 

# 4. 좋은 매칭점 선별 (Lowe's Ratio Test)
good_matches = [] 
for m, n in matches: # m은 1순위 매칭점, n은 2순위 매칭점
    # 1순위 매칭점의 거리가 2순위 거리의 70% 미만인 경우만 확실한 매칭으로 인정합니다.
    if m.distance < 0.7 * n.distance: 
        good_matches.append(m) 

# 거리가 짧은 순(매칭 품질이 좋은 순)으로 정렬하여 상위 매칭점만 선별하기 쉽게 합니다.
good_matches = sorted(good_matches, key=lambda x: x.distance) 

# 5. 매칭 결과 시각화
# cv.drawMatches()를 사용하여 두 이미지 간의 좋은 매칭점(상위 50개)을 선으로 이어 그립니다.
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    good_matches[:50], None, # 상위 50개만 시각화
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # 매칭되지 않은 점은 그리지 않음
) 

# 6. 결과 출력
plt.figure(figsize=(16, 8)) 
plt.imshow(img_matches) 
plt.title(f'SIFT Feature Matching (Top 50 Good Matches)') 
plt.axis('off') 
plt.tight_layout() 
plt.show() 
```

### 💡 핵심 함수 정리

* `cv.BFMatcher(normType)`: 브루트 포스(Brute-Force) 매칭 객체를 생성합니다. SIFT나 SURF 같은 실수형 디스크립터는 주로 `cv.NORM_L2`를 사용합니다.
* `bf.knnMatch(queryDescriptors, trainDescriptors, k)`: 쿼리 세트의 각 특징점에 대해 훈련 세트에서 가장 가까운 `k`개의 매칭점을 찾습니다.
* `cv.drawMatches(...)`: 두 이미지에서 검출된 특징점들을 가로로 나란히 배치하고, 서로 매칭되는 점들을 선으로 연결하여 그려줍니다.

---

## 03. 호모그래피를 이용한 이미지 정합 (Image Alignment)

### 📌 과제 설명

앞서 추출한 SIFT 매칭점들을 바탕으로 두 이미지 간의 기하학적 변환 관계인 호모그래피(Homography) 행렬을 계산합니다. 이때 `RANSAC` 알고리즘을 적용하여 이상점(Outlier)의 영향을 배제합니다. 계산된 행렬을 이용해 한 이미지를 투시 변환(`warpPerspective`)하여 다른 이미지와 하나의 넓은 파노라마로 정렬합니다.

### 🖼️ 중간 및 최종 결과물
 
* **최종 결과 (Warped Panorama Image):**
<img width="1910" height="1024" alt="3번 과제" src="https://github.com/user-attachments/assets/9b7f5c94-0101-4384-be52-01455f9bafcc" />


### 💻 소스 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 기본 처리 (이전 과정과 동일)
img1 = cv.imread('img1.jpg') 
img2 = cv.imread('img2.jpg') 
img1_rgb, img2_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB), cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray, img2_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출 및 KNN 매칭, Ratio Test를 수행하여 good_matches를 구합니다.
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# 2. 호모그래피 계산
# findHomography 연산을 위해 매칭점들의 픽셀 좌표를 float32 형태의 배열로 변환합니다.
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 

# cv.findHomography에 RANSAC 알고리즘을 적용하여 최적의 변환 행렬(H)과 정상 매칭점 마스크(mask)를 계산합니다.
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0) 

# 3. 이미지 정합 (Warping)
h1, w1 = img1_rgb.shape[:2] 
h2, w2 = img2_rgb.shape[:2] 

# 두 이미지를 이어 붙이기 위해 결과 캔버스의 너비(w1+w2)와 높이(max(h1,h2))를 설정합니다.
panorama_w = w1 + w2 
panorama_h = max(h1, h2) 

# cv.warpPerspective를 사용하여 계산된 행렬(H)을 바탕으로 img2를 투시 변환합니다.
warped_img = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h)) 

# 변환된 이미지의 좌측 빈 공간에 원본 img1을 덮어씌워 파노라마를 완성합니다.
warped_img[0:h1, 0:w1] = img1_rgb 

# 4. 시각화 (RANSAC Inliers 표시 및 파노라마 결과)
# 정상적으로 매칭된 Inlier들만 초록색 선으로 그리도록 마스크를 설정합니다.
matchesMask = mask.ravel().tolist() 
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img_matching = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, **draw_params)

# 원본 이미지, 매칭 이미지, 정합된 파노라마 이미지를 시각화합니다.
plt.figure(figsize=(20, 10)) 
plt.subplot(1, 2, 1) 
plt.imshow(img_matching) 
plt.title('Matching Result (Inliers Only)') 
plt.axis('off') 

plt.subplot(1, 2, 2) 
plt.imshow(warped_img) 
plt.title('Warped Image (Image Alignment)') 
plt.axis('off') 

plt.tight_layout() 
plt.show() 
```

### 💡 핵심 함수 정리

* `cv.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)`: 두 평면 간의 투시 변환 행렬인 3x3 호모그래피 행렬을 계산합니다. `cv.RANSAC`을 사용하여 오차(Outlier)를 무시하고 견고한 행렬을 추정할 수 있습니다.
* `cv.warpPerspective(src, M, dsize)`: 3x3 투시 변환 행렬 `M`(호모그래피 행렬)을 사용하여 이미지 `src`에 기하학적 투시 변환을 적용하고 지정된 크기(`dsize`)의 결과 이미지를 반환합니다.

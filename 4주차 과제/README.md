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
import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt # 시각화를 위해 matplotlib의 pyplot 모듈을 plt라는 이름으로 불러옵니다.

# 1. 이미지 불러오기 및 색상 변환
img = cv.imread('mot_color70.jpg') # 'mot_color70.jpg' 파일을 BGR 포맷으로 읽어옵니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # matplotlib에서 올바른 색상으로 출력하기 위해 BGR을 RGB로 변환합니다.
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # SIFT 특징점 검출은 명암(Intensity) 정보를 사용하므로 흑백(그레이스케일)으로 변환합니다.

# 2. SIFT 객체 생성
sift = cv.SIFT_create(nfeatures=300) # SIFT 알고리즘 객체를 생성하며, 화면이 너무 복잡해지지 않도록 최대 특징점 개수를 300개로 제한합니다.

# 3. 특징점 검출 및 디스크립터 계산
keypoints, descriptors = sift.detectAndCompute(img_gray, None) # 흑백 이미지에서 특징점(keypoints)의 위치와 디스크립터(특징 벡터)를 계산하여 반환합니다. 마스크는 사용하지 않습니다(None).

# 4. 특징점 시각화
img_with_keypoints = cv.drawKeypoints( # 검출된 특징점을 이미지 위에 시각적으로 그려줍니다.
    img_rgb, # 특징점을 그릴 바탕이 되는 원본 RGB 이미지입니다.
    keypoints, # detectAndCompute 함수로 찾아낸 특징점 객체들의 리스트입니다.
    None, # 결과가 출력될 이미지를 따로 지정하지 않고 함수의 반환값으로 받습니다.
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # 단순한 점이 아니라 특징점의 크기(Scale)와 방향(Orientation)까지 원과 선으로 상세히 표시하는 옵션입니다.
)

# 5. 시각화 (화면 출력)
plt.figure(figsize=(14, 7)) # 그림이 그려질 전체 도화지(Figure)의 크기를 가로 14, 세로 7 인치로 설정합니다.

plt.subplot(1, 2, 1) # 1행 2열로 나눈 그리드 공간 중 첫 번째(왼쪽) 칸을 지정합니다.
plt.imshow(img_rgb) # 원본 RGB 이미지를 해당 공간에 그립니다.
plt.title('Original Image') # 왼쪽 이미지의 상단에 'Original Image'라는 제목을 표시합니다.
plt.axis('off') # 불필요한 x축, y축 눈금과 테두리를 화면에서 숨깁니다.

plt.subplot(1, 2, 2) # 1행 2열로 나눈 그리드 공간 중 두 번째(오른쪽) 칸을 지정합니다.
plt.imshow(img_with_keypoints) # 특징점이 그려진 결과 이미지를 해당 공간에 그립니다.
plt.title(f'SIFT Keypoints (nfeatures={len(keypoints)})') # 제목에 실제로 검출된 특징점의 개수를 포매팅하여 표시합니다.
plt.axis('off') # 불필요한 x축, y축 눈금과 테두리를 화면에서 숨깁니다.

plt.tight_layout() # 두 개의 이미지가 서로 겹치거나 여백이 낭비되지 않도록 간격을 자동으로 조절합니다.
plt.show() # 설정된 모든 이미지와 제목을 실제 화면의 창으로 띄워서 보여줍니다.
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
import cv2 as cv # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt # 시각화를 위해 matplotlib의 pyplot 모듈을 plt라는 이름으로 불러옵니다.

# 1. 두 개의 이미지 불러오기 및 변환
img1 = cv.imread('mot_color70.jpg') # 매칭의 기준이 될 첫 번째 원본 이미지를 불러옵니다.
img2 = cv.imread('mot_color80.jpg') # 매칭의 대상이 될 두 번째 원본 이미지를 불러옵니다.

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB) # 첫 번째 이미지의 BGR 색상을 시각화용 RGB 색상으로 변환합니다.
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB) # 두 번째 이미지의 BGR 색상을 시각화용 RGB 색상으로 변환합니다.
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # 특징점 추출을 위해 첫 번째 이미지를 흑백으로 변환합니다.
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # 특징점 추출을 위해 두 번째 이미지를 흑백으로 변환합니다.

# 2. SIFT 객체 생성 및 특징점, 디스크립터 추출
sift = cv.SIFT_create() # 기본 파라미터를 사용하여 SIFT 객체를 생성합니다. (개수 제한 없음)
kp1, des1 = sift.detectAndCompute(img1_gray, None) # 첫 번째 이미지에서 특징점(kp1)과 특징 벡터인 디스크립터(des1)를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2_gray, None) # 두 번째 이미지에서 특징점(kp2)과 특징 벡터인 디스크립터(des2)를 추출합니다.

# 3. 특징점 매칭 (BFMatcher & knnMatch)
bf = cv.BFMatcher(cv.NORM_L2) # SIFT 디스크립터는 실수형 값이므로, 거리 계산 방식으로 유클리디안 거리(L2 Norm)를 사용하는 매처(Matcher)를 생성합니다.

matches = bf.knnMatch(des1, des2, k=2) # 첫 번째 이미지의 각 특징점에 대해, 두 번째 이미지에서 가장 가까운 특징점 2개(k=2)를 찾습니다.

# 4. 좋은 매칭점 선별 (Lowe's Ratio Test 적용)
good_matches = [] # 오매칭을 걸러내고 진짜(좋은) 매칭점만 담아둘 빈 리스트를 만듭니다.
for m, n in matches: # 검출된 매칭점 쌍(m: 1순위로 가까운 점, n: 2순위로 가까운 점)에 대해 반복문을 실행합니다.
    if m.distance < 0.7 * n.distance: # 1순위 거리가 2순위 거리의 70%보다 짧은 경우에만 (즉, 1순위가 압도적으로 유사한 경우에만)
        good_matches.append(m) # 해당 매칭점(m)을 신뢰할 수 있다고 판단하여 good_matches 리스트에 추가합니다.

# 매칭 결과가 너무 많아 선이 얽히는 것을 방지하기 위해, 매칭 품질이 좋은 순서(거리가 짧은 순서)로 오름차순 정렬합니다.
good_matches = sorted(good_matches, key=lambda x: x.distance) 

# 5. 매칭 결과 시각화
img_matches = cv.drawMatches( # 두 이미지의 특징점을 선으로 연결하여 그려주는 함수를 호출합니다.
    img1_rgb, kp1, # 첫 번째 바탕 이미지와 그 이미지의 특징점들을 지정합니다.
    img2_rgb, kp2, # 두 번째 바탕 이미지와 그 이미지의 특징점들을 지정합니다.
    good_matches[:50], None, # 정렬된 우수 매칭점 중 상위 50개만 선별하여 그립니다. 출력 이미지는 따로 지정하지 않습니다.
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # 짝을 찾지 못해 매칭되지 않은 나머지 특징점(단일 점)들은 화면에 그리지 않는 옵션입니다.
)

# 6. 결과 출력
plt.figure(figsize=(16, 8)) # 가로 16, 세로 8 인치 크기의 도화지를 생성합니다.
plt.imshow(img_matches) # 선으로 연결된 두 이미지의 매칭 결과를 도화지에 그립니다.
plt.title(f'SIFT Feature Matching (Top 50 Good Matches)') # 이미지 상단에 제목을 출력합니다.
plt.axis('off') # 지저분해 보이지 않도록 x, y 축 눈금을 숨깁니다.
plt.tight_layout() # 여백을 최적화합니다.
plt.show() # 최종 매칭 결과를 화면에 띄웁니다.
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
import cv2 as cv # OpenCV 라이브러리를 cv로 불러옵니다.
import numpy as np # 수학적 배열 연산을 처리하기 위해 numpy 라이브러리를 np로 불러옵니다.
import matplotlib.pyplot as plt # 결과 이미지 시각화를 위해 matplotlib을 불러옵니다.

# 1. 이미지 불러오기 및 기본 처리 (이전 과정과 동일)
img1 = cv.imread('img1.jpg') # 파노라마의 좌측 기준이 될 첫 번째 이미지를 불러옵니다.
img2 = cv.imread('img2.jpg') # 파노라마의 우측에 이어 붙일 두 번째 이미지를 불러옵니다.

# Matplotlib 출력을 위한 RGB 변환을 각각 수행합니다.
img1_rgb, img2_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB), cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# SIFT 특징점 연산을 위해 그레이스케일(흑백) 변환을 각각 수행합니다.
img1_gray, img2_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출 및 추출을 진행합니다.
sift = cv.SIFT_create() # SIFT 객체 생성
kp1, des1 = sift.detectAndCompute(img1_gray, None) # img1의 특징점과 디스크립터 계산
kp2, des2 = sift.detectAndCompute(img2_gray, None) # img2의 특징점과 디스크립터 계산

# BFMatcher 객체를 생성하고 KNN(k=2) 매칭을 수행합니다.
bf = cv.BFMatcher(cv.NORM_L2) # L2 거리 방식을 사용하는 매처 생성
matches = bf.knnMatch(des1, des2, k=2) # 1순위, 2순위 매칭점 검색

# Lowe's Ratio Test (비율 0.7)를 적용하여 신뢰도 높은 매칭점만 good_matches 리스트에 필터링합니다.
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# 2. 호모그래피 계산
# findHomography 함수는 점들의 픽셀 좌표가 필요하므로, 매칭된 특징점들의 (x, y) 좌표만 추출하여 Numpy 배열로 만듭니다.
# queryIdx는 img1의 특징점 인덱스, trainIdx는 img2의 특징점 인덱스입니다. 차원 맞춤을 위해 reshape(-1, 1, 2)를 적용합니다.
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 변환의 기준이 되는 목적지 좌표(img1)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 변환을 수행할 출발지 좌표(img2)

# RANSAC 알고리즘을 사용하여 오차(Outlier)를 제외하고 최적의 호모그래피 변환 행렬(H)을 구합니다. (허용 오차 거리: 5.0 픽셀)
# mask는 계산에 실제 사용된 진짜 매칭점(Inlier, 1)과 버려진 오매칭점(Outlier, 0)의 상태를 담고 있습니다.
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0) 

# 3. 이미지 정합 (Warping)
h1, w1 = img1_rgb.shape[:2] # 첫 번째 이미지의 높이(h1)와 너비(w1)를 구합니다.
h2, w2 = img2_rgb.shape[:2] # 두 번째 이미지의 높이(h2)와 너비(w2)를 구합니다.

panorama_w = w1 + w2 # 두 이미지가 가로로 합쳐질 것이므로, 새 캔버스의 전체 너비를 두 너비의 합으로 넉넉하게 잡습니다.
panorama_h = max(h1, h2) # 높이는 두 이미지 중 더 큰 쪽에 맞춰 캔버스가 잘리지 않게 설정합니다.

# cv.warpPerspective를 호출하여 구한 호모그래피 행렬(H)을 바탕으로 img2를 투시 변환(왜곡)시켜 새로운 좌표계로 이동시킵니다.
warped_img = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h)) 

# 이미 변환된 img2가 그려진 넓은 캔버스 좌측 원점 영역에, 기준이 되는 img1을 있는 그대로 덮어씌워 파노라마를 합성합니다.
warped_img[0:h1, 0:w1] = img1_rgb 

# 4. 시각화 (RANSAC Inliers 표시 및 파노라마 결과)
matchesMask = mask.ravel().tolist() # 마스크 배열을 1차원 리스트로 쭉 펴서(ravel) drawMatches 함수의 인자 형태로 변환합니다.

# 시각화 함수에 전달할 옵션들을 딕셔너리 형태로 묶어 정리합니다.
draw_params = dict(matchColor=(0, 255, 0), # RANSAC을 통과한 진짜 매칭점 선의 색상을 초록색(0, 255, 0)으로 지정합니다.
                   singlePointColor=None, # 매칭되지 않은 점들은 굳이 그리지 않도록 설정합니다.
                   matchesMask=matchesMask, # RANSAC이 Outlier로 판정한 선은 그리지 않도록 마스크를 적용합니다.
                   flags=2) # DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 와 같은 역할을 하는 정수 2를 입력합니다.

# 설정한 파라미터(**draw_params)를 적용하여 진짜 매칭점(Inliers)만 초록색 선으로 연결된 결과 이미지를 생성합니다.
img_matching = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, **draw_params)

# 최종 결과를 나란히 출력하기 위해 화면을 구성합니다.
plt.figure(figsize=(20, 10)) # 전체 가로 20, 세로 10 인치로 창 크기를 크게 설정합니다.

plt.subplot(1, 2, 1) # 좌측 칸 지정
plt.imshow(img_matching) # RANSAC 인라이어 매칭선이 그려진 이미지를 띄웁니다.
plt.title('Matching Result (Inliers Only)') # 좌측 그림의 제목입니다.
plt.axis('off') # 축 눈금 제거

plt.subplot(1, 2, 2) # 우측 칸 지정
plt.imshow(warped_img) # 두 이미지가 성공적으로 정합된 최종 파노라마 이미지를 띄웁니다.
plt.title('Warped Image (Image Alignment)') # 우측 그림의 제목입니다.
plt.axis('off') # 축 눈금 제거

plt.tight_layout() # 이미지 겹침 방지
plt.show() # 결과 확인
```

### 💡 핵심 함수 정리

* `cv.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold)`: 두 평면 간의 투시 변환 행렬인 3x3 호모그래피 행렬을 계산합니다. `cv.RANSAC`을 사용하여 오차(Outlier)를 무시하고 견고한 행렬을 추정할 수 있습니다.
* `cv.warpPerspective(src, M, dsize)`: 3x3 투시 변환 행렬 `M`(호모그래피 행렬)을 사용하여 이미지 `src`에 기하학적 투시 변환을 적용하고 지정된 크기(`dsize`)의 결과 이미지를 반환합니다.

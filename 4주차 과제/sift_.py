import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기 (샘플 파일 img1.jpg, img2.jpg 사용)
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 특징점 검출 및 매칭 (2번 과제와 동일)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# 힌트 반영: 거리 비율 임계값(0.7)을 적용하여 좋은 매칭점 선별
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 3. 호모그래피 계산
# findHomography 함수에 넣기 위해 매칭된 특징점들의 좌표만 추출 (float32 형태로 변환)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 힌트 반영: cv.RANSAC을 사용하여 이상점(Outlier)의 영향을 배제하고 호모그래피 행렬(H) 계산
# img2를 img1의 시점으로 변환하기 위해 목적지(dst_pts)에서 출발지(src_pts)로의 변환 행렬을 구함
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

# 4. 이미지 정합 (Warping)
h1, w1 = img1_rgb.shape[:2]
h2, w2 = img2_rgb.shape[:2]

# 힌트 반영: 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
panorama_w = w1 + w2
panorama_h = max(h1, h2)

# cv.warpPerspective를 사용하여 img2를 변환
warped_img = cv.warpPerspective(img2_rgb, H, (panorama_w, panorama_h))

# 변환된 넓은 캔버스에 원본 img1을 제자리에 덮어씌움 (두 이미지가 합쳐짐)
warped_img[0:h1, 0:w1] = img1_rgb

# 5. 매칭 결과 이미지 생성 (시각화용)
# RANSAC을 통해 걸러진 진짜 매칭점(Inlier)만 초록색 선으로 표시되도록 마스크 설정
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0), 
                   singlePointColor=None,
                   matchesMask=matchesMask, 
                   flags=2)

img_matching = cv.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, **draw_params)

# 6. matplotlib을 이용한 결과 나란히 출력
plt.figure(figsize=(20, 10))

# 매칭 결과 (Matching Result)
plt.subplot(1, 2, 1)
plt.imshow(img_matching)
plt.title('Matching Result (Inliers Only)')
plt.axis('off')

# 변환된 이미지 (Warped Image)
plt.subplot(1, 2, 2)
plt.imshow(warped_img)
plt.title('Warped Image (Image Alignment)')
plt.axis('off')

plt.tight_layout()
plt.show()
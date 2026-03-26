import cv2 as cv
import matplotlib.pyplot as plt

# 1. 두 개의 이미지 불러오기 및 RGB/Gray 변환
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

# matplotlib 출력을 위한 RGB 변환
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# SIFT 특징점 추출을 위한 그레이스케일 변환
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점, 디스크립터 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 3. 특징점 매칭 (힌트 반영: BFMatcher와 knnMatch 사용)
# SIFT 디스크립터는 실수형이므로 NORM_L2를 사용합니다.
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2) # 가장 가까운 2개의 이웃을 찾음

# 4. 좋은 매칭점 선별 (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 💡 추가된 부분: 매칭점들을 거리(distance) 기준으로 오름차순 정렬 (거리가 짧을수록 좋은 매칭)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# 시각화할 매칭점 개수 제한 (예: 상위 50개)
limit = 50

# 5. 매칭 결과 시각화 (상위 50개만 잘라서 그리기)
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    good_matches[:limit], # 👈 good_matches 리스트를 limit 개수만큼만 슬라이싱
    None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 6. matplotlib을 이용한 결과 출력 (기존과 동일)
plt.figure(figsize=(16, 8))
plt.imshow(img_matches)
plt.title(f'SIFT Feature Matching (Top {limit} of {len(good_matches)} Good Matches)')
plt.axis('off')

plt.tight_layout()
plt.show()
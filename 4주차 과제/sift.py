import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 색상 변환
img = cv.imread('mot_color70.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 (힌트 반영: nfeatures로 특징점 개수 제한)
sift = cv.SIFT_create(nfeatures=500)

# 3. 특징점 검출
keypoints, _ = sift.detectAndCompute(img_gray, None)

# 4. 특징점 시각화 (힌트 반영: 방향과 크기 표시)
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. matplotlib을 이용한 2개 나란히 출력
plt.figure(figsize=(14, 7))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 특징점 시각화 이미지
plt.subplot(1, 2, 2)
plt.imshow(img_with_keypoints)
plt.title(f'SIFT Keypoints (nfeatures={len(keypoints)})')
plt.axis('off')

plt.tight_layout()
plt.show()
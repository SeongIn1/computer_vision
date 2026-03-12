import cv2  # OpenCV 영상 처리 라이브러리를 불러옵니다.
import numpy as np  # 배열 및 수치 연산을 위해 NumPy 라이브러리를 불러옵니다.
from pathlib import Path  # 폴더 생성을 위해 Path 클래스를 불러옵니다.

# 출력 폴더 생성 (과제 제출용 결과물을 모아두기 위함)
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. 이미지 로드 및 정보 추출
# -----------------------------
# 변환을 적용할 원본 장미 이미지를 불러옵니다. (경로가 다를 경우 수정 필요)
img = cv2.imread('rose.png')

if img is None:  # 이미지를 정상적으로 불러오지 못했는지 확인합니다.
    raise FileNotFoundError("이미지를 찾지 못했습니다. 파일 경로를 확인해 주세요.")

# 읽어온 이미지의 형태(shape) 정보에서 높이(h)와 너비(w) 값을 추출합니다.
h, w = img.shape[:2]

# -----------------------------
# 2. 변환 행렬 생성 (회전 및 크기 조절)
# -----------------------------
# 이미지의 정중앙을 회전 기준으로 삼기 위해 너비와 높이의 절반 값을 계산합니다.
center = (w / 2.0, h / 2.0)

# cv2.getRotationMatrix2D()를 사용하여 회전 및 스케일 변환 행렬을 만듭니다[cite: 89].
# 중심점(center)을 기준으로 시계 반대 방향으로 30도 회전하고, 크기를 0.8배로 축소합니다[cite: 86, 87].
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# -----------------------------
# 3. 변환 행렬 수정 (평행이동 반영)
# -----------------------------
# 2x3 회전 행렬(M)의 마지막 열 값이 평행이동(Translation)을 담당하므로 이 값을 직접 조정합니다[cite: 91].
# 1행 3열(M[0, 2]) 값에 80을 더해 x축 방향으로 +80px(오른쪽) 이동시킵니다.
M[0, 2] += 80
# 2행 3열(M[1, 2]) 값에서 40을 빼서 y축 방향으로 -40px(위쪽) 이동시킵니다.
M[1, 2] -= 40

# -----------------------------
# 4. 최종 이미지 변환 적용
# -----------------------------
# cv2.warpAffine() 함수를 사용하여 원본 이미지(img)에 완성된 변환 행렬(M)을 적용합니다[cite: 90].
# 출력되는 결과 이미지의 캔버스 크기는 원본과 동일하게 (w, h)로 유지합니다.
result_img = cv2.warpAffine(img, M, (w, h))

# -----------------------------
# 5. 저장 및 결과 출력
# -----------------------------
# 변환이 완료된 이미지를 outputs 폴더에 'transformed_rose.png'라는 이름으로 저장합니다.
cv2.imwrite(str(output_dir / "transformed_rose.png"), result_img)

# 원본 이미지와 변환된 이미지를 각각 화면에 띄워 비교할 수 있도록 합니다.
cv2.imshow('Original', img)
cv2.imshow('Rotated + Scaled + Translated', result_img)

# 사용자가 키보드의 아무 키나 누를 때까지 창을 닫지 않고 무한히 대기합니다.
cv2.waitKey(0)
# 대기가 끝나면 열려있는 모든 OpenCV 결과 창을 안전하게 닫아줍니다.
cv2.destroyAllWindows()
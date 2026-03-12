# 💻 [L02] Image Formation 실습 과제

본 리포지토리는 카메라 캘리브레이션, 기하학적 이미지 변환, 스테레오 비전을 통한 깊이 추정 등 3가지 핵심 이미지 형성(Image Formation) 원리를 직접 구현한 코드와 결과물을 포함하고 있습니다. 제출된 모든 파이썬 스크립트에는 실행 흐름과 수학적 원리를 설명하는 상세한 주석이 작성되어 있습니다.

---

## 📌 1. 체크보드 기반 카메라 캘리브레이션 (Camera Calibration)

### 📝 과제 설명
[cite_start]다양한 각도에서 촬영된 체크보드 패턴 이미지를 활용하여 카메라의 내부 행렬(Camera Matrix, $K$)과 왜곡 계수(Distortion Coefficients)를 추정하고, 이를 바탕으로 렌즈의 방사 왜곡 및 접선 왜곡을 보정합니다[cite: 16].

### 📸 결과물
* **중간 결과물 (코너 검출):**
  *(설명: cv2.findChessboardCorners()를 통해 이미지 내 2D 격자 코너를 서브픽셀 단위로 정밀하게 검출한 화면입니다.)*
  <img width="640" height="480" alt="corner_detection" src="https://github.com/user-attachments/assets/908635b5-f91c-4876-9373-c2112a3a4e68" />


* **최종 결과물 (왜곡 보정):**
  *(설명: 좌측은 렌즈 왜곡이 포함된 원본 이미지, 우측은 도출된 카메라 파라미터를 적용해 직선이 올바르게 펴진 최종 보정 이미지입니다.)*
  <img width="1292" height="524" alt="1번과제 최종" src="https://github.com/user-attachments/assets/5266440a-4c8b-408d-8623-06772e6e36fb" />


### 🔑 핵심 함수 정리
* [cite_start]`cv2.findChessboardCorners(image, patternSize)`: 흑백 이미지에서 체크보드의 내부 코너점(2D 이미지 좌표)을 검출합니다[cite: 40, 42, 43].
* [cite_start]`cv2.calibrateCamera(objPoints, imgPoints, imageSize)`: 실제 3D 좌표와 이미지에서 찾은 2D 코너 좌표의 대응 관계를 이용해 카메라 내부 파라미터와 왜곡 계수를 계산합니다[cite: 21, 44, 46].
* [cite_start]`cv2.undistort(src, cameraMatrix, distCoeffs)`: 구해진 카메라 파라미터를 원본 이미지에 적용하여 렌즈 왜곡을 평탄하게 보정합니다[cite: 22].

---

## 📌 2. 이미지 Rotation & Transformation

### 📝 과제 설명
[cite_start]한 장의 단일 이미지(장미)에 대해 아핀 변환(Affine Transformation) 행렬을 구성하여 회전(중심 기준 +30도), 크기 조절(0.8배 스케일링), 평행이동(x축 +80px, y축 -40px)을 한 번의 연산으로 동시에 적용합니다[cite: 83, 86, 87, 88].

### 📸 결과물
* **최종 결과물:**
  *(설명: 좌측 원본 이미지에 3가지 기하학적 변환을 모두 적용한 결과 이미지입니다.)*
  <img width="1188" height="792" alt="rose" src="https://github.com/user-attachments/assets/ec2a329b-b819-48d2-a71e-76bb6b05ece0" />
  <img width="1188" height="792" alt="transformed_rose" src="https://github.com/user-attachments/assets/ee59d7af-6fa8-418b-bcb8-30260c86e8f0" />



### 🔑 핵심 함수 정리
* [cite_start]`cv2.getRotationMatrix2D(center, angle, scale)`: 이미지의 중심 좌표, 회전 각도, 확대/축소 비율을 입력받아 2x3 크기의 2D 회전 및 스케일 변환 행렬을 반환합니다[cite: 89].
* [cite_start]`cv2.warpAffine(src, M, dsize)`: 입력 이미지에 2x3 아핀 변환 행렬(M)을 적용하여 최종적으로 형태가 변환된 이미지를 생성합니다[cite: 90].

---

## 📌 3. Stereo Disparity 기반 Depth 추정

### 📝 과제 설명
[cite_start]스테레오 카메라(Left/Right)에서 촬영된 동일한 장면의 두 이미지를 분석하여 픽셀 위치 차이(Disparity)를 구하고, 물리적 거리($Z = \frac{fB}{d}$)인 Depth Map을 추정합니다[cite: 108, 115, 125]. [cite_start]추가로 지정된 3개의 관심 영역(Painting, Frog, Teddy)에 대해 평균 거리를 계산합니다[cite: 111].

### 📸 결과물
중간 및 최종 결과물 종합:
(설명: 좌측 상단부터 시계방향으로 Disparity Map, Left ROI, Right ROI, Depth Map 결과입니다.)
<img width="916" height="832" alt="3번과제 최종" src="https://github.com/user-attachments/assets/a76aa68b-c46c-4e9b-b25f-e7aee2e0852b" />

수치 분석 결과 (Terminal):
(설명: 각 ROI 영역의 평균 Disparity와 계산된 평균 Depth 값입니다.)
<img width="389" height="74" alt="3번과제 결과" src="https://github.com/user-attachments/assets/e37ea07b-a9a8-49b3-8cfd-dc71dfd237f3" />

[영역별 평균 거리 분석 결과]

가장 가까운 객체: Frog (개구리 인형) (평균 Depth: 2.51)

가장 먼 객체: Painting (액자 그림) (평균 Depth: 4.44)
(해석: 분석 결과 Frog의 평균 시차(Disparity)가 33.68로 가장 크며, 이에 따라 계산된 실제 거리(Depth)가 가장 짧게 측정되었습니다.)

### 🔑 핵심 함수 정리
* [cite_start]`cv2.StereoBM_create(numDisparities, blockSize)`: Block Matching 알고리즘을 사용하여 좌우 이미지로부터 Disparity Map을 계산하는 객체를 생성합니다[cite: 110, 132, 133].
* [cite_start]`stereo.compute(left_img, right_img)`: 그레이스케일로 변환된 좌우 이미지를 비교하여 정수형 disparity 값을 16배 스케일 해서 반환하므로, Depth 계산 전 실수형 변환 및 16으로 나누는 과정이 필수적입니다[cite: 134, 135].

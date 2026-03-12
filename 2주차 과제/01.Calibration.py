import cv2  # OpenCV 영상 처리 라이브러리를 불러옵니다.
import numpy as np  # 배열 및 수치 연산을 위해 NumPy 라이브러리를 불러옵니다.
import glob  # 지정된 패턴과 일치하는 여러 파일 경로를 한 번에 찾기 위해 glob 모듈을 불러옵니다.

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)  # 가로 9개, 세로 6개의 내부 코너를 가진 체크보드 패턴 크기를 설정합니다.

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0  # 체크보드 격자 한 칸의 실제 한 변의 크기를 25mm로 설정합니다[cite: 24].

# 코너 정밀화 조건
# 최대 30번 반복하거나 정확도가 0.001에 도달하면 코너 위치 탐색(서브픽셀 정밀화)을 종료하는 기준을 설정합니다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)  # (54, 3) 크기의 0으로 채워진 3차원 공간 좌표 배열을 만듭니다.
# mgrid를 사용해 2D 격자 좌표를 생성하고 형태를 바꾼 뒤 objp의 x, y 좌표 부분에 대입합니다.
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # 생성된 기본 격자 좌표에 실제 한 칸의 크기(25.0)를 곱해 실제 물리적 3D 좌표로 완성합니다.

# 저장할 좌표
objpoints = []  # 캘리브레이션에 사용할 실제 세계의 3D 점들을 차곡차곡 저장할 빈 리스트입니다[cite: 43].
imgpoints = []  # 캘리브레이션에 사용할 이미지 평면의 2D 코너점들을 저장할 빈 리스트입니다[cite: 43].

images = glob.glob("calibration_images/left*.jpg")  # 'calibration_images' 폴더 안에서 이름이 'left'로 시작하는 모든 jpg 파일 경로를 가져옵니다.

img_size = None  # 카메라 캘리브레이션 함수에 전달할 이미지의 해상도(가로, 세로 크기)를 저장할 변수를 비워둡니다.

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:  # glob으로 찾은 이미지 파일 경로들을 하나씩 순회합니다.
    img = cv2.imread(fname)  # 현재 순서의 이미지 파일을 읽어와 img 변수에 저장합니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 코너 검출 알고리즘 적용을 위해 컬러 이미지를 흑백(그레이스케일)으로 변환합니다.
    
    if img_size is None:  # 이미지 해상도 정보가 아직 없다면 (첫 번째 이미지를 처리할 때)
        img_size = gray.shape[::-1]  # 흑백 이미지의 형태(높이, 너비)를 역순(너비, 높이)으로 만들어 img_size에 저장합니다.

    # cv2.findChessboardCorners 함수를 사용해 이미지 내에서 2D 체크보드 코너 위치를 찾습니다[cite: 42].
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:  # 코너점들을 모두 성공적으로 검출했다면
        objpoints.append(objp)  # 미리 만들어둔 실제 3D 격자 좌표(objp)를 objpoints 리스트에 추가합니다.
        
        # 찾아낸 코너점들의 위치를 픽셀 단위보다 더 정밀한 서브픽셀(sub-pixel) 수준으로 조정합니다.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # 정밀해진 2D 코너 좌표를 imgpoints 리스트에 추가합니다.

        # 검출된 코너점들을 원본 이미지 위에 무지개색 선과 점으로 시각화하여 그립니다.
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corner Detection', img)  # 코너가 잘 찾아졌는지 확인하기 위해 창을 띄워 이미지를 보여줍니다.
        cv2.waitKey(100)  # 이미지가 넘어가는 것을 눈으로 확인할 수 있도록 100밀리초(0.1초) 동안 대기합니다.

cv2.destroyAllWindows()  # 모든 이미지의 코너 검출 작업과 확인이 끝났으므로 띄워둔 창을 닫습니다.

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 구축된 3D 실제 좌표(objpoints)와 2D 이미지 좌표(imgpoints)를 바탕으로 카메라의 내부 행렬(K)과 왜곡 계수(dist)를 계산합니다[cite: 46, 47].
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")  # 초점 거리(f)와 주점(p)이 포함된 3x3 카메라 내부 파라미터 행렬 출력을 알립니다[cite: 66, 67, 71].
print(K)  # 계산된 카메라 행렬 K를 콘솔창에 출력합니다.

print("\nDistortion Coefficients:")  # 방사 왜곡 및 접선 왜곡 계수 출력을 알립니다[cite: 72, 73].
print(dist)  # 계산된 왜곡 계수 배열(k1, k2, p1, p2, k3 등)을 콘솔창에 출력합니다.

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:  # 캘리브레이션에 사용된 이미지가 최소 한 장 이상 존재한다면
    test_img = cv2.imread(images[0])  # 왜곡 보정 결과를 테스트해 볼 첫 번째 원본 이미지를 다시 읽어옵니다.
    
    # 계산해 낸 카메라 행렬(K)과 왜곡 계수(dist)를 cv2.undistort 함수에 넣어 렌즈 왜곡이 펴진 깔끔한 이미지를 생성합니다[cite: 22].
    undistorted_img = cv2.undistort(test_img, K, dist, None, K)
    
    cv2.imshow('Original Distorted Image', test_img)  # 렌즈 왜곡이 남아있는 원래의 테스트 이미지를 화면에 띄웁니다.
    cv2.imshow('Undistorted Image', undistorted_img)  # 캘리브레이션 파라미터를 통해 왜곡이 보정된 결과 이미지를 다른 창에 띄웁니다.
    
    cv2.waitKey(0)  # 보정 전후 이미지를 충분히 비교할 수 있도록 사용자가 키보드를 누를 때까지 프로그램 실행을 멈추고 대기합니다.
    cv2.destroyAllWindows()  # 사용자가 키를 누르면 열려있던 결과 비교 창들을 모두 닫고 스크립트를 완전히 종료합니다.
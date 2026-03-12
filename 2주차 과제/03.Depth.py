import cv2  # 영상 처리를 위한 OpenCV 라이브러리를 불러옵니다.
import numpy as np  # 행렬 연산을 위한 NumPy 라이브러리를 불러옵니다.
from pathlib import Path  # 파일 및 디렉토리 경로 작업을 위해 Path 클래스를 불러옵니다.

# 출력 폴더 생성
output_dir = Path("./outputs")  # 결과물을 저장할 폴더 경로를 설정합니다.
output_dir.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성하고, 이미 있어도 에러를 발생시키지 않습니다.

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")  # 왼쪽 카메라로 찍은 컬러 이미지를 불러옵니다.
right_color = cv2.imread("right.png")  # 오른쪽 카메라로 찍은 컬러 이미지를 불러옵니다.

if left_color is None or right_color is None:  # 이미지를 정상적으로 불러오지 못했는지 확인합니다.
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 이미지가 없으면 에러를 발생시킵니다.

# 카메라 파라미터
f = 700.0  # 카메라의 초점 거리(focal length)를 설정합니다[cite: 126].
B = 0.12  # 두 카메라 사이의 거리인 베이스라인(baseline)을 설정합니다[cite: 127].

# ROI 설정
rois = {  # 측정할 세 가지 관심 영역(ROI)의 (x, y, w, h) 좌표를 딕셔너리로 설정합니다.
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)  # 왼쪽 컬러 이미지를 흑백(그레이스케일)으로 변환합니다[cite: 110].
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 컬러 이미지를 흑백으로 변환합니다[cite: 110].

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)  # StereoBM 알고리즘 객체를 생성합니다[cite: 132, 133].
disparity_16S = stereo.compute(left_gray, right_gray)  # 좌우 흑백 이미지를 비교하여 16배 스케일된 정수형 disparity 맵을 계산합니다[cite: 134].
disparity = disparity_16S.astype(np.float32) / 16.0  # 실수형 연산을 위해 float32로 변환 후 16으로 나누어 실제 disparity 값을 구합니다[cite: 135].

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
valid_mask = disparity > 0  # disparity 값이 0보다 큰 유효한 픽셀만 필터링하기 위한 마스크를 생성합니다[cite: 110].
depth_map = np.zeros_like(disparity, dtype=np.float32)  # disparity 맵과 동일한 크기의 빈 0 배열을 생성합니다.
# 유효한 픽셀 위치에 대해서만 깊이 계산 공식 Z = fB / d 를 적용하여 거리를 구합니다[cite: 115, 125].
depth_map[valid_mask] = (f * B) / disparity[valid_mask] 

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}  # 결과를 저장할 빈 딕셔너리를 만듭니다.

for name, (x, y, w, h) in rois.items():  # 정의해둔 ROI들을 하나씩 순회합니다.
    roi_disp = disparity[y:y+h, x:x+w]  # 전체 disparity 맵에서 해당 ROI 영역만 잘라냅니다.
    roi_depth = depth_map[y:y+h, x:x+w]  # 전체 depth 맵에서 해당 ROI 영역만 잘라냅니다.
    roi_valid = valid_mask[y:y+h, x:x+w]  # 해당 영역 중 유효한 픽셀 위치만 가져옵니다.
    
    if np.any(roi_valid):  # 유효한 픽셀이 하나라도 존재한다면
        mean_disp = np.mean(roi_disp[roi_valid])  # 유효한 픽셀들의 평균 disparity를 계산합니다[cite: 111].
        mean_depth = np.mean(roi_depth[roi_valid])  # 유효한 픽셀들의 평균 depth를 계산합니다[cite: 111].
        results[name] = {"disp": mean_disp, "depth": mean_depth}  # 계산된 값을 딕셔너리에 저장합니다.

# -----------------------------
# 4. 결과 출력
# -----------------------------
# disparity 값이 가장 큰(가장 가까운) ROI를 찾습니다[cite: 114, 129].
closest_roi = max(results, key=lambda k: results[k]["disp"]) 
# disparity 값이 가장 작은(가장 먼) ROI를 찾습니다[cite: 114, 129].
farthest_roi = min(results, key=lambda k: results[k]["disp"]) 

for name, data in results.items():  # 각 ROI의 계산 결과를 화면에 출력합니다.
    print(f"[{name}] 평균 Disparity: {data['disp']:.2f}, 평균 Depth: {data['depth']:.2f}")

print(f"\n해석 결과: 가장 가까운 영역은 '{closest_roi}', 가장 먼 영역은 '{farthest_roi}' 입니다. [cite: 112, 141]")

# -----------------------------
# 5. disparity 시각화 (기존 작성 코드 유지)
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)
if d_max <= d_min:
    d_max = d_min + 1e-6
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화 (기존 작성 코드 유지)
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)
    if z_max <= z_min:
        z_max = z_min + 1e-6
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시 (기존 작성 코드 유지)
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()
for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
# 생성된 컬러맵 이미지와 ROI가 표시된 이미지를 outputs 폴더에 저장합니다.
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color) 
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)

# -----------------------------
# 9. 출력
# -----------------------------
# 처리된 이미지들을 화면의 새로운 창에 띄워서 보여줍니다.
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)
cv2.imshow("Left ROI", left_vis)

cv2.waitKey(0)  # 사용자가 키보드의 아무 키나 누를 때까지 창을 닫지 않고 대기합니다.
cv2.destroyAllWindows()  # 대기가 끝나면 열려있는 모든 OpenCV 창을 닫아 메모리를 정리합니다.
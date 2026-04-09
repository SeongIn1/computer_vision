# [컴퓨터비전] L07. YOLOv3와 SORT를 이용한 다중 객체 추적(MOT)

본 레포지토리는 OpenCV의 DNN 모듈을 이용한 **YOLOv3** 객체 검출기와 **SORT(Simple Online and Realtime Tracking)** 알고리즘을 결합하여, 영상 내 여러 객체를 실시간으로 탐지하고 고유 ID를 부여하여 지속적으로 추적하는 실습 과제를 다룹니다.

---

## 01. YOLOv3 & SORT 기반 다중 객체 추적 실습

### 📌 과제 설명

본 과제는 YOLOv3 검출기가 제공하는 바운딩 박스 정보를 바탕으로 **칼만 필터(Kalman Filter)**를 사용하여 객체의 다음 위치를 예측하고, **헝가리안 알고리즘(Hungarian Algorithm)**을 통해 이전 프레임의 객체와 현재 검출된 객체를 최적으로 매칭합니다. 이를 통해 객체가 일시적으로 가려지거나 겹치더라도 안정적으로 추적을 유지하는 다중 객체 추적(Multi-Object Tracking) 시스템을 구현합니다.

### 🖼️ 최종 결과물
<img width="638" height="387" alt="1번과제 결과" src="https://github.com/user-attachments/assets/bfa9431b-6762-46cf-a837-11830ce85fce" />

### 💻 소스 코드 (`mot_sort.py`)

```python
import cv2 # 영상 처리를 위한 OpenCV 라이브러리를 불러옵니다.
import numpy as np # 수치 계산 및 배열 조작을 위해 numpy를 불러옵니다.
from scipy.optimize import linear_sum_assignment # 데이터 연관을 위한 헝가리안 알고리즘 함수를 불러옵니다.
from filterpy.kalman import KalmanFilter # 객체의 상태 예측을 위한 칼만 필터 라이브러리를 불러옵니다.

# =====================================================================
# 1. SORT 알고리즘 내부 구현 (객체 상태 관리 및 매칭 logic)
# =====================================================================

def convert_bbox_to_z(bbox): # [x1, y1, x2, y2] 좌표를 칼만 필터 입력 포맷 [x, y, s, r]로 변환합니다.
    w = bbox[2] - bbox[0] # 바운딩 박스의 너비를 계산합니다.
    h = bbox[3] - bbox[1] # 바운딩 박스의 높이를 계산합니다.
    x = bbox[0] + w/2. # 박스 중심의 x 좌표를 계산합니다.
    y = bbox[1] + h/2. # 박스 중심의 y 좌표를 계산합니다.
    s = w * h # 박스의 전체 면적(Scale)을 계산합니다.
    r = w / float(h) # 박스의 가로세로 비율(Aspect Ratio)을 계산합니다.
    return np.array([x, y, s, r]).reshape((4, 1)) # 4x1 형태의 열벡터로 반환합니다.

def convert_x_to_bbox(x, score=None): # 칼만 필터 상태 [x, y, s, r]을 다시 바운딩 박스 [x1, y1, x2, y2]로 변환합니다.
    w = np.sqrt(x[2] * x[3]) # 면적과 비율 정보를 이용해 너비를 역산합니다.
    h = x[2] / w # 면적과 너비 정보를 이용해 높이를 역산합니다.
    if score is None: # 신뢰도 점수가 없는 경우 좌표 4개만 반환합니다.
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    else: # 점수가 포함된 경우 좌표 4개와 점수를 함께 반환합니다.
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

class KalmanBoxTracker(object): # 개별 객체의 추적 상태를 관리하는 칼만 필터 추적기 클래스입니다.
    count = 0 # 모든 추적기에 고유한 ID를 부여하기 위한 클래스 변수입니다.
    def __init__(self, bbox): # 초기 검출 데이터로 새로운 추적기를 초기화합니다.
        self.kf = KalmanFilter(dim_x=7, dim_z=4) # 7개의 상태(위치, 속도 등)와 4개의 관측값을 설정합니다.
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0], 
                              [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]) # 등속도 모델 상태 전이 행렬입니다.
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]]) # 상태를 관측값으로 사상하는 행렬입니다.
        self.kf.R[2:,2:] *= 10. # 측정 노이즈에 대한 신뢰도를 조절합니다.
        self.kf.P[4:,4:] *= 1000. # 초기 속도에 대한 불확실성을 매우 높게 설정합니다.
        self.kf.P *= 10. # 전체적인 오차 공분산의 크기를 조절합니다.
        self.kf.Q[-1,-1] *= 0.01 # 프로세스 노이즈의 가로세로 비율 성분을 조절합니다.
        self.kf.Q[4:,4:] *= 0.01 # 속도 변화에 따른 프로세스 노이즈를 낮게 설정합니다.
        self.kf.x[:4] = convert_bbox_to_z(bbox) # 초기 검출 좌표를 필터의 초기 상태로 입력합니다.
        self.time_since_update = 0 # 마지막 업데이트 이후 경과된 프레임 수입니다.
        self.id = KalmanBoxTracker.count # 현재 추적기에 고유 ID를 할당합니다.
        KalmanBoxTracker.count += 1 # 다음 객체를 위해 카운트를 증가시킵니다.
        self.history = [] # 예측된 궤적들의 기록을 저장하는 리스트입니다.
        self.hits = 0 # 이 추적기가 검출값과 매칭된 총 횟수입니다.
        self.hit_streak = 0 # 연속으로 매칭에 성공한 횟수입니다.
        self.age = 0 # 객체가 생성된 이후 흐른 총 프레임 수입니다.

    def update(self, bbox): # 새로운 검출 정보를 바탕으로 필터 상태를 보정합니다.
        self.time_since_update = 0 # 새로운 관측이 이루어졌으므로 경과 시간을 초기화합니다.
        self.history = [] # 히스토리를 갱신하기 위해 비웁니다.
        self.hits += 1 # 총 히트 수를 1 증가시킵니다.
        self.hit_streak += 1 # 연속 히트 스트릭을 1 증가시킵니다.
        self.kf.update(convert_bbox_to_z(bbox)) # 측정된 박스 좌표로 칼만 필터를 업데이트합니다.

    def predict(self): # 다음 프레임에서의 객체 위치를 칼만 필터로 예측합니다.
        if (self.kf.x[6] + self.kf.x[2]) <= 0: # 면적이 0 이하가 되어 수치 오류가 나는 것을 방지합니다.
            self.kf.x[6] *= 0.0
        self.kf.predict() # 예측 수식을 실행하여 상태를 전이시킵니다.
        self.age += 1 # 추적기의 나이를 증가시킵니다.
        if self.time_since_update > 0: # 업데이트가 되지 않은 프레임이라면 연속 성공 횟수를 0으로 만듭니다.
            self.hit_streak = 0
        self.time_since_update += 1 # 마지막 업데이트 이후 프레임 수를 증가시킵니다.
        self.history.append(convert_x_to_bbox(self.kf.x)) # 예측된 상태를 박스 포맷으로 변환해 저장합니다.
        return self.history[-1] # 가장 최근에 예측된 박스 정보를 반환합니다.

    def get_state(self): # 현재 필터가 추정하고 있는 객체의 최신 위치를 반환합니다.
        return convert_x_to_bbox(self.kf.x)

def iou_batch(bb_test, bb_gt): # 검출 리스트와 추적기 리스트 사이의 IoU를 한꺼번에 계산합니다.
    in_xy1 = np.maximum(bb_test[:, None, :2], bb_gt[:, :2]) # 교차 영역의 좌상단 좌표 행렬을 구합니다.
    in_xy2 = np.minimum(bb_test[:, None, 2:4], bb_gt[:, 2:4]) # 교차 영역의 우하단 좌표 행렬을 구합니다.
    inter_area = np.maximum(in_xy2 - in_xy1, 0).prod(axis=2) # 교차 영역의 가로x세로 넓이를 구합니다.
    test_area = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1]) # 검출 박스들의 넓이입니다.
    gt_area = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1]) # 기존 추적 박스들의 넓이입니다.
    union_area = test_area[:, None] + gt_area - inter_area # 합집합 영역의 넓이를 계산합니다.
    return inter_area / union_area # IoU 값(교차영역/합집합영역) 행렬을 반환합니다.

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3): # 검출값과 추적기를 최적으로 할당합니다.
    if len(trackers) == 0: # 기존에 추적 중인 객체가 없다면 모든 검출값은 미매칭으로 처리합니다.
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers) # 모든 검출값과 추적기 사이의 IoU 행렬을 생성합니다.
    
    if min(iou_matrix.shape) > 0: # 할당할 대상이 있는 경우 헝가리안 알고리즘을 사용합니다.
        a = (iou_matrix > iou_threshold).astype(np.int32) # 임계값 미만의 낮은 IoU는 매칭 후보에서 제외합니다.
        if a.sum(1).max() == 1 and a.sum(0).max() == 1: # 매칭 관계가 일대일로 명확한 경우입니다.
            matched_indices = np.stack(np.where(a), axis=1)
        else: # 복잡한 경우 linear_sum_assignment를 통해 최적의 짝을 찾습니다.
            matched_indices = np.array(list(zip(*linear_sum_assignment(-iou_matrix))))
    else: # 매칭할 데이터가 존재하지 않는 경우 빈 배열을 만듭니다.
        matched_indices = np.empty(shape=(0,2))
        
    unmatched_detections = [] # 매칭되지 못한 새로운 검출 객체의 인덱스 리스트입니다.
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = [] # 매칭되지 못한 기존 추적기들의 인덱스 리스트입니다.
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    matches = [] # 최종 확정된 매칭 결과 쌍입니다.
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold: # 매칭되었어도 IoU가 기준치보다 낮으면 취소합니다.
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0: # 확정된 매칭이 없으면 빈 배열을 반환합니다.
        matches = np.empty((0, 2), dtype=int)
    else: # 매칭 결과가 있으면 하나로 묶어 반환합니다.
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object): # 다중 객체 추적 시스템을 총괄적으로 관리하는 클래스입니다.
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3): # 추적 정책 설정을 초기화합니다.
        self.max_age = max_age # 객체가 사라진 후 추적을 유지할 최대 프레임 수입니다.
        self.min_hits = min_hits # 추적을 정식으로 인정하기 위해 필요한 최소 검출 횟수입니다.
        self.iou_threshold = iou_threshold # 매칭 판단의 기준이 되는 IoU 임계값입니다.
        self.trackers = [] # 현재 시스템 내에서 활성화된 추적기 객체 리스트입니다.
        self.frame_count = 0 # 지금까지 처리한 총 프레임 수입니다.

    def update(self, dets=np.empty((0, 5))): # 매 프레임의 검출 결과를 입력받아 추적 상태를 갱신합니다.
        self.frame_count += 1 # 프레임 카운트를 1 증가시킵니다.
        trks = np.zeros((len(self.trackers), 5)) # 기존 추적기들의 예측 위치를 저장할 배열입니다.
        to_del = [] # 제거해야 할 유효하지 않은 추적기 인덱스 리스트입니다.
        ret = [] # 최종적으로 사용자에게 반환할 추적 데이터입니다.
        for t, trk in enumerate(trks): # 관리 중인 모든 추적기에 대해 다음 위치를 예측합니다.
            pos = self.trackers[t].predict()[0] # 칼만 필터 예측을 수행하여 새로운 위치를 얻습니다.
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0] # 예측 좌표를 저장합니다.
            if np.any(np.isnan(pos)): # 예측값이 비정상적(NaN)인 경우 삭제 대상으로 분류합니다.
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # 유효하지 않은 데이터 행을 제거합니다.
        for t in reversed(to_del): # 뒤에서부터 인덱스를 추적하여 삭제 대상 추적기를 리스트에서 제거합니다.
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold) # 매칭을 수행합니다.
        
        for m in matched: # 매칭 성공 시 해당 추적기를 새로운 검출값으로 업데이트합니다.
            self.trackers[m[1]].update(dets[m[0], :])
        for i in unmatched_dets: # 매칭되지 않은 새 객체는 새로운 추적기로 등록합니다.
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers): # 활성 추적기 중 최종 출력할 대상을 선별합니다.
            d = trk.get_state()[0]
            # 최근에 관측되었고 충분히 신뢰할 수 있는 히트 수를 확보한 경우 결과를 반환 리스트에 넣습니다.
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age: # 너무 오랫동안 나타나지 않은 추적기는 시스템에서 제거합니다.
                self.trackers.pop(i)
        if len(ret) > 0: # 유효한 추적 결과가 존재하면 하나로 합쳐 반환합니다.
            return np.concatenate(ret)
        return np.empty((0, 5)) # 추적 결과가 없으면 빈 배열을 반환합니다.

# =====================================================================
# 2. YOLOv3 검출기 로드 및 실행부
# =====================================================================

# 미리 학습된 YOLOv3 가중치 파일과 구성 파일을 로드합니다.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames() # 모델의 모든 레이어 이름을 리스트로 가져옵니다.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # 모델의 출력 레이어 정보를 설정합니다.

# 위에서 구현한 SORT 추적 관리 시스템을 초기화합니다.
mot_tracker = Sort()

# 분석할 동영상 파일을 불러옵니다.
cap = cv2.VideoCapture("slow_traffic_small.mp4")

while cap.isOpened(): # 영상이 정상적으로 열려있는 동안 반복 실행합니다.
    ret, frame = cap.read() # 영상에서 프레임을 한 장씩 읽어옵니다.
    if not ret: # 더 이상 읽을 프레임이 없으면 루프를 빠져나옵니다.
        break

    height, width, channels = frame.shape # 프레임의 세로, 가로, 채널 정보를 얻습니다.

    # 이미지를 416x416 크기의 블롭(Blob) 데이터로 변환하여 YOLO 입력용으로 준비합니다.
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob) # 변환된 블롭 데이터를 신경망 모델의 입력으로 설정합니다.
    outs = net.forward(output_layers) # 전방향 추론을 통해 객체 검출 결과를 얻습니다.

    class_ids, confidences, boxes = [], [], [] # 검출된 정보를 저장할 리스트들입니다.

    for out in outs: # 각 출력 레이어에서 나온 결과에 대해 반복합니다.
        for detection in out: # 검출된 각 객체 후보군에 대해 반복합니다.
            scores = detection[5:] # 클래스별 확률값들을 추출합니다.
            class_id = np.argmax(scores) # 가장 높은 확률을 가진 클래스 ID를 찾습니다.
            confidence = scores[class_id] # 해당 클래스의 확률값을 저장합니다.
            
            if confidence > 0.5: # 신뢰도가 0.5 이상인 유효한 객체만 처리합니다.
                center_x = int(detection[0] * width) # 중심 x 좌표를 복원합니다.
                center_y = int(detection[1] * height) # 중심 y 좌표를 복원합니다.
                w = int(detection[2] * width) # 너비를 복원합니다.
                h = int(detection[3] * height) # 높이를 복원합니다.
                x = int(center_x - w / 2) # 박스의 좌상단 x 좌표를 계산합니다.
                y = int(center_y - h / 2) # 박스의 좌상단 y 좌표를 계산합니다.

                boxes.append([x, y, w, h]) # 박스 리스트에 추가합니다.
                confidences.append(float(confidence)) # 신뢰도 리스트에 추가합니다.
                class_ids.append(class_id) # 클래스 리스트에 추가합니다.

    # 겹치는 여러 박스 중 가장 신뢰도가 높은 것만 남기는 NMS 연산을 수행합니다.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    dets = [] # SORT에 전달할 형식으로 가공된 검출 리스트입니다.
    if len(indexes) > 0: # 유효하게 남은 검출 박스가 있다면
        for i in indexes.flatten(): # 각 인덱스를 꺼내어
            x, y, w, h = boxes[i]
            dets.append([x, y, x + w, y + h, confidences[i]]) # [x1, y1, x2, y2, score] 포맷으로 저장합니다.
    dets = np.array(dets) # 넘파이 배열 형태로 최종 변환합니다.

    # SORT 추적기를 업데이트하여 현재 프레임에서 추적 중인 객체의 ID와 좌표를 얻습니다.
    if len(dets) > 0:
        trackers = mot_tracker.update(dets)
    else: # 검출된 것이 없어도 내부 칼만 필터 예측 유지를 위해 빈 배열을 전달합니다.
        trackers = mot_tracker.update(np.empty((0, 5)))

    # 추적 결과(박스와 고유 ID)를 화면에 시각화합니다.
    for d in trackers: # 현재 유지 중인 모든 추적 객체에 대해 반복합니다.
        x1, y1, x2, y2, track_id = [int(i) for i in d] # 정수형 좌표와 할당된 고유 ID를 가져옵니다.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 객체 주위에 초록색 박스를 그립니다.
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 박스 상단에 ID를 표시합니다.

    # 최종 추적 결과가 반영된 프레임을 화면 창에 띄웁니다.
    cv2.imshow("Multi-Object Tracking (Standalone)", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: # 사용자가 'ESC' 키를 누르면 루프를 중단합니다.
        break

cap.release() # 영상 캡처 자원을 시스템에 반납합니다.
cv2.destroyAllWindows() # 생성된 모든 윈도우 창을 닫습니다.

💡 핵심 함수 정리
Sort.update(dets): 현재 프레임의 모든 검출 정보(dets)를 입력받아 내부의 칼만 필터 상태를 갱신하고, 최종적으로 추적 중인 객체들의 좌표와 고유 ID 리스트를 반환하는 메인 관리 함수입니다.

KalmanFilter: 객체의 이전 상태와 물리적 이동 모델(등속도 등)을 이용해 다음 위치를 확률적으로 예측하고, 실제 측정값과의 오차를 최소화하며 상태를 보정하는 필터 알고리즘입니다.

linear_sum_assignment: 이전 프레임의 추적기와 현재 프레임의 검출값 사이의 거리(또는 -IoU) 합이 최소가 되도록 짝을 짓는 최적 할당(헝가리안 알고리즘) 함수입니다.

cv2.dnn.NMSBoxes: 동일한 객체 주변에 중복으로 생성된 여러 바운딩 박스를 신뢰도와 겹침 정도를 기준으로 정리하여 가장 적합한 하나만 남기는 전처리 함수입니다.

# [컴퓨터비전] L06. Face Mesh 및 Landmarker 실습 과제

본 레포지토리는 **Mediapipe**의 최신 **Face Landmarker API**와 **OpenCV**를 활용하여 이미지에서 얼굴의 468개(또는 그 이상)의 특징점(Landmarks)을 정밀하게 추출하고, 이를 시각화하는 실습 과제의 결과물을 담고 있습니다.

---

## 02. Mediapipe를 이용한 얼굴 랜드마크 검출 및 시각화

### 📌 과제 설명

Google의 **Mediapipe Face Landmarker** 모델을 사용하여 사진 속 얼굴에서 고밀도 랜드마크를 검출합니다. 특히 Windows 환경의 한글 경로 버그를 방지하기 위해 모델 파일을 바이너리 버퍼(`model_asset_buffer`)로 읽어오는 방식을 채택하였습니다. 검출된 정규화 좌표를 이미지 크기에 맞게 픽셀 단위로 변환한 뒤, OpenCV를 통해 특징점들을 초록색 점으로 시각화하여 최종 결과물을 생성합니다.

### 🖼️ 중간 및 최종 결과물

![face_result](https://github.com/user-attachments/assets/5399e13e-c239-4034-b93c-73f8d60a37d5)




### 💻 소스 코드

```python
import os # 운영체제 기능을 사용하여 파일 경로 등을 처리하기 위한 라이브러리입니다.
import urllib.request # 네트워크를 통해 모델 파일을 다운로드하기 위해 사용하는 모듈입니다.
import cv2 # 이미지 처리 및 시각화를 위한 OpenCV 라이브러리입니다.
import mediapipe as mp # 구글의 머신러닝 솔루션인 Mediapipe 라이브러리입니다.
from mediapipe.tasks.python import BaseOptions # 모델 로드 설정을 위한 기본 옵션 클래스입니다.
from mediapipe.tasks.python.vision import ( # 시각 지능 작업을 위한 세부 클래스들을 불러옵니다.
    FaceLandmarker, # 얼굴 랜드마크 검출 메인 클래스입니다.
    FaceLandmarkerOptions, # 검출기 설정 옵션 클래스입니다.
    RunningMode, # 작업 모드(이미지, 비디오, 라이브 스트림)를 설정하는 클래스입니다.
)

# 1. 모델 파일 다운로드 및 경로 설정
MODEL_URL = "[https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)" # 모델 파일의 저장소 주소입니다.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task") # 현재 실행 파일과 같은 위치에 모델 경로를 설정합니다.
IMAGE_FILE = "face_test.jpg" # 분석할 원본 이미지 파일의 이름입니다.

def download_model(): # 모델 파일이 로컬에 없는 경우 자동으로 다운로드하는 함수입니다.
    if not os.path.exists(MODEL_PATH): # 지정된 경로에 모델 파일이 존재하는지 확인합니다.
        print("모델 파일이 없습니다. 다운로드를 시작합니다...") # 안내 메시지를 출력합니다.
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH) # URL로부터 파일을 다운로드하여 저장합니다.
        print("다운로드 완료.") # 완료 메시지를 출력합니다.

def main(): # 메인 실행 로직입니다.
    download_model() # 프로그램 시작 전 모델 파일 존재 여부를 체크합니다.

    if not os.path.exists(IMAGE_FILE): # 입력 이미지 파일이 있는지 확인합니다.
        print(f"에러: {IMAGE_FILE} 파일이 없습니다.") # 파일이 없으면 에러 메시지를 출력합니다.
        return # 프로그램을 종료합니다.

    # 2. 한글 경로 버그 방지를 위한 모델 데이터 로드
    # MediaPipe 엔진(C++)이 한글이나 공백이 포함된 경로를 직접 읽지 못하는 문제를 해결하기 위해 파이썬이 먼저 파일을 읽습니다.
    with open(MODEL_PATH, "rb") as f: # 모델 파일을 바이너리 읽기 모드로 엽니다.
        model_data = f.read() # 파일의 전체 데이터를 메모리(버퍼)에 로드합니다.

    # 3. FaceLandmarker 설정
    options = FaceLandmarkerOptions( # 검출기 동작을 위한 세부 설정을 정의합니다.
        base_options=BaseOptions(model_asset_buffer=model_data), # 경로 문자열 대신 읽어온 버퍼 데이터를 직접 전달합니다.
        running_mode=RunningMode.IMAGE, # 정지된 이미지 한 장을 처리하는 모드로 설정합니다.
        num_faces=1, # 검출할 최대 얼굴 개수를 1개로 제한합니다.
        min_face_detection_confidence=0.5, # 얼굴 검출을 확신할 최소 임계값입니다.
    )
    
    # 설정된 옵션을 바탕으로 랜드마크 검출기 객체를 생성합니다.
    landmarker = FaceLandmarker.create_from_options(options)

    # 4. 이미지 전처리
    image = cv2.imread(IMAGE_FILE) # OpenCV를 통해 이미지 파일을 읽어옵니다.
    if image is None: # 이미지를 제대로 읽지 못했을 경우의 예외 처리입니다.
        print("이미지를 로드할 수 없습니다.") # 에러 메시지를 출력합니다.
        return # 프로그램을 종료합니다.
    
    h, w, _ = image.shape # 이미지의 높이(h)와 너비(w) 정보를 가져옵니다.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV의 BGR 형식을 Mediapipe의 RGB 형식으로 변환합니다.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image) # Mediapipe 전용 이미지 객체로 변환합니다.

    # 5. 랜드마크 검출 수행
    print("랜드마크 추출 중...") # 작업 시작 메시지를 출력합니다.
    result = landmarker.detect(mp_image) # 이미지에서 랜드마크를 검출하고 결과를 반환받습니다.

    # 6. 결과 시각화
    if result.face_landmarks: # 검출된 랜드마크 데이터가 존재하는지 확인합니다.
        print("추출 성공! 시각화를 시작합니다.") # 성공 메시지를 출력합니다.
        for face_landmarks in result.face_landmarks: # 검출된 각 얼굴의 랜드마크 리스트에 대해 반복합니다.
            for landmark in face_landmarks: # 얼굴 내 개별 랜드마크 점들에 대해 반복합니다.
                # 0.0 ~ 1.0 사이의 정규화된 좌표를 이미지의 실제 픽셀 좌표로 변환합니다.
                x = int(landmark.x * w) # x 좌표 비율에 너비를 곱합니다.
                y = int(landmark.y * h) # y 좌표 비율에 높이를 곱합니다.
                # OpenCV를 사용하여 해당 위치에 작은 초록색 점(반지름 1)을 그립니다.
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1) 
    else: # 얼굴을 찾지 못한 경우의 처리입니다.
        print("얼굴을 검출하지 못했습니다.") # 실패 메시지를 출력합니다.

    # 7. 결과 출력 및 파일 저장
    cv2.imshow("Face Landmark Result", image) # 결과 이미지를 창에 표시합니다.
    cv2.imwrite("result_landmark.jpg", image) # 시각화된 결과를 이미지 파일로 저장합니다.
    
    print("아무 키나 누르면 종료됩니다.") # 종료 안내 메시지입니다.
    cv2.waitKey(0) # 사용자가 아무 키나 누를 때까지 창을 유지합니다.
    cv2.destroyAllWindows() # 모든 OpenCV 창을 닫습니다.
    landmarker.close() # 사용한 검출기 자원을 해제합니다.

if __name__ == "__main__": # 스크립트가 직접 실행될 때 main 함수를 호출합니다.
    main()

💡 핵심 함수 정리
FaceLandmarker.create_from_options(options): 설정된 옵션(모델 버퍼, 실행 모드 등)을 바탕으로 얼굴 랜드마크 검출 인스턴스를 생성합니다.
model_asset_buffer: 모델 파일의 경로 대신 바이너리 데이터를 직접 전달하여, Windows 시스템의 한글/공백 경로 인식 오류를 해결하는 핵심 옵션입니다.
landmarker.detect(mp_image): 입력된 이미지에서 얼굴 특징점 좌표를 추출합니다. 결과는 정규화된 $x, y, z$ 좌표 리스트로 반환됩니다.
좌표 변환: landmark.x * w와 같이 정규화 좌표에 이미지 크기를 곱하여 실제 화면 좌표(Pixel)를 얻습니다.

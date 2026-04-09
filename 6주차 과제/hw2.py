import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# 1. 모델 및 이미지 설정
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# 테스트할 이미지 경로 (여기에 본인의 이미지 파일명을 적으세요)
IMAGE_FILE = "person.png" 

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("모델 다운로드 완료.")

def main():
    download_model()

    # 이미지 파일 존재 확인
    if not os.path.exists(IMAGE_FILE):
        print(f"에러: '{IMAGE_FILE}' 파일을 찾을 수 없습니다. 이미지 파일 이름을 확인해 주세요.")
        return

    # 2. 파일 버퍼 읽기 (한글 경로 버그 방지)
    with open(MODEL_PATH, "rb") as f:
        model_data = f.read()

    # 3. FaceLandmarker 설정 (IMAGE 모드로 변경)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=RunningMode.IMAGE, # 실시간 스트림 대신 이미지 모드 사용
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    
    landmarker = FaceLandmarker.create_from_options(options)

    # 4. 이미지 로드 및 변환
    image = cv2.imread(IMAGE_FILE)
    if image is None:
        print("에러: 이미지를 불러올 수 없습니다.")
        return
    
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 5. 랜드마크 검출 (동기 방식)
    print("랜드마크 검출 중...")
    result = landmarker.detect(mp_image)

    # 6. 시각화
    if result.face_landmarks:
        print(f"성공: {len(result.face_landmarks[0])}개의 랜드마크를 찾았습니다.")
        for face_landmarks in result.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    else:
        print("얼굴을 검출하지 못했습니다.")

    # 7. 결과 출력 및 저장
    cv2.imshow("Face Landmark - Static Image", image)
    cv2.imwrite("result_landmark.jpg", image) # 결과 저장
    print("결과가 'result_landmark.jpg'로 저장되었습니다.")
    
    cv2.waitKey(0) # 아무 키나 누르면 종료
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
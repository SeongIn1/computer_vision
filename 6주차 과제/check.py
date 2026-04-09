import mediapipe
import os

print(f"현재 인식된 mediapipe 경로: {mediapipe.__file__}")
print(f"현재 작업 디렉토리 파일 목록: {os.listdir('.')}")